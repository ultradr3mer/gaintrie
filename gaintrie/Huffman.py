import heapq
import itertools
import sys
from pathlib import Path

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.sandbox_paths import sandbox_path
from clarautils import get_bitmask


class HuffmanCoder:
    def __init__(self, values, counts):
        self.values = values.tolist()
        self.counts = counts.tolist()
        self.tree = None
        self.codes = {}
        self._build()

    def _build(self):
        counter = itertools.count()  # eindeutige IDs

        heap = [(c, next(counter), v) for v, c in zip(self.values, self.counts)]
        heapq.heapify(heap)

        while len(heap) > 1:
            c1, _, n1 = heapq.heappop(heap)
            c2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (c1 + c2, next(counter), (n1, n2)))

        self.tree = heap[0][2]
        self._build_codes(self.tree)

    def _build_codes(self, node, prefix=""):
        if isinstance(node, int):
            self.codes[node] = prefix or "0"
            return
        left, right = node
        self._build_codes(left, prefix + "0")
        self._build_codes(right, prefix + "1")

    def average_bits(self):
        total = sum(self.counts)
        return sum(len(self.codes[v]) * c for v, c in zip(self.values, self.counts)) / total

    def compression_ratio(self, original_bits=16):
        return self.average_bits() / original_bits


def get_bit_count(value: np.unsignedinteger):
    return np.log2(value+1)

def plot_code_length_histogram(coder, title):
    import matplotlib.pyplot as plt
    import numpy as np

    lengths = []
    weights = []

    for v, c in zip(coder.values, coder.counts):
        l = len(coder.codes[v])
        lengths.append(l)
        weights.append(c)

    lengths = np.array(lengths)
    weights = np.array(weights)

    plt.figure()
    plt.hist(lengths, bins=np.arange(lengths.min(), lengths.max() + 2) - 0.5,
             weights=weights)

    plt.xlabel("Code length (bits)")
    plt.ylabel("Total occurrences")
    plt.title(f"Code length distribution: {title}")
    plt.show()

def plot(c,title):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(c, bins=np.min((c.data_bit_count, 256)))
    plt.title(f"Histogram of counts: {title}")
    plt.xlabel("Count frequency")
    plt.ylabel("Number of uint16 values")

    plt.show()


if __name__ == "__main__":
    base = sandbox_path("bins")
    bits_to_take = 8
    bits_to_shift = 7
    mask = get_bitmask(bits_to_take)
    num_possible = np.pow(2,bits_to_take)
    for path in base.glob("model.layers.0*.bin"):
        with open(path, "rb") as f:
            buffer = f.read()

        x = (np.frombuffer(buffer, dtype=np.uint16).astype(np.int32) >> bits_to_shift) & mask

        sparse_counts = np.bincount(
            x,
            minlength=num_possible
        )

        values = np.flatnonzero(sparse_counts)
        counts = sparse_counts[values]

        num_unique = values.size
        ratio = num_unique / num_possible

        bit_req = get_bit_count(num_unique)
        print(f"{path.name}: {num_unique}({bit_req:.3f} bits) unique, ratio={ratio:.6f}")

        coder = HuffmanCoder(values, counts)

        avg_bits = coder.average_bits()
        ratio_bits = coder.compression_ratio(bits_to_take)

        print(f"{path.name}: avg_bits={avg_bits:.3f}, compression={ratio_bits:.3f}, {avg_bits-bits_to_take:.3f}")

        # plot_code_length_histogram(coder, path.name)