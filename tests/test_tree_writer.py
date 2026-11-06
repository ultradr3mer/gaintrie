"""Round-trip tests for the layer-wise GainCoder tree writer.

Run: python tests\\test_tree_writer.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.GainCoder import GainCoder
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import encode, decode, read_tree, size_report
from clarautils import BitReader


def make(x, bit_count):
    values, counts = np.unique(x, return_counts=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return GainCoder(values, counts, bit_count)


rng = np.random.default_rng(3)
CASES = [(np.array([0, 2]), 2), (np.array([2, 3, 4, 5, 5, 2]), 3), (np.array([0, 1]), 3),
         (np.array([1, 4, 6, 7, 9, 12, 200, 201, 255, 4, 4]), 8),
         (rng.choice(1 << 16, 300, replace=False), 16),
         (rng.integers(0, 1 << 32, 500, dtype=np.uint64), 32),
         (rng.geometric(0.05, 2000) % 4096, 12),
         # width 64 (header stores width-1); values stay below 2**63 because
         # clarautils.get_defined_bits overflows on the top bit
         (rng.integers(0, 2 ** 63, 400, dtype=np.uint64), 64),
         (np.array([0, 1, (1 << 62) | 5, 2 ** 63 - 1, (1 << 62) | 5], dtype=np.uint64), 64)]


def test_hand_checked_tiny():
    # N=2 | width-1=1 | one=leaf | strait '11' pos '1' bit '0' | split '10' (0 pos bits)
    # | leaves (rem empty, nothing) | stream 0->'0', 2->'1'
    data, stats = encode(make(np.array([0, 2]), 2), np.array([0, 2]))
    assert data.tolist() == [0, 0, 0, 2, 0b00000110, 0b11101001]
    assert stats["stream"] == 2 and stats["leaf"] == 0


def test_round_trip():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        data, _ = encode(coder, x)
        assert np.array_equal(decode(data), x.astype(np.uint64))


def test_read_tree_matches_codes():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        data, _ = encode(coder, x)
        br = BitReader(data)
        br.get_int(32)
        table = read_tree(br)
        assert {v: c for c, v in table.items()} == {int(v): c for v, c in coder.codes.items()}


class FakeLen:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_item_count_limit():
    coder = make(np.array([0, 2]), 2)
    try:
        encode(coder, FakeLen(1 << 32))
    except ValueError:
        pass
    else:
        raise AssertionError("2**32 items must be rejected by the 32-bit header field")
    # the largest count that fits gets past the check (fails later on the fake items)
    try:
        encode(coder, FakeLen((1 << 32) - 1))
    except ValueError:
        raise AssertionError("2**32 - 1 items fit the header field")
    except Exception:
        pass


def test_real_data():
    path = sandbox_path("bins", "model.layers.0.input_layernorm.weight.bin")
    if not path.exists():
        print("test_real_data: SKIPPED (bin file missing)")
        return
    for dtype, bits in ((np.uint16, 16), (np.uint32, 32)):
        x = np.frombuffer(path.read_bytes(), dtype=dtype)
        coder = make(x, bits)
        print(f"-- {path.name} {bits}b")
        size_report(coder, x)  # asserts the round trip


TESTS = [test_hand_checked_tiny, test_round_trip, test_read_tree_matches_codes, test_item_count_limit,
         test_real_data]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
