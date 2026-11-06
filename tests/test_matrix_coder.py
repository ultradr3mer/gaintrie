"""Round-trip tests for the matrix container (global Huffman table, global / hybrid mode).

Run: python tests\\test_matrix_coder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.code_utils import huffman_lengths
from gaintrie.matrix_coder import encode_matrix, decode_matrix, stream_bits, decode_stream, GLOBAL, HYBRID
from tests.test_tree_writer import CASES

rng = np.random.default_rng(7)
EXTRA = [(rng.geometric(0.02, 20000) % 3000, 16),
         (np.concatenate([np.full(300, 5), rng.integers(0, 50, 300)]), 8)]  # constant bins in hybrid


def test_round_trip():
    for x, bits in CASES + EXTRA:
        for mode, size in ((GLOBAL, 1 << 20), (HYBRID, 3), (HYBRID, 64), (HYBRID, 300), (HYBRID, 1 << 20)):
            data, stats = encode_matrix(x, bits, mode, size)
            assert np.array_equal(decode_matrix(data), x.astype(np.uint64)), (bits, mode, size)
            assert len(data) == (sum(stats.values()) + 7) // 8


def test_stream_codec():
    for n_sym in (1, 2, 5, 300):
        counts = rng.geometric(0.01, n_sym).tolist()
        lengths = huffman_lengths(counts)
        symbols = rng.integers(0, n_sym, 50000)
        bits = stream_bits(symbols, lengths)
        assert len(bits) == sum(lengths[s] for s in symbols.tolist())
        assert np.array_equal(decode_stream(bits, len(symbols), lengths), symbols)


TESTS = [test_round_trip, test_stream_codec]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
