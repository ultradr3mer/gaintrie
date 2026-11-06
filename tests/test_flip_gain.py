"""Round-trip tests for FlipGainCoder (flip mask, hybrid straits, pairs).

Run: python tests\\test_flip_gain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from research.FlipGainCoder import FlipGainCoder, encode, decode
from tests.test_tree_writer import CASES, make

VARIANTS = ((False, "raw"), (True, "raw"), (True, "exc"))


def test_round_trip():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        for flip in (0.5, 0.7, "dp"):
            for k in (None, 1, 4, 16):
                for pairs, values in VARIANTS:
                    data, _ = encode(FlipGainCoder(coder, flip, k, pairs, values), x)
                    assert np.array_equal(decode(data), x.astype(np.uint64)), (bit_count, flip, k, pairs, values)


def test_codes_prefix_free():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        codes = sorted(set(encode_table(FlipGainCoder(coder)).values()))
        assert all(not b.startswith(a) for a, b in zip(codes, codes[1:]))


def encode_table(fc):
    from clarautils import BitWriter
    from research.FlipGainCoder import write_tree
    return write_tree(fc, BitWriter())[1]


TESTS = [test_round_trip, test_codes_prefix_free]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
