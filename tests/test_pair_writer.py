"""Round-trip tests for the PAIR-node tree writer.

Run: python tests\\test_pair_writer.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import encode as gain_encode
from gaintrie.code_utils import huffman_lengths, canonical_codes
from gaintrie.pair_writer import (PairWriter, encode, decode, RAW, COMB, TREE_CODES, HUFFMAN, AUTO,
                                  POS_MSB, POS_ORDER, POS_AUTO)
from tests.test_tree_writer import CASES, make


def test_round_trip():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        for mode in (RAW, COMB):
            for stream in (TREE_CODES, HUFFMAN, AUTO):
                for pos in (POS_MSB, POS_ORDER, POS_AUTO):
                    data, _ = encode(PairWriter(coder, mode), x, stream, pos)
                    assert np.array_equal(decode(data), x.astype(np.uint64)), (bit_count, mode, stream, pos)


def test_stream_unchanged():
    # a pair keeps the two items one branch bit apart, so every code keeps its length
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        _, stats = encode(PairWriter(coder), x, TREE_CODES)
        assert stats["stream"] == gain_encode(coder, x)[1]["stream"]


def test_pos_auto_never_worse():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        sizes = {p: sum(encode(PairWriter(coder), x, AUTO, p)[1].values())
                 for p in (POS_MSB, POS_ORDER, POS_AUTO)}
        assert sizes[POS_AUTO] == min(v for k, v in sizes.items() if k != POS_AUTO), sizes


def test_canonical_codes():
    rng = np.random.default_rng(5)
    for n in (1, 2, 3, 17, 300):
        counts = rng.geometric(0.05, n).tolist()
        lengths = huffman_lengths(counts)
        codes = canonical_codes(lengths)
        assert [len(c) for c in codes] == lengths
        ordered = sorted(codes)
        assert len(set(codes)) == n and all(not b.startswith(a) for a, b in zip(ordered, ordered[1:]))


def test_auto_never_worse():
    for x, bit_count in CASES:
        coder = make(x, bit_count)
        sizes = {s: sum(encode(PairWriter(coder), x, s)[1].values()) for s in (TREE_CODES, HUFFMAN, AUTO)}
        assert sizes[AUTO] == min(sizes[TREE_CODES], sizes[HUFFMAN]), sizes


def test_real_data():
    path = sandbox_path("bins", "model.layers.0.input_layernorm.weight.bin")
    if not path.exists():
        print("test_real_data: SKIPPED (bin file missing)")
        return
    for dtype, bits in ((np.uint16, 16), (np.uint32, 32)):
        x = np.frombuffer(path.read_bytes(), dtype=dtype)
        coder = make(x, bits)
        base = sum(gain_encode(coder, x)[1].values())
        for mode in (RAW, COMB):
            for stream in (TREE_CODES, HUFFMAN):
                data, stats = encode(PairWriter(coder, mode), x, stream)
                assert np.array_equal(decode(data), x.astype(np.uint64))
                print(f"-- {path.name} {bits}b mask={mode} stream={stream}: {sum(stats.values())} vs GainCoder {base}")


TESTS = [test_round_trip, test_stream_unchanged, test_pos_auto_never_worse, test_canonical_codes,
         test_auto_never_worse, test_real_data]

if __name__ == "__main__":
    for t in TESTS:
        with contextlib.redirect_stdout(io.StringIO()) if t is not test_real_data else contextlib.nullcontext():
            t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
