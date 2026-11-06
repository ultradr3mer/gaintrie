"""Preconfigured tree tests for GainCoder.

N = split Node(bit_idx, true, false), S = StraitNode(op, child),
leaf = original value. coder.tree is the strait-augmented structure,
coder.node the split-only v0-parity tree (degenerate leaves).

Run: python test_tree.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.GainCoder import GainCoder, Node, StraitNode


def make(values, bit_count):
    values = np.array(values, dtype=np.uint32)
    counts = np.ones(len(values), dtype=np.uint32)
    with contextlib.redirect_stdout(io.StringIO()):
        return GainCoder(values, counts, bit_count)


def plain(tree):
    if isinstance(tree, StraitNode):
        return ("S", tree.op.idx, tree.op.bit, plain(tree.child))
    if isinstance(tree, Node):
        return ("N", tree.bit_idx, plain(tree.true_node), plain(tree.false_node))
    return ("L", int(tree.value))


def test_strait_above_root_split():
    coder = make([0, 2], 2)
    assert plain(coder.tree) == \
        ("S", 0, 0, ("N", None, ("L", 2), ("L", 0)))
    assert coder.codes == {2: "1", 0: "0"}


def test_pure_split_no_straits():
    coder = make([0, 3], 2)
    assert plain(coder.tree) == ("N", None, ("L", 3), ("L", 0))


def test_split_then_subsplit():
    coder = make([0, 2, 3], 2)
    assert plain(coder.tree) == \
        ("N", None, ("L", 3), ("N", 0, ("L", 2), ("L", 0)))


def test_straits_in_both_branches():
    coder = make([2, 3, 4, 5], 3)
    assert plain(coder.tree) == \
        ("N", None,
         ("S", 1, 0, ("N", 0, ("L", 3), ("L", 2))),
         ("S", 1, 1, ("N", 0, ("L", 5), ("L", 4))))
    assert coder.codes == {3: "11", 2: "10", 5: "01", 4: "00"}


def test_root_strait_chain():
    coder = make([0, 1], 3)
    assert plain(coder.tree) == \
        ("S", 2, 0,
         ("S", 1, 0, ("N", None, ("L", 1), ("L", 0))))


def test_node_tree_stays_split_only():
    coder = make([2, 3, 4, 5], 3)
    assert plain(coder.node) == \
        ("N", None,
         ("N", 0, ("L", 0), ("L", 0)),
         ("N", 0, ("L", 0), ("L", 0)))


def test_runs_match_tree():
    coder = make([0, 2], 2)
    assert coder.runs == [
        ((1, 0, "strait", 1), (0, 1, "split", 2)),
        ((1, 0, "strait", 1), (0, 0, "split", 2)),
    ]


CODE_CASES = [([0, 2], 2), ([2, 3, 4, 5], 3), ([0, 1], 3),
              ([1, 4, 6, 7, 9, 12, 200, 201, 255], 8),
              (list(np.random.default_rng(7).choice(1 << 16, 300, replace=False)), 16)]


def decode(tree, code):
    bits = iter(code)
    while isinstance(tree, (Node, StraitNode)):
        tree = tree.child if isinstance(tree, StraitNode) \
            else tree.true_node if next(bits) == "1" else tree.false_node
    assert next(bits, None) is None
    return int(tree.value)


def test_codes_unique_and_prefix_free():
    for values, bit_count in CODE_CASES:
        codes = sorted(make(values, bit_count).codes.values())
        assert len(set(codes)) == len(codes)
        assert not any(b.startswith(a) for a, b in zip(codes, codes[1:]))


def test_code_length_is_split_count():
    for values, bit_count in CODE_CASES:
        coder = make(values, bit_count)
        # runs and codes are both recorded in leaf order
        for run, code in zip(coder.runs, coder.codes.values(), strict=True):
            assert len(code) == sum(op.kind == "split" for op in run)


def test_decode_round_trip():
    for values, bit_count in CODE_CASES:
        coder = make(values, bit_count)
        for v, code in coder.codes.items():
            assert decode(coder.tree, code) == int(v)


def main():
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in sorted(tests):
        fn()
        print(f"{name}: OK")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
