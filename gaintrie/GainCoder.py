from typing import NamedTuple, List

import numpy as np
from pathlib import Path

import sys

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import Bitty, NBitArray, SliceView, get_bits, get_number, symbol_to_str
from gaintrie.gain_stats import print_stats, plot_bit_definition_order, plot_strait_counts
from gaintrie.tree_printer import TreePrinter, Char
from gaintrie.tree_printer import RootBegin, NodeBegin, Strait, NodeSplit, RootSplit, NodeEnd, Leaf
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import size_report

# Bit positions are MSB-first (clarautils) internally; `idx`/`bit_idx` in the
# tree and the display stay LSB-relative to the node's remaining bits (v0).


class DefineBitOp(NamedTuple):
    idx: int
    bit: 1 | 0

    def __repr__(s):
        return f"op:{s.idx}={s.bit}"


class RunOp(NamedTuple):
    abs_pos: int
    bit: 1 | 0
    kind: str
    level: int


class EntropyDiff(NamedTuple):
    entropy_before: np.array
    entropy_after: np.array
    idx: int


class Split(NamedTuple):
    bit_idx: int
    gains: np.ndarray
    entropy_begin: np.ndarray
    entropy_after_with: np.ndarray
    entropy_after_wout: np.ndarray


class Node(NamedTuple):
    bit_idx: int | None
    true_node: NamedTuple | None | np.generic
    false_node: NamedTuple | None | np.generic


class StraitNode(NamedTuple):
    op: DefineBitOp
    child: "StraitNode | Node | np.generic"


def lsb_entropy(view: NBitArray):
    # LSB-first and contiguous: keeps v0's summation order (split tie-breaking)
    return np.ascontiguousarray(view.get_bitwise_entropy()[::-1])


def find_split(view: NBitArray) -> Split:
    width = view.get_bit_count()
    begin = lsb_entropy(view)
    begin_sum = sum(begin)
    max_g, best, gains = 0, (0, [], []), []
    for i in range(width):
        groups = view.group_by_bit(width - 1 - i)
        n_with, n_wout = groups[1].get_item_count(), groups[0].get_item_count()
        with_e, wout_e = lsb_entropy(groups[1]), lsb_entropy(groups[0])
        gain = ((begin_sum - np.sum(with_e)) * n_with + (begin_sum - np.sum(wout_e)) * n_wout) / (n_with + n_wout)
        if gain > max_g:
            max_g, best = gain, (i, with_e, wout_e)
        gains.append(gain)
    return Split(best[0], np.array(gains), begin, best[1], best[2])


class GainCoder:
    def __init__(self, values, counts, bit_count, display=None):
        self.bit_count = int(bit_count)
        self.values = np.array(values)
        self.counts = np.array(counts)
        self.display = display
        self.root = Bitty(self.values, max_bit=self.bit_count)
        self.codes = {}
        self.runs: List[tuple] = []
        self.straits: List[RunOp] = []
        self.depths = []
        self.node, self.tree = self._build(SliceView(self.root))
        self.avg_bits = np.average(self.depths)

    def emit(self, event):
        if self.display is not None:
            self.display.handle(event)

    def pattern(self, ref, unknown, mark=None):
        w = self.bit_count
        return "".join(Char.branch if p == mark
                       else Char.fill if p in unknown
                       else str((ref >> (w - 1 - p)) & 1)
                       for p in range(w))

    def _build(self, view: SliceView, level=0, code='', run=(), op=None, entropy=None):
        is_root = op is None
        name = "root" if is_root else f"lvl:{level + 1},{op}"
        ref = int(self.root.get_array()[view.get_item_indices()[0]])
        unknown = set(view.get_bit_indices())
        value = self.pattern(ref, unknown)
        value_in = None if is_root else self.pattern(ref, unknown, mark=run[-1].abs_pos)

        if view.get_item_count() == 1:
            return self._leaf(view, ref, name, value_in, value, level, code, run)

        self.emit(RootBegin(value) if is_root else NodeBegin(name, value_in, value, entropy))

        width, positions = view.get_bit_count(), view.get_bit_indices()
        strait_ops = []
        defined = view.get_defined_bits()
        for d in defined:
            level += 1
            strait_op = DefineBitOp(width - 1 - d.idx, d.bit)
            unknown.discard(positions[d.idx])
            after = self.pattern(ref, unknown)
            self.emit(Strait(strait_op, value, after))
            value = after
            strait_run = RunOp(positions[d.idx], d.bit, 'strait', level)
            self.straits.append(strait_run)
            run += (strait_run,)
            strait_ops.append(strait_op)
        if defined:
            view = view.rm_b([d.idx for d in defined])

        split = find_split(view)
        if is_root:
            self.root_split_idx = split.bit_idx  # root Node keeps bit_idx=None (v0 parity)
        msb =view.get_bit_count() - 1 - split.bit_idx
        split_pos = view.get_bit_indices()[msb]
        out = self.pattern(ref, unknown, mark=split_pos)
        self.emit(RootSplit(value, out, np.max(split.gains)) if is_root
                  else NodeSplit(split.bit_idx, value, out))

        groups = view.group_by_bit(msb)
        (t_node, t_tree), (f_node, f_tree) = [
            self._build(groups[bit], level + 1,
                        code + str(bit),
                        run + (RunOp(split_pos, bit, 'split', level + 1),),
                        DefineBitOp(split.bit_idx, bit),
                        EntropyDiff(split.entropy_begin, after_e, split.bit_idx))
            for bit, after_e in ((1, split.entropy_after_with), (0, split.entropy_after_wout))]

        bit_idx = None if is_root else split.bit_idx
        node, tree = Node(bit_idx, t_node, f_node), Node(bit_idx, t_tree, f_tree)
        for strait_op in reversed(strait_ops):
            tree = StraitNode(strait_op, tree)
        self.emit(NodeEnd(name))
        return node, tree

    def _leaf(self, view, ref, name, value_in, value, level, code, run):
        leaf_value = view.get_array()[0]
        leaf_bits = get_bits(leaf_value, view.get_bit_count())
        full = self.pattern(ref, ())
        self.emit(Leaf(name, value_in, value, full, leaf_value, symbol_to_str(leaf_bits)))
        number = get_number(get_bits(full))
        self.codes[number.value] = code
        self.depths.append(level + 1)
        self.runs.append(run)
        return get_number(leaf_bits), number

    def get_avg_bits(self):
        return self.avg_bits

    def average_bits(self):
        total = sum(self.counts)
        return sum(len(self.codes[v]) * c for v, c in zip(np.array(self.values), self.counts)) / total

    def compression_ratio(self, original_bits=16):
        return self.average_bits() / original_bits

    def print(self):
        if self.display is not None:
            self.display.print()


def parse_from_np_array(x, bits_to_take, name):
    values, counts = np.unique(x, return_counts=True)

    num_unique = len(values)
    ratio = num_unique / (np.iinfo(np.uint32).max + 1)
    bit_req = int(np.ceil(np.log2(num_unique + 1)))
    print(f"{name}: {num_unique}({bit_req:.3f} bits) unique, ratio={ratio:.6f}")

    coder = GainCoder(values, counts, bits_to_take, display=TreePrinter())
    print_stats(coder)

    avg_bits = coder.average_bits()
    ratio_bits = coder.compression_ratio(bits_to_take)

    # coder.print()
    plot_bit_definition_order(coder)
    plot_strait_counts(coder)

    print(f"{name}: avg_bits={avg_bits:.3f}, compression={ratio_bits:.3f}, {avg_bits - bits_to_take:.3f}")
    print("END")
    return coder


if __name__ == "__main__":
    base = sandbox_path("bins")

    bits_to_take = 32
    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        name = path.name
        x = np.frombuffer(path.read_bytes(), dtype=np.uint32)
        coder = parse_from_np_array(x, bits_to_take, name)
        size_report(coder, x)
