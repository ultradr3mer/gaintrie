"""GainCoder variant: the bits a node leaves undefined are predicted by a flip
mask (FlipSort idea); straits are optional, 2-item subtrees can be PAIR nodes.

Same splits as GainCoder. Every node carries `rem` (MSB positions still
undefined) and `mask` (predicted bit per position, all 0 at the root). A flip
node toggles mask[pos] for its subtree; a leaf stores only its exceptions, the
rem positions where the value differs from mask.

Hybrid: a GainCoder strait is kept (pos + bit relative to mask, removes pos
from rem like in GainCoder) when its subtree has at least
`strait_min_leaves` leaves, otherwise mask + exceptions cover it.
None = no straits at all.

Pairs (pair_writer idea): a 2-item split becomes one PAIR node: diff mask over
rem + the "1" item's bits at rem without the branch position (= first diff),
either raw ("raw") or as exceptions against the mask ("exc").

Flip placement:
  float t  probability switch with hysteresis: flip when P(B != mask) > t
           (t=0.5 is the plain majority switch)
  "dp"     per position the flip set minimizing flips + exceptions

Layout (BFS, nodes dock by order like tree_writer):
  header  N:32  width-1:6  mask_mode:1  value_mode:1  used kinds-1:3  kinds by frequency:3 each
  tree    type   prefix code 0 / 10 / 110 / ... by frequency, skipped when rem is empty (leaf)
          strait index into rem + bit ^ mask[pos] (removes pos from rem)
          flip   index into rem
          split  index into rem (removes pos from rem)
          leaf   exception count of len(rem)+1, colex rank of C(rem, count)
          pair   diff mask (raw len(rem) bits, or count-1 of len(rem) + colex rank),
                 "1" item: raw len(rem)-1 bits, or exception count of len(rem) + colex rank
  stream  the code of every item (own table: pairs pick their branch bit)
All indices are truncated binary over their range (code_utils.put_tb).
"""
from collections import Counter, deque
from math import comb
from pathlib import Path
from typing import NamedTuple
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader, build_bins_n_print
from gaintrie.GainCoder import GainCoder
from gaintrie.code_utils import (kind_codes, colex_rank, colex_unrank, tb_bits, put_tb, read_tb,
                                 print_type_sizes, gain_type_sizes)
from gaintrie.pair_writer import pair_split, best_writer as pair_best_writer
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import (walk, kind, pos_bits, bit_of, put_int, put_bits, read_int, bits_written,
                                  encode as gain_encode, SPLIT, STRAIT, LEAF)

FLIP, PAIR = 3, 4
KIND_NAMES = {SPLIT: "split", STRAIT: "strait", LEAF: "leaf", FLIP: "flip", PAIR: "pair"}
RAW, COMB = 0, 1  # pair mask modes
FLIP_TYPE_COST = 2  # dp estimate of the type code share of a flip


class StraitNode(NamedTuple):
    pos: int
    bit: int
    child: "StraitNode | FlipNode | SplitNode | PairNode | FlipLeaf"


class FlipNode(NamedTuple):
    pos: int
    child: "FlipNode | SplitNode | PairNode | FlipLeaf"


class SplitNode(NamedTuple):
    pos: int
    true_node: "StraitNode | FlipNode | SplitNode | PairNode | FlipLeaf"
    false_node: "StraitNode | FlipNode | SplitNode | PairNode | FlipLeaf"


class PairNode(NamedTuple):
    one: int  # item with bit 1 at the branch position, code + "1"
    other: int
    diff: tuple  # indices into rem where the two differ, diff[0] = branch
    exceptions: tuple  # indices into rem without the branch where one != mask


class FlipLeaf(NamedTuple):
    value: int
    exceptions: tuple


def flip_kind(t):
    return {StraitNode: STRAIT, FlipNode: FLIP, SplitNode: SPLIT, PairNode: PAIR}.get(type(t), LEAF)


class FlipGainCoder:
    def __init__(self, coder: GainCoder, flip="dp", strait_min_leaves=16, pairs=True, pair_values="raw",
                 exception_weight=3.0):
        self.coder = coder
        self.width = coder.bit_count
        self.flip = flip
        self.strait_min_leaves = strait_min_leaves
        self.pairs = pairs
        self.pair_values = pair_values
        self.exception_weight = exception_weight
        self.abs_pos = {id(t): rem[msb] for t, rem, msb in walk(coder) if kind(t) != LEAF}
        self.values = {}
        self._collect(coder.tree)
        self.flips_at = self._dp_flips() if flip == "dp" else None
        self.tree = self._build(coder.tree, list(range(self.width)), [0] * self.width)

    @staticmethod
    def _skip(t):
        while kind(t) == STRAIT:
            t = t.child
        return t

    def _keep_strait(self, n_leaves):
        return self.strait_min_leaves is not None and n_leaves >= self.strait_min_leaves

    def _is_pair(self, t):
        return self.pairs and kind(t) == SPLIT and len(self.values[id(t)]) == 2

    def _enter(self, t, rem):
        """-> (node after the strait chain, kept straits [(pos, bit)], rem without them)"""
        kept = []
        head = self._skip(t)
        if self._keep_strait(len(self.values[id(head)])):
            while kind(t) == STRAIT:
                kept.append((self.abs_pos[id(t)], t.op.bit))
                t = t.child
        gone = {p for p, _ in kept}
        return head, kept, [p for p in rem if p not in gone]

    def _collect(self, t):
        t = self._skip(t)
        if kind(t) == LEAF:
            vals = np.array([int(t.value)], dtype=np.uint64)
        else:
            vals = np.concatenate((self._collect(t.true_node), self._collect(t.false_node)))
        self.values[id(t)] = vals
        return vals

    def _bit_means(self, t):
        shifts = np.arange(self.width - 1, -1, -1, dtype=np.uint64)
        return ((self.values[id(t)][:, None] >> shifts) & np.uint64(1)).mean(axis=0)

    def _pair_parts(self, t, rem):
        """-> (one, other, diff, rem without the branch position)"""
        a, b = (int(v) for v in self.values[id(t)])
        diff, one = pair_split(a, b, rem, self.width)
        rest = [p for i, p in enumerate(rem) if i != diff[0]]
        return one, one ^ a ^ b, diff, rest

    def _dp_flips(self):
        """per node and position: flip there iff it lowers flips*cost + exceptions*weight."""
        w, e_w = self.width, self.exception_weight
        flips_at = {}
        inner_of = {}

        def mismatch_cost(value, positions):
            cost = np.zeros((w, 2))
            for p in positions:
                cost[p, 1 - bit_of(value, p, w)] = e_w
            return cost

        def solve(t, rem):  # -> cost[p, m]: subtree cost if mask[p] == m at entry
            t, _, rem = self._enter(t, rem)
            if kind(t) == LEAF:
                return mismatch_cost(int(t.value), rem)
            if self._is_pair(t):
                one, _, _, rest = self._pair_parts(t, rem)
                inner = mismatch_cost(one, rest) if self.pair_values == "exc" else np.zeros((w, 2))
            else:
                sp = self.abs_pos[id(t)]
                crem = [p for p in rem if p != sp]
                inner = solve(t.true_node, crem) + solve(t.false_node, crem)
            inner_of[id(t)] = inner
            f = pos_bits(len(rem)) + FLIP_TYPE_COST
            return np.minimum(inner, inner[:, ::-1] + f)

        def place(t, rem, mask):
            t, _, rem = self._enter(t, rem)
            if kind(t) == LEAF:
                return
            inner, f = inner_of[id(t)], pos_bits(len(rem)) + FLIP_TYPE_COST
            sp = self.abs_pos[id(t)]
            mask = mask.copy()
            for p in rem:
                if p != sp and inner[p, 1 - mask[p]] + f < inner[p, mask[p]]:
                    flips_at.setdefault(id(t), []).append(p)
                    mask[p] ^= 1
            if self._is_pair(t):
                return
            crem = [p for p in rem if p != sp]
            place(t.true_node, crem, mask)
            place(t.false_node, crem, mask)

        solve(self.coder.tree, list(range(w)))
        place(self.coder.tree, list(range(w)), [0] * w)
        return flips_at

    def _flips(self, t, rem, mask):
        sp = self.abs_pos[id(t)]
        if self.flips_at is not None:
            return self.flips_at.get(id(t), [])
        means = self._bit_means(t)
        return [p for p in rem if p != sp and (means[p] if mask[p] == 0 else 1 - means[p]) > self.flip]

    def _build(self, t, rem, mask):
        t, kept, rem = self._enter(t, rem)
        if kind(t) == LEAF:
            v = int(t.value)
            node = FlipLeaf(v, tuple(i for i, p in enumerate(rem) if bit_of(v, p, self.width) != mask[p]))
        else:
            flips = self._flips(t, rem, mask)
            mask = mask.copy()
            for p in flips:
                mask[p] ^= 1
            if self._is_pair(t):
                one, other, diff, rest = self._pair_parts(t, rem)
                exc = tuple(i for i, p in enumerate(rest) if bit_of(one, p, self.width) != mask[p])
                node = PairNode(one, other, tuple(diff), exc)
            else:
                sp = self.abs_pos[id(t)]
                crem = [p for p in rem if p != sp]
                node = SplitNode(sp, self._build(t.true_node, crem, mask), self._build(t.false_node, crem, mask))
            for p in reversed(flips):
                node = FlipNode(p, node)
        for p, b in reversed(kept):
            node = StraitNode(p, b, node)
        return node


def flip_walk(tree, width):
    """BFS yielding (node, rem, level, mask, code) — as seen at the node's entry."""
    queue = deque([(tree, list(range(width)), 0, [0] * width, "")])
    while queue:
        t, rem, level, mask, code = queue.popleft()
        yield t, rem, level, mask, code
        k = flip_kind(t)
        if k == STRAIT:
            queue.append((t.child, [p for p in rem if p != t.pos], level + 1, mask, code))
        elif k == FLIP:
            child_mask = mask.copy()
            child_mask[t.pos] ^= 1
            queue.append((t.child, rem, level + 1, child_mask, code))
        elif k == SPLIT:
            crem = [p for p in rem if p != t.pos]
            queue.extend(((t.true_node, crem, level + 1, mask, code + "1"),
                          (t.false_node, crem, level + 1, mask, code + "0")))


def pair_mask_bits(r, diff, mode):
    d = len(diff)
    return r if mode == RAW else tb_bits(d - 1, r) + tb_bits(colex_rank(diff), comb(r, d))


def write_tree(fc: FlipGainCoder, bw: BitWriter):
    width = fc.width
    nodes = list(flip_walk(fc.tree, width))
    order = [k for k, _ in Counter(flip_kind(t) for t, *_ in nodes).most_common()]
    codes = kind_codes(order)
    pairs = [(t, rem) for t, rem, *_ in nodes if flip_kind(t) == PAIR]
    mask_mode = min((RAW, COMB), key=lambda m: sum(pair_mask_bits(len(r), t.diff, m) for t, r in pairs))
    value_exc = fc.pair_values == "exc"
    stats, table = Counter(), {}
    put_int(bw, width - 1, 6)
    put_int(bw, mask_mode, 1)
    put_int(bw, int(value_exc), 1)
    put_int(bw, len(order) - 1, 3)
    for k in order:
        put_int(bw, k, 3)
    stats["header"] += 11 + 3 * len(order)
    for t, rem, _, mask, code in nodes:
        k = flip_kind(t)
        if rem:
            put_bits(bw, codes[k])
            stats[f"{KIND_NAMES[k]}_type"] += len(codes[k])
        else:
            assert k == LEAF
        r = len(rem)
        if k == LEAF:
            e = len(t.exceptions)
            put_tb(bw, e, r + 1)
            put_tb(bw, colex_rank(t.exceptions), comb(r, e))
            stats["leaf_count"] += tb_bits(e, r + 1)
            stats["leaf_exc"] += tb_bits(colex_rank(t.exceptions), comb(r, e))
            table[t.value] = code
        elif k == PAIR:
            d = len(t.diff)
            if mask_mode == RAW:
                put_bits(bw, (i in t.diff for i in range(r)))
            else:
                put_tb(bw, d - 1, r)
                put_tb(bw, colex_rank(t.diff), comb(r, d))
            stats["pair_mask"] += pair_mask_bits(r, t.diff, mask_mode)
            rest = [p for i, p in enumerate(rem) if i != t.diff[0]]
            if value_exc:
                e = len(t.exceptions)
                put_tb(bw, e, r)
                put_tb(bw, colex_rank(t.exceptions), comb(r - 1, e))
                stats["pair_value"] += tb_bits(e, r) + tb_bits(colex_rank(t.exceptions), comb(r - 1, e))
            else:
                put_bits(bw, (bit_of(t.one, p, width) for p in rest))
                stats["pair_value"] += r - 1
            table[t.one], table[t.other] = code + "1", code + "0"
        else:
            put_tb(bw, rem.index(t.pos), r)
            stats[f"{KIND_NAMES[k]}_pos"] += tb_bits(rem.index(t.pos), r)
            if k == STRAIT:
                bw.put(bool(t.bit ^ mask[t.pos]))
                stats["strait_bit"] += 1
    return stats, table


def read_tree(br: BitReader) -> dict:
    width = read_int(br, 6) + 1
    mask_mode = read_int(br, 1)
    value_exc = read_int(br, 1)
    order = [read_int(br, 3) for _ in range(read_int(br, 3) + 1)]
    shift = lambda p: width - 1 - p
    table = {}
    queue = deque([(list(range(width)), [0] * width, 0, "")])
    while queue:
        rem, mask, value, code = queue.popleft()
        if not rem:
            k = LEAF
        else:
            i = 0
            while i < len(order) - 1 and br.get_bool():
                i += 1
            k = order[i]
        r = len(rem)
        if k == LEAF:
            e = read_tb(br, r + 1)
            exc = set(colex_unrank(read_tb(br, comb(r, e)), e))
            for i, p in enumerate(rem):
                value |= (mask[p] ^ (i in exc)) << shift(p)
            table[code] = value
            continue
        if k == PAIR:
            if mask_mode == RAW:
                diff = [i for i in range(r) if br.get_bool()]
            else:
                d = read_tb(br, r) + 1
                diff = colex_unrank(read_tb(br, comb(r, d)), d)
            one = value | (1 << shift(rem[diff[0]]))
            rest = [p for i, p in enumerate(rem) if i != diff[0]]
            if value_exc:
                e = read_tb(br, r)
                exc = set(colex_unrank(read_tb(br, comb(r - 1, e)), e))
                for i, p in enumerate(rest):
                    one |= (mask[p] ^ (i in exc)) << shift(p)
            else:
                for p in rest:
                    one |= int(br.get_bool()) << shift(p)
            table[code + "1"] = one
            table[code + "0"] = one ^ sum(1 << shift(rem[i]) for i in diff)
            continue
        pos = rem[read_tb(br, r)]
        crem = [p for p in rem if p != pos]
        if k == STRAIT:
            bit = int(br.get_bool()) ^ mask[pos]
            queue.append((crem, mask, value | (bit << shift(pos)), code))
        elif k == FLIP:
            mask = mask.copy()
            mask[pos] ^= 1
            queue.append((rem, mask, value, code))
        else:
            queue.append((crem, mask, value | (1 << shift(pos)), code + "1"))
            queue.append((crem, mask, value, code + "0"))
    return table


def encode(fc: FlipGainCoder, x) -> tuple[np.ndarray, Counter]:
    bw = BitWriter()
    if len(x) >= 1 << 32:
        raise ValueError(f"item count {len(x)} does not fit the 32-bit header field")
    put_int(bw, len(x), 32)
    stats, table = write_tree(fc, bw)
    stats["header"] += 32
    for v in np.asarray(x).tolist():
        put_bits(bw, (c == "1" for c in table[v]))
        stats["stream"] += len(table[v])
    assert sum(stats.values()) == bits_written(bw)
    return bw.get_bytes(), stats


def decode(data) -> np.ndarray:
    br = BitReader(data)
    n = read_int(br, 32)
    table = read_tree(br)
    out = []
    for _ in range(n):
        code = ""
        while code not in table:
            code += "1" if br.get_bool() else "0"
        out.append(table[code])
    return np.array(out, dtype=np.uint64)


def mode_name(fc: FlipGainCoder) -> str:
    pairs = f" pairs={fc.pair_values}" if fc.pairs else " no pairs"
    return f"flip={fc.flip} straits>={fc.strait_min_leaves}{pairs}"


def print_flip_stats(fc: FlipGainCoder):
    nodes = list(flip_walk(fc.tree, fc.width))
    by_kind = {k: [n for n in nodes if flip_kind(n[0]) == k] for k in KIND_NAMES}
    splits, straits, flips = by_kind[SPLIT], by_kind[STRAIT], by_kind[FLIP]
    leafs, pairs = by_kind[LEAF], by_kind[PAIR]
    exc_pos = [rem[i] for t, rem, *_ in leafs for i in t.exceptions]
    hits = sum(t.bit == m[t.pos] for t, _, _, m, _ in straits)

    print(f"==FlipGain {mode_name(fc)}==")
    print("==Data==")
    print("Codes:", len(leafs) + 2 * len(pairs), "Nodes:", len(splits), "Straits:", len(straits),
          f"({hits} predicted by mask)", "Flips:", len(flips), "Pairs:", len(pairs), "Leafs:", len(leafs),
          "Leaf exceptions:", len(exc_pos))
    print("==LeafRem==")
    if leafs:
        build_bins_n_print([len(rem) for _, rem, *_ in leafs], [45, 90, 100])
    print("==LeafExceptions==")
    if leafs:
        build_bins_n_print([len(t.exceptions) for t, *_ in leafs], [45, 90, 100])
    if pairs:
        print("==PairRem==")
        build_bins_n_print([len(rem) for _, rem, *_ in pairs], [45, 90, 100])
        print("==PairDiff==")
        build_bins_n_print([len(t.diff) for t, *_ in pairs], [45, 90, 100])
        print("==PairValueExceptions==")
        build_bins_n_print([len(t.exceptions) for t, *_ in pairs], [45, 90, 100])
    print("==Flags==")
    build_bins_n_print([pos_bits(len(rem)) for _, rem, *_ in splits + straits + flips], [45, 90, 100])
    for title, group in (("Strait", straits), ("Flip", flips)):
        print(f"=={title}Levels==")
        if group:
            build_bins_n_print([lv for _, _, lv, _, _ in group], [45, 90, 100])
        print(f"==Abs{title}s==")
        if group:
            build_bins_n_print([t.pos for t, *_ in group], [45, 90, 100])
    rules = Counter((t.pos, m[t.pos] ^ 1) for t, _, _, m, _ in flips)
    print(f"Flip rules: {len(rules)} unique of {len(flips)}")
    print("Top:", ", ".join(f"{p}->{b}×{c}" for (p, b), c in rules.most_common(5)))
    print("==AbsSplits==")
    if splits:
        build_bins_n_print([t.pos for t, *_ in splits], [45, 90, 100])
    print("==AbsExceptions==")
    if exc_pos:
        build_bins_n_print(exc_pos, [45, 90, 100])


def plot_flip_counts(fc: FlipGainCoder):
    width = fc.width
    counts = np.zeros((4, width), dtype=np.uint32)  # strait, flip->0, flip->1, exceptions
    for t, rem, _, m, _ in flip_walk(fc.tree, width):
        k = flip_kind(t)
        if k == STRAIT:
            counts[0, t.pos] += 1
        elif k == FLIP:
            counts[1 + (m[t.pos] ^ 1), t.pos] += 1
        elif k == LEAF:
            for i in t.exceptions:
                counts[3, rem[i]] += 1
    print(f"Count bins (pos: strait/to0/to1/exceptions): "
          f"{[(p, *map(int, counts[:, p])) for p in range(width) if counts[:, p].any()]}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(width)
    bar = 0.21
    for i, label in enumerate(("strait", "flip to 0", "flip to 1", "leaf exception")):
        ax.bar(positions + (i - 1.5) * bar, counts[i], bar, label=label)
    ax.set_title(f"Straits, flips and exceptions per abs bit pos ({mode_name(fc)})")
    ax.set_xlabel("abs bit pos")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig("flip_counts.png")
    plt.show()


def size_report(fc: FlipGainCoder, x):
    data, stats = encode(fc, x)
    assert np.array_equal(decode(data), np.asarray(x, dtype=np.uint64)), "round trip failed"
    names = ("split", "strait", "flip", "pair", "leaf")
    rows = {name: [0, stats[f"{name}_type"], 0] for name in names}
    rows["split"][2], rows["flip"][2] = stats["split_pos"], stats["flip_pos"]
    rows["strait"][2] = stats["strait_pos"] + stats["strait_bit"]
    rows["pair"][2] = stats["pair_mask"] + stats["pair_value"]
    rows["leaf"][2] = stats["leaf_count"] + stats["leaf_exc"]
    for t, *_ in flip_walk(fc.tree, fc.width):
        rows[KIND_NAMES[flip_kind(t)]][0] += 1
    print(f"(round trip OK, {len(data)} bytes) leaf: count={stats['leaf_count']} exceptions={stats['leaf_exc']}"
          f"  pair: mask={stats['pair_mask']} value={stats['pair_value']}")
    return print_type_sizes(rows, {"stream": stats["stream"], "header": stats["header"]},
                            len(x), fc.width, f"FlipGain {mode_name(fc)}")


def gain_tb_total(coder, base_stats) -> int:
    """GainCoder total with truncated binary positions (computed from the tree, not written)"""
    tb = sum(tb_bits(msb, len(rem)) - pos_bits(len(rem)) for t, rem, msb in walk(coder) if kind(t) != LEAF)
    return sum(base_stats.values()) + tb


if __name__ == "__main__":
    bits_to_take = 32
    flip_modes = (0.7, "dp")
    strait_rules = (None, 4, 8, 16, 32)
    variants = (("flip", False, "raw"), ("flip+pair raw", True, "raw"), ("flip+pair exc", True, "exc"))
    for name in ("model.layers.0.input_layernorm.weight.bin", "model.layers.0.post_attention_layernorm.weight.bin"):
        x = np.frombuffer(sandbox_path("bins", name).read_bytes(), dtype=np.uint32)
        values, counts = np.unique(x, return_counts=True)
        print(f"{name}: {len(values)} unique of {len(x)}")
        coder = GainCoder(values, counts, bits_to_take)
        _, base = gain_encode(coder, x)
        base_total = sum(base.values())
        _, _, pair_stats = pair_best_writer(coder, x)
        totals = {"GainCoder": base_total, "GainCoder+tb": gain_tb_total(coder, base),
                  "PairWriter": sum(pair_stats.values())}

        print(f"==Sweep (total bits)==")
        print(f"{'variant':<17}{'flip':>6}" + "".join(f"{'s>=' + str(k):>9}" for k in strait_rules))
        best = {}
        for label, pairs, values_mode in variants:
            for flip in flip_modes:
                row = []
                for k in strait_rules:
                    fc = FlipGainCoder(coder, flip, k, pairs, values_mode)
                    total = sum(encode(fc, x)[1].values())
                    row.append(total)
                    if label not in best or total < best[label][0]:
                        best[label] = (total, flip, k)
                print(f"{label:<17}{str(flip):>6}" + "".join(f"{t:>9}" for t in row))
        for label, (total, _, _) in best.items():
            totals[label] = total

        label = min(best, key=lambda l: best[l][0])
        _, flip, k = best[label]
        pairs, values_mode = next((p, v) for l, p, v in variants if l == label)
        fc = FlipGainCoder(coder, flip, k, pairs, values_mode)
        print_flip_stats(fc)
        plot_flip_counts(fc)
        size_report(fc, x)
        print_type_sizes(gain_type_sizes(coder), {"stream": base["stream"], "header": base["header"]},
                         len(x), bits_to_take, "GainCoder (baseline)")

        print(f"==Summary {name}== (GainCoder+tb is computed, all others written + decoded)")
        for label, total in totals.items():
            cfg = f"  flip={best[label][1]} s>={best[label][2]}" if label in best else ""
            print(f"  {label:<17} {total:>7}  {total / len(x):.3f} bits/item  "
                  f"{total - base_total:+7d} vs GainCoder{cfg}")
        print("END")
