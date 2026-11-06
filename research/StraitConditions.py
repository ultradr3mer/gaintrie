"""Global strait rules: "wherever bits `cmask` have values `cval`, bit p = b",
applicable anywhere in the tree, chosen so that few rules replace many straits.

GainCoder runs unchanged; afterwards every node's definition state (bits
defined on its path + their values) is logged with the AND/OR over its items.
A rule is valid iff every logged node matching its condition has all items
with bit p == b (never defines a bit wrongly). It covers a strait if the
strait's state matches the condition.

Candidates: per (p, b) group, each strait seeds a condition (its full state),
which is grown by intersecting with the other straits of the group (keep the
common bits only) as long as it stays valid. Set cover: greedily pick the rule
with the largest gain = tree cost of the newly covered straits - RULE_COST,
stop when no rule gains anymore. Uncovered straits stay in the tree.
Masks are position masks, MSB-first: pos p <-> bit (width-1-p).
"""
from collections import Counter
from pathlib import Path
from typing import NamedTuple
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import build_bins_n_print
from gaintrie.GainCoder import GainCoder
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import kind, type_codes, pos_bits, SPLIT, STRAIT, LEAF


def rule_cost(width) -> int:
    """mask + value + output bit + output position"""
    return 2 * width + 1 + pos_bits(width)


class NodeState(NamedTuple):
    kind: int
    level: int
    defined: int
    values: int
    all1: int  # AND over the node's items
    any1: int  # OR over the node's items
    pos: int  # strait / split position, -1 for leaves
    bit: int  # strait bit, -1 otherwise
    cost: int  # tree bits of a strait (type + pos + value), 0 otherwise


class Rule(NamedTuple):
    cmask: int
    cval: int
    pos: int
    bit: int


def log_states(coder) -> list:
    width = coder.bit_count
    states = []
    kinds = Counter()

    def count(t):
        kinds[kind(t)] += 1
        for c in (t.child,) if kind(t) == STRAIT else (t.true_node, t.false_node) if kind(t) == SPLIT else ():
            count(c)

    count(coder.tree)
    strait_type = len(type_codes(kinds.most_common(1)[0][0])[STRAIT])

    def rec(t, rem, defined, values, level):
        k = kind(t)
        if k == LEAF:
            v = int(t.value)
            states.append(NodeState(LEAF, level, defined, values, v, v, -1, -1, 0))
            return v, v
        idx = t.op.idx if k == STRAIT else (coder.root_split_idx if t.bit_idx is None else t.bit_idx)
        msb = len(rem) - 1 - idx
        p = rem[msb]
        m = 1 << (width - 1 - p)
        crem = rem[:msb] + rem[msb + 1:]
        i = len(states)
        states.append(None)
        if k == STRAIT:
            a, o = rec(t.child, crem, defined | m, values | (m if t.op.bit else 0), level + 1)
            cost = strait_type + pos_bits(len(rem)) + 1
            states[i] = NodeState(STRAIT, level, defined, values, a, o, p, int(t.op.bit), cost)
        else:
            a1, o1 = rec(t.true_node, crem, defined | m, values | m, level + 1)
            a0, o0 = rec(t.false_node, crem, defined | m, values, level + 1)
            a, o = a1 & a0, o1 | o0
            states[i] = NodeState(SPLIT, level, defined, values, a, o, p, -1, 0)
        return a, o

    rec(coder.tree, list(range(width)), 0, 0, 0)
    return states


class RuleMiner:
    def __init__(self, coder: GainCoder):
        self.coder = coder
        self.width = coder.bit_count
        self.rule_cost = rule_cost(self.width)
        self.states = log_states(coder)
        cols = np.array([(s.defined, s.values, s.all1, s.any1) for s in self.states], dtype=np.uint64)
        self.D, self.V, self.A, self.O = cols.T
        self.straits = [s for s in self.states if s.kind == STRAIT]
        self.groups = {}
        for i, s in enumerate(self.straits):
            self.groups.setdefault((s.pos, s.bit), []).append(i)
        self.candidates = {}  # Rule -> strait indices it covers
        for (p, b), members in self.groups.items():
            self._group_candidates(p, b, members)
        self.chosen = self._set_cover()

    def bit(self, p):
        return 1 << (self.width - 1 - p)

    def _ok(self, p, b):
        m = np.uint64(self.bit(p))
        return (self.A & m) != 0 if b else (self.O & m) == 0

    def valid(self, rule: Rule) -> bool:
        cm, cv = np.uint64(rule.cmask), np.uint64(rule.cval)
        match = ((self.D & cm) == cm) & ((self.V & cm) == cv)
        return not np.any(match & ~self._ok(rule.pos, rule.bit))

    def _group_candidates(self, p, b, members):
        bad = np.flatnonzero(~self._ok(p, b))
        Db, Vb = self.D[bad], self.V[bad]
        valid = lambda cm, cv: not np.any(((Db & np.uint64(cm)) == np.uint64(cm)) & ((Vb & np.uint64(cm)) == np.uint64(cv)))
        Dg = np.array([self.straits[j].defined for j in members], dtype=np.uint64)
        Vg = np.array([self.straits[j].values for j in members], dtype=np.uint64)
        seen = set()
        for seed in members:
            cm, cv = self.straits[seed].defined, self.straits[seed].values
            common = lambda j: cm & self.straits[j].defined & ~(cv ^ self.straits[j].values)
            for j in sorted(members, key=lambda j: -common(j).bit_count()):
                ncm = common(j)
                if ncm != cm and valid(ncm, cv & ncm):
                    cm, cv = ncm, cv & ncm
            cv &= cm
            if (cm, cv) in seen:
                continue
            seen.add((cm, cv))
            cover = np.flatnonzero(((Dg & np.uint64(cm)) == np.uint64(cm)) & ((Vg & np.uint64(cm)) == np.uint64(cv)))
            self.candidates[Rule(cm, cv, p, b)] = [members[i] for i in cover]

    def _set_cover(self) -> list:
        covered = set()
        chosen = []
        cands = dict(self.candidates)
        while cands:
            gain = lambda r: sum(self.straits[j].cost for j in cands[r] if j not in covered) - self.rule_cost
            best = max(cands, key=gain)
            if gain(best) <= 0:
                break
            new = [j for j in cands.pop(best) if j not in covered]
            covered.update(new)
            chosen.append((best, new))
        self.covered = covered
        return chosen

    def pattern(self, rule: Rule) -> str:
        return "".join("#" if p == rule.pos
                       else str(int(bool(rule.cval & self.bit(p)))) if rule.cmask & self.bit(p)
                       else "." for p in range(self.width))

    def matches(self, rule: Rule) -> int:
        cm, cv = np.uint64(rule.cmask), np.uint64(rule.cval)
        return int(np.sum(((self.D & cm) == cm) & ((self.V & cm) == cv)))

    def item_valid(self, rule: Rule) -> bool:
        """stricter: no item (unique value) matching the condition has bit p != b"""
        vals = np.asarray(self.coder.values, dtype=np.uint64)
        cm, cv, m = np.uint64(rule.cmask), np.uint64(rule.cval), np.uint64(self.bit(rule.pos))
        hit = (vals & cm) == cv
        return not np.any(hit & (((vals & m) != 0) != bool(rule.bit)))


def print_stats(miner: RuleMiner):
    straits, chosen = miner.straits, miner.chosen
    group_size = Counter((s.pos, s.bit) for s in straits)
    best_cover = Counter()
    for r, cover in miner.candidates.items():
        best_cover[(r.pos, r.bit)] = max(best_cover[(r.pos, r.bit)], len(cover))

    print("==Data==")
    print("Nodes:", len(miner.states), "Straits:", len(straits), "(pos,bit) groups:", len(group_size),
          "Candidates:", len(miner.candidates), f"Rule cost: {miner.rule_cost} bits")
    print("==GroupSize== (straits per (pos,bit))")
    build_bins_n_print(list(group_size.values()), [45, 90, 100])
    print("==BestCandidateCover== (max straits one valid rule covers, per group)")
    build_bins_n_print(list(best_cover.values()), [45, 90, 100])
    print("==ChosenRuleCover==")
    if chosen:
        build_bins_n_print([len(c) for _, c in chosen], [45, 90, 100])
        build_bins_n_print([r.cmask.bit_count() for r, _ in chosen], [45, 90, 100])
    print("==ChosenRules== (pattern: 0/1 = condition, # = defined bit, . = free)")
    for r, cover in chosen[:25]:
        saved = sum(straits[j].cost for j in cover)
        print(f"  {miner.pattern(r)} -> {r.pos:>2}={r.bit}  covers {len(cover):>4}/{group_size[(r.pos, r.bit)]:<4}"
              f" gain {saved - miner.rule_cost:>5} bits, matches {miner.matches(r):>5} nodes")
    if len(chosen) > 25:
        print(f"  ... {len(chosen) - 25} more")
    print("==Top groups== ((pos,bit): straits, best single-rule cover)")
    print("  " + ", ".join(f"{p}={b}: {n}/{best_cover[(p, b)]}" for (p, b), n in group_size.most_common(12)))


def size_estimate(miner: RuleMiner):
    straits = miner.straits
    total = sum(s.cost for s in straits)
    saved = sum(straits[j].cost for j in miner.covered)
    table = len(miner.chosen) * miner.rule_cost
    print("==SizeEstimate== (rules firing earlier / type-code shifts ignored)")
    print(f"GainCoder straits: {len(straits)} -> {total} bits")
    print(f"rules: {len(miner.chosen)} x {miner.rule_cost} = {table} bits, cover {len(miner.covered)} straits "
          f"({saved} bits) -> net {table - saved:+d} bits, straits left {total - saved + table} bits")


def verify(miner: RuleMiner):
    for r, cover in miner.chosen:
        assert miner.valid(r), f"invalid rule {miner.pattern(r)} -> {r.pos}={r.bit}"
        for j in cover:
            s = miner.straits[j]
            assert (s.pos, s.bit) == (r.pos, r.bit) and s.defined & r.cmask == r.cmask \
                and s.values & r.cmask == r.cval, "rule does not match a covered strait"
    item_ok = sum(miner.item_valid(r) for r, _ in miner.chosen)
    print(f"(verify OK: chosen rules valid on all {len(miner.states)} nodes, covered straits match; "
          f"{item_ok}/{len(miner.chosen)} also valid item-level)")


if __name__ == "__main__":
    bits_to_take = 32
    for name in ("model.layers.0.input_layernorm.weight.bin", "model.layers.0.post_attention_layernorm.weight.bin"):
        x = np.frombuffer(sandbox_path("bins", name).read_bytes(), dtype=np.uint32)
        values, counts = np.unique(x, return_counts=True)
        print(f"{name}: {len(values)} unique of {len(x)}")
        miner = RuleMiner(GainCoder(values, counts, bits_to_take))
        print_stats(miner)
        verify(miner)
        size_estimate(miner)
        print("END")
