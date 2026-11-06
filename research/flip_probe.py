"""Probe for FlipGainCoder: flip counts and estimated tree costs on the
GainCoder split tree. Args: <bin name> <width>, default input_layernorm 32."""
import sys
from pathlib import Path
import numpy as np
from collections import Counter

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaintrie.GainCoder import GainCoder
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import walk, kind, SPLIT, STRAIT, LEAF, bit_of, pos_bits

name, width = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("model.layers.0.input_layernorm.weight.bin", 32)
x = np.frombuffer(sandbox_path("bins", name).read_bytes(), dtype=np.uint32 if width == 32 else np.uint16)
values, counts = np.unique(x, return_counts=True)
c = GainCoder(values, counts, width)
nodes = list(walk(c))
info = {id(t): (rem, msb) for t, rem, msb in nodes}

def leaves(t):
    k = kind(t)
    if k == LEAF: return [int(t.value)]
    if k == STRAIT: return leaves(t.child)
    return leaves(t.true_node) + leaves(t.false_node)

def probs(t, rem):
    v = np.array(leaves(t), dtype=np.uint64)
    return {p: float(np.mean((v >> np.uint64(width - 1 - p)) & np.uint64(1))) for p in rem}

S = Counter()
# A) heuristic: flips at entry of each split-group (node incl. its strait chain), majority switch
def rec(t, mask):
    rem, _ = info[id(t)]
    if kind(t) == LEAF:
        v = int(t.value)
        S["leaf_mismatch"] += sum(bit_of(v, p, width) != mask[p] for p in rem)
        S["leaf_bits"] += len(rem)
        return
    mask = dict(mask)
    for p, pr in probs(t, rem).items():
        maj = 1 if pr > 0.5 else 0 if pr < 0.5 else mask[p]
        if maj != mask[p]:
            S["flips"] += 1; S["flip_pos_bits"] += pos_bits(len(rem)); mask[p] = maj
    while kind(t) == STRAIT:
        r, m = info[id(t)]
        assert t.op.bit == mask[r[m]]
        S["straits"] += 1
        t = t.child
    if kind(t) == LEAF:
        return rec(t, mask)
    S["splits"] += 1
    rec(t.true_node, mask); rec(t.false_node, mask)
rec(c.tree, {p: 0 for p in range(width)})

# B) minimum flips so every strait value is predicted (per position, Fitch-style on split-groups)
def need(t, p):
    """cost table {bit: min flips in subtree if mask[p]==bit at entry}, None = p unconstrained"""
    while kind(t) == STRAIT:
        r, m = info[id(t)]
        if r[m] == p:
            b = t.op.bit
            return {b: 0, 1 - b: 1}
        t = t.child
    if kind(t) == LEAF:
        return None
    a, b = need(t.true_node, p), need(t.false_node, p)
    if a is None and b is None: return None
    out = {}
    for bit in (0, 1):
        tot = 0
        for ch in (a, b):
            if ch is not None: tot += min(ch[bit], ch[1 - bit] + 1)
        out[bit] = tot
    return out
minflips = 0
for p in range(width):
    n = need(c.tree, p)
    if n is not None: minflips += min(n[0], n[1] + 1)
S["min_flips"] = minflips

# straits agreeing with parent group's mask without any flips (mask = all 0) → baseline count
print(name, width, dict(S))

# C) no straits: rem = positions not split on; mask flips at split nodes; leaf = exceptions
from math import comb, log2, ceil
def skip(t):
    while kind(t) == STRAIT: t = t.child
    return t
def split_pos(t):
    rem, msb = info[id(t)]
    return rem[msb]

def run_c(flip_rule):
    R = Counter()
    def rec(t, rem, mask):
        t = skip(t)
        if kind(t) == LEAF:
            v = int(t.value)
            e = sum(bit_of(v, p, width) != mask[p] for p in rem)
            r = len(rem)
            R["leaves"] += 1; R["leaf_rem"] += r; R["exceptions"] += e
            R["leaf_cost"] += ceil(log2(r + 1)) + ceil(log2(comb(r, e))) if r else 0
            return
        mask = dict(mask)
        for p, pr in probs(t, rem).items():
            if flip_rule(pr, mask[p]):
                mask[p] ^= 1; R["flips"] += 1; R["flip_cost"] += pos_bits(len(rem)) + 1
        R["split_cost"] += pos_bits(len(rem)) + 1
        sp = split_pos(t)
        crem = [p for p in rem if p != sp]
        rec(t.true_node, crem, mask); rec(t.false_node, crem, mask)
    rec(c.tree, list(range(width)), {p: 0 for p in range(width)})
    R["total"] = R["leaf_cost"] + R["flip_cost"] + R["split_cost"]
    return dict(R)

maj = lambda pr, m: (pr > 0.5 and m == 0) or (pr < 0.5 and m == 1)
for thr in (0.5, 0.7, 0.85, 0.95):
    rule = (lambda t: lambda pr, m: (m == 0 and pr > t) or (m == 1 and pr < 1 - t))(thr)
    print(f" C thr={thr}", run_c(rule))
print(" C noflip", run_c(lambda pr, m: False))

# D) DP-optimal flip placement per position (weights: flip = pos+2, exception = e_w)
def dp_mask(e_w):
    flips_at = {}  # (id(node), p) -> True
    def solve(t, rem_before, p):
        t = skip(t)
        if kind(t) == LEAF:
            v = int(t.value)
            b = bit_of(v, p, width)
            return {m: (0 if m == b else e_w) for m in (0, 1)}, None
        f = pos_bits(len(rem_before)) + 2
        sp = split_pos(t)
        if sp == p:
            return {0: 0, 1: 0}, None
        crem = [q for q in rem_before if q != sp]
        ca, _ = solve(t.true_node, crem, p); cb, _ = solve(t.false_node, crem, p)
        inner = {m: ca[m] + cb[m] for m in (0, 1)}
        return {m: min(inner[m], inner[1 - m] + f) for m in (0, 1)}, inner
    def place(t, rem_before, p, m):
        t = skip(t)
        if kind(t) == LEAF: return
        sp = split_pos(t)
        if sp == p: return
        f = pos_bits(len(rem_before)) + 2
        crem = [q for q in rem_before if q != sp]
        ca, _ = solve(t.true_node, crem, p); cb, _ = solve(t.false_node, crem, p)
        inner = {k: ca[k] + cb[k] for k in (0, 1)}
        if inner[1 - m] + f < inner[m]:
            flips_at[(id(t), p)] = True; m = 1 - m
        place(t.true_node, crem, p, m); place(t.false_node, crem, p, m)
    for p in range(width):
        place(c.tree, list(range(width)), p, 0)
    return lambda node, p: (id(node), p) in flips_at

def run_c_set(flip_at):
    R = Counter(); ecount = Counter()
    def rec(t, rem, mask):
        t = skip(t)
        if kind(t) == LEAF:
            v = int(t.value)
            e = sum(bit_of(v, p, width) != mask[p] for p in rem); r = len(rem)
            ecount[e] += 1
            R["exceptions"] += e
            R["leaf_cost"] += ceil(log2(r + 1)) + ceil(log2(comb(r, e))) if r else 0
            return
        mask = dict(mask)
        for p in rem:
            if flip_at(t, p):
                mask[p] ^= 1; R["flips"] += 1; R["flip_cost"] += pos_bits(len(rem)) + 1
        R["split_cost"] += pos_bits(len(rem)) + 1
        sp = split_pos(t); crem = [p for p in rem if p != sp]
        rec(t.true_node, crem, mask); rec(t.false_node, crem, mask)
    rec(c.tree, list(range(width)), {p: 0 for p in range(width)})
    R["total"] = R["leaf_cost"] + R["flip_cost"] + R["split_cost"]
    return dict(R), sorted(ecount.items())
for ew in (2, 3, 4, 5):
    print(f" D e_w={ew}", *run_c_set(dp_mask(ew)))
