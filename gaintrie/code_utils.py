"""Shared coding helpers for the experimental tree writers (pair_writer, FlipGainCoder):
type prefix codes, colex ranks of index sets, truncated binary and the per-type
size table."""
from collections import Counter
from math import comb
from pathlib import Path
import heapq
import itertools
import sys

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader
from gaintrie.tree_writer import walk, kind, type_codes, pos_bits, put_int, read_int, STRAIT, LEAF


def kind_codes(order) -> dict:
    """prefix code by frequency rank: 0, 10, 110, ... (last one without the 0)."""
    return {k: (1,) * i + ((0,) if i < len(order) - 1 else ()) for i, k in enumerate(order)}


def colex_rank(idx) -> int:
    return sum(comb(i, k + 1) for k, i in enumerate(sorted(idx)))


def colex_unrank(rank: int, count: int) -> list:
    out = []
    for k in range(count, 0, -1):
        c = k - 1
        while comb(c + 1, k) <= rank:
            c += 1
        rank -= comb(c, k)
        out.append(c)
    return out[::-1]


def tb_bits(idx: int, n: int) -> int:
    """truncated binary length of idx in range(n): the first 2^k - n values take k-1 bits"""
    if n <= 1:
        return 0
    k = pos_bits(n)
    return k - 1 if idx < (1 << k) - n else k


def put_tb(bw: BitWriter, idx: int, n: int):
    if n <= 1:
        return
    k = pos_bits(n)
    u = (1 << k) - n
    if idx < u:
        put_int(bw, idx, k - 1)
    else:
        put_int(bw, idx + u, k)


def read_tb(br: BitReader, n: int) -> int:
    if n <= 1:
        return 0
    k = pos_bits(n)
    u = (1 << k) - n
    x = read_int(br, k - 1)
    return x if x < u else ((x << 1) | int(br.get_bool())) - u


def huffman_lengths(counts: list) -> list:
    """Huffman code length per symbol (a single symbol gets 0 bits)"""
    if len(counts) == 1:
        return [0]
    tie = itertools.count()
    heap = [(c, next(tie), [i]) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    lengths = [0] * len(counts)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for i in s1 + s2:
            lengths[i] += 1
        heapq.heappush(heap, (c1 + c2, next(tie), s1 + s2))
    return lengths


def canonical_codes(lengths: list) -> list:
    """canonical prefix codes ("0"/"1" strings) for the given lengths, ordered by (length, index)"""
    codes = [""] * len(lengths)
    code, prev = 0, 0
    for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        code <<= lengths[i] - prev
        prev = lengths[i]
        codes[i] = format(code, f"0{prev}b") if prev else ""
        code += 1
    return codes


def gain_type_sizes(coder) -> dict:
    """GainCoder tree per kind: {name: (count, type bits, payload bits)}, as tree_writer writes it."""
    nodes = list(walk(coder))
    codes = type_codes(Counter(kind(t) for t, _, _ in nodes).most_common(1)[0][0])
    names = ("split", "strait", "leaf")
    out = {n: [0, 0, 0] for n in names}
    for t, rem, _ in nodes:
        k = kind(t)
        row = out[names[k]]
        row[0] += 1
        row[1] += len(codes[k]) if rem else 0
        row[2] += len(rem) if k == LEAF else pos_bits(len(rem)) + (k == STRAIT)
    return out


def print_type_sizes(rows: dict, stats_extra: dict, n: int, width: int, title: str):
    print(f"==Types {title}==")
    print(f"{'type':<8}{'count':>8}{'type bits':>11}{'payload':>10}{'total':>10}{'bits/node':>11}{'share':>8}")
    total = sum(r[1] + r[2] for r in rows.values()) + sum(stats_extra.values())
    for name, (count, tb, pb) in rows.items():
        print(f"{name:<8}{count:>8}{tb:>11}{pb:>10}{tb + pb:>10}"
              f"{(tb + pb) / count if count else 0:>11.2f}{(tb + pb) / total:>8.1%}")
    tree = sum(r[1] + r[2] for r in rows.values())
    print(f"{'tree':<8}{'':>8}{'':>11}{'':>10}{tree:>10}{'':>11}{tree / total:>8.1%}")
    for name, bits in stats_extra.items():
        print(f"{name:<8}{'':>8}{'':>11}{'':>10}{bits:>10}{'':>11}{bits / total:>8.1%}")
    print(f"{'total':<8}{'':>8}{'':>11}{'':>10}{total:>10}   "
          f"{total / n:.3f} bits/item, {total / (n * width):.3f} of raw {n * width}")
    return total
