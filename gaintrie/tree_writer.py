"""Bitwise serialization of a GainCoder tree + code stream.

Layout (MSB-first, clarautils BitWriter):
  header  N:32 (< 2**32 items)  width-1:6 (widths 1..64)  one_bit_kind:2
  tree    breadth-first; each node docks onto the previous layer by order
          (split -> 2 children (1 first), strait -> 1, leaf -> 0).
          Every node carries `rem`, the MSB-ordered absolute positions still
          undefined; split/strait remove one, so positions shrink per step.
            type   prefix code, skipped when rem is empty (must be a leaf)
            split  index into rem, ceil(log2(len(rem))) bits
            strait index into rem + 1 value bit
            leaf   residual: the value's bits at rem
  stream  the code of every item; branch bits are value bits, so a code
          plus the tree path fully defines the value.
"""
from collections import Counter, deque
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader

SPLIT, STRAIT, LEAF = 0, 1, 2
KIND_NAMES = ("split", "strait", "leaf")


def kind(t):
    # duck-typed: GainCoder may run as __main__ (separate Node classes)
    return SPLIT if hasattr(t, "true_node") else STRAIT if hasattr(t, "child") else LEAF


def pos_bits(n: int) -> int:
    return (n - 1).bit_length() if n > 1 else 0


def type_codes(one: int):
    others = [k for k in (SPLIT, STRAIT, LEAF) if k != one]
    return {one: (0,), others[0]: (1, 0), others[1]: (1, 1)}


def bit_of(value: int, pos: int, width: int) -> int:
    return (value >> (width - 1 - pos)) & 1


def put_int(bw: BitWriter, value: int, length: int):
    if length > 0:  # BitWriter.put writes nothing for length 0 / non-bool
        bw.put(int(value), length=length)


def put_bits(bw: BitWriter, bits):
    for b in bits:
        bw.put(bool(b))


def read_int(br: BitReader, length: int) -> int:
    return int(br.get_int(length).value) if length > 0 else 0


def bits_written(bw: BitWriter) -> int:
    return len(bw.bytes) * 8 + bw.current_bit


def walk(coder):
    """BFS over coder.tree yielding (node, rem, msb) — msb is the defined index into rem."""
    width = coder.bit_count
    queue = deque([(coder.tree, list(range(width)))])
    while queue:
        t, rem = queue.popleft()
        k = kind(t)
        if k == LEAF:
            yield t, rem, None
            continue
        idx = t.op.idx if k == STRAIT else (coder.root_split_idx if t.bit_idx is None else t.bit_idx)
        msb = len(rem) - 1 - idx
        yield t, rem, msb
        child_rem = rem[:msb] + rem[msb + 1:]
        children = (t.child,) if k == STRAIT else (t.true_node, t.false_node)
        queue.extend((c, child_rem) for c in children)


def write_tree(coder, bw: BitWriter) -> Counter:
    width = coder.bit_count
    nodes = list(walk(coder))
    one = Counter(kind(t) for t, _, _ in nodes).most_common(1)[0][0]
    codes = type_codes(one)
    stats = Counter()

    assert 1 <= width <= 64, f"unsupported width {width}"
    put_int(bw, width - 1, 6)
    put_int(bw, one, 2)
    stats["header"] += 8
    for t, rem, msb in nodes:
        k = kind(t)
        if rem:
            put_bits(bw, codes[k])
            stats["type"] += len(codes[k])
        else:
            assert k == LEAF
        if k == LEAF:
            value = int(t.value)
            put_bits(bw, (bit_of(value, p, width) for p in rem))
            stats["leaf"] += len(rem)
        else:
            name = KIND_NAMES[k]
            put_int(bw, msb, pos_bits(len(rem)))
            stats[f"{name}_pos"] += pos_bits(len(rem))
            if k == STRAIT:
                bw.put(bool(t.op.bit))
                stats["strait_bit"] += 1
    return stats


def read_tree(br: BitReader) -> dict:
    """Returns {code: value}, values rebuilt from split, strait and leaf bits."""
    width = read_int(br, 6) + 1
    one = read_int(br, 2)
    others = [k for k in (SPLIT, STRAIT, LEAF) if k != one]
    table = {}
    queue = deque([(list(range(width)), 0, "")])
    while queue:
        rem, value, code = queue.popleft()
        k = LEAF if not rem else one if not br.get_bool() else others[int(br.get_bool())]
        if k == LEAF:
            for p in rem:
                value |= int(br.get_bool()) << (width - 1 - p)
            table[code] = value
            continue
        msb = read_int(br, pos_bits(len(rem)))
        shift = width - 1 - rem[msb]
        child_rem = rem[:msb] + rem[msb + 1:]
        if k == STRAIT:
            queue.append((child_rem, value | (int(br.get_bool()) << shift), code))
        else:
            queue.append((child_rem, value | (1 << shift), code + "1"))
            queue.append((child_rem, value, code + "0"))
    return table


def encode(coder, x) -> tuple[np.ndarray, Counter]:
    bw = BitWriter()
    if len(x) >= 1 << 32:
        raise ValueError(f"item count {len(x)} does not fit the 32-bit header field")
    put_int(bw, len(x), 32)
    stats = write_tree(coder, bw)
    stats["header"] += 32
    codes = {int(v): c for v, c in coder.codes.items()}
    for v in np.asarray(x).tolist():
        put_bits(bw, (c == "1" for c in codes[v]))
        stats["stream"] += len(codes[v])
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


def mirror_potential(coder):
    """Straits directly below both children of a split at the same position.
    Such a bit is constant in each child but not in the parent, so it is
    always the inverse of its twin — the false-side copy is redundant."""
    width = coder.bit_count
    nodes = {id(t): (t, rem, msb) for t, rem, msb in walk(coder)}

    def strait_chain(t):
        chain = {}
        while kind(t) == STRAIT:
            _, rem, msb = nodes[id(t)]
            chain[rem[msb]] = (t.op.bit, len(rem))
            t = t.child
        return chain

    pairs, saved = 0, 0
    for t, _, _ in nodes.values():
        if kind(t) != SPLIT:
            continue
        a, b = strait_chain(t.true_node), strait_chain(t.false_node)
        for pos in a.keys() & b.keys():
            assert a[pos][0] != b[pos][0]
            pairs += 1
            saved += pos_bits(b[pos][1]) + 1  # position + value of the twin
    return pairs, saved


def leaf_id_potential(coder):
    """Cost of a per-width residual table + ids instead of inline residuals."""
    width = coder.bit_count
    by_width = {}
    for t, rem, _ in walk(coder):
        if kind(t) == LEAF:
            residual = tuple(bit_of(int(t.value), p, width) for p in rem)
            by_width.setdefault(len(rem), []).append(residual)
    inline = sum(w * len(r) for w, r in by_width.items())
    table = sum(w * len(set(r)) + len(r) * pos_bits(len(set(r))) for w, r in by_width.items())
    unique = sum(len(set(r)) for r in by_width.values())
    return inline, table, unique


def size_report(coder, x):
    from gaintrie.Huffman import HuffmanCoder

    data, stats = encode(coder, x)
    assert np.array_equal(decode(data), np.asarray(x, dtype=np.uint64)), "round trip failed"
    n, width = len(x), coder.bit_count
    tree = sum(v for k, v in stats.items() if k not in ("header", "stream"))
    total = sum(stats.values())
    values, counts = np.unique(x, return_counts=True)
    huff = HuffmanCoder(values, counts)
    huff_code = sum(len(huff.codes[v]) * c for v, c in zip(huff.values, huff.counts))
    huff_dict = len(values) * width

    print("==TreeWriter== (round trip OK)")
    print("tree:", ", ".join(f"{k}={stats[k]}" for k in ("type", "split_pos", "strait_pos", "strait_bit", "leaf")),
          f"-> {tree}")
    print(f"stream={stats['stream']} header={stats['header']} total={total} "
          f"({total / n:.3f} bits/item, {total / (n * width):.3f} of raw {n * width})")
    print(f"huffman: code={huff_code} + dict={huff_dict} -> {huff_code + huff_dict} "
          f"({(huff_code + huff_dict) / n:.3f} bits/item)")
    pairs, saved = mirror_potential(coder)
    print(f"mirror: {pairs} inverted sibling straits, -{saved} bits (pos+value, excl. type)")
    inline, table, unique = leaf_id_potential(coder)
    print(f"leaf ids: inline={inline}, table+ids={table} ({unique} unique residuals)")
    return stats
