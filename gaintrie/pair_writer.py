"""tree_writer variant with a PAIR node: a subtree holding exactly two items is
written as one node instead of straits + split + two leaves.

At a 2-item node the two values agree on some rem positions and differ on the
others (at least one). PAIR stores the diff mask over rem and the value of one
item; the other item is that value with the mask bits inverted. The branch
position is the first differing rem position; the item with bit 1 there gets
code + "1" (its bit there is known, so only len(rem)-1 value bits are written).

The encoder may cut a strait chain: keep the first k straits classic and put
the PAIR below them, or keep the whole subtree classic — whatever is cheapest.

Layout (BFS, nodes dock by order like tree_writer):
  header  N:32  width-1:6  mask_mode:1  pos_mode:1  used kinds-1:2  kinds by frequency:2 each
          pos_mode 1: one order of the bit positions (permutation, each entry an index
          into the positions left), chosen to minimize the sum of all position indices
  tree    type   prefix code 0 / 10 / 110 / 111 by frequency, skipped when rem is empty
          split  position: pos_mode 0: index into rem (MSB order), 1: index among the
                 open positions in the stored order
          strait position like split + value bit
          leaf   residual: the value's bits at rem
          pair   diff mask over rem (mode 0: raw len(rem) bits, mode 1: count-1 of
                 len(rem) + colex rank of C(rem, count)),
                 then the "1" item's bits at rem without the branch position
  stream  mode:1 — 0: every item's tree code (its path)
                   1: Huffman over the leaves by count: max length:6, then each leaf's
                      code length-1 of max length (leaf order = BFS order in which the
                      tree rebuilds them), canonical codes; then every item's code
Indices (positions, mask count, colex rank) are truncated binary over their
range n: the first 2^k - n values take k-1 bits, the rest k (k = pos_bits(n)).
"""
from collections import Counter, deque
from math import comb
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader
from gaintrie.GainCoder import GainCoder
from gaintrie.code_utils import (kind_codes, colex_rank, colex_unrank, tb_bits, put_tb, read_tb,
                                 huffman_lengths, canonical_codes, print_type_sizes, gain_type_sizes)
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import (kind, pos_bits, bit_of, put_int, put_bits, read_int, bits_written,
                                  encode as gain_encode, SPLIT, STRAIT, LEAF)

PAIR = 3
KIND_NAMES = {SPLIT: "split", STRAIT: "strait", LEAF: "leaf", PAIR: "pair"}
RAW, COMB = 0, 1
TREE_CODES, HUFFMAN, AUTO = 0, 1, 2
POS_MSB, POS_ORDER, POS_AUTO = 0, 1, 2
MAX_LEN_BITS = 6


def step(coder, t, rem):
    """(msb index into rem, child rem) of a split / strait, as tree_writer.walk"""
    idx = t.op.idx if kind(t) == STRAIT else (coder.root_split_idx if t.bit_idx is None else t.bit_idx)
    msb = len(rem) - 1 - idx
    return msb, rem[:msb] + rem[msb + 1:]


def leaf_values(t) -> list:
    k = kind(t)
    if k == LEAF:
        return [int(t.value)]
    if k == STRAIT:
        return leaf_values(t.child)
    return leaf_values(t.true_node) + leaf_values(t.false_node)


def pair_split(a: int, b: int, rem, width):
    """-> (diff indices into rem, the item with bit 1 at the branch position)"""
    diff = [i for i, p in enumerate(rem) if bit_of(a, p, width) != bit_of(b, p, width)]
    one = a if bit_of(a, rem[diff[0]], width) else b
    return diff, one


def mask_bits(r: int, diff, mode: int) -> int:
    d = len(diff)
    return r if mode == RAW else tb_bits(d - 1, r) + tb_bits(colex_rank(diff), comb(r, d))


class PairWriter:
    def __init__(self, coder: GainCoder, mask_mode=COMB):
        self.coder = coder
        self.width = coder.bit_count
        self.mask_mode = mask_mode
        self.pairs = self._choose()

    def pair_cost(self, t, rem, codes):
        a, b = leaf_values(t)
        diff, _ = pair_split(a, b, rem, self.width)
        return len(codes[PAIR]) + mask_bits(len(rem), diff, self.mask_mode) + len(rem) - 1

    def classic_cost(self, t, rem, codes):
        k = kind(t)
        tb = len(codes[k]) if rem else 0
        if k == LEAF:
            return tb + len(rem)
        msb, crem = step(self.coder, t, rem)
        if k == STRAIT:
            return tb + tb_bits(msb, len(rem)) + 1 + self.classic_cost(t.child, crem, codes)
        return tb + tb_bits(msb, len(rem)) + self.classic_cost(t.true_node, crem, codes) \
            + self.classic_cost(t.false_node, crem, codes)

    def _heads(self):
        """topmost non-leaf nodes with exactly 2 items, with their rem"""
        out = []

        def rec(t, rem):
            k = kind(t)
            if k == LEAF:
                return
            if len(leaf_values(t)) == 2:
                out.append((t, rem))
                return
            _, crem = step(self.coder, t, rem)
            for c in (t.child,) if k == STRAIT else (t.true_node, t.false_node):
                rec(c, crem)

        rec(self.coder.tree, list(range(self.width)))
        return out

    def _choose(self):
        heads = self._heads()
        codes = kind_codes([STRAIT, LEAF, SPLIT, PAIR])
        for _ in range(4):  # type code lengths depend on the choices: iterate to a fixpoint
            pairs = {}
            for head, rem in heads:
                best, best_cost = None, self.classic_cost(head, rem, codes)
                t, r, before = head, rem, 0
                while True:
                    cost = before + self.pair_cost(t, r, codes)
                    if cost < best_cost:
                        best, best_cost = (id(t), r), cost
                    if kind(t) != STRAIT:
                        break
                    msb, nr = step(self.coder, t, r)
                    before += len(codes[STRAIT]) + tb_bits(msb, len(r)) + 1
                    r = nr
                    t = t.child
                if best is not None:
                    pairs[best[0]] = best[1]
            self.pairs = pairs
            order = [k for k, _ in Counter(k for _, _, _, k in self.walk()).most_common()]
            new = kind_codes(order + [k for k in KIND_NAMES if k not in order])  # unused kinds stay priced
            if new == codes:
                break
            codes = new
        return pairs

    def walk(self):
        """BFS yielding (node, rem, code, kind); PAIR nodes are not descended"""
        queue = deque([(self.coder.tree, list(range(self.width)), "")])
        while queue:
            t, rem, code = queue.popleft()
            k = PAIR if id(t) in self.pairs else kind(t)
            yield t, rem, code, k
            if k in (PAIR, LEAF):
                continue
            _, crem = step(self.coder, t, rem)
            if k == STRAIT:
                queue.append((t.child, crem, code))
            else:
                queue.extend(((t.true_node, crem, code + "1"), (t.false_node, crem, code + "0")))


def defined_pos(pw: PairWriter, t, rem) -> int:
    msb, _ = step(pw.coder, t, rem)
    return rem[msb]


def order_cost(order, W) -> int:
    """sum of all written indices: W[q, p] counts nodes defining p while q is open"""
    o = np.asarray(order)
    return int(np.triu(W[np.ix_(o, o)], 1).sum())


def position_order(pw: PairWriter, nodes):
    """one order of the bit positions minimizing the sum of the split / strait indices
    (index = rank of the defined position among the open ones in this order). Start:
    average definition time (bits already defined when the position gets defined),
    then move single positions to their best slot until nothing improves.
    -> (order, W, start order)"""
    width = pw.width
    W = np.zeros((width, width), dtype=np.int64)
    t_sum, t_cnt = np.zeros(width), np.zeros(width)
    for t, rem, _, k in nodes:
        if k in (SPLIT, STRAIT):
            p = defined_pos(pw, t, rem)
            W[rem, p] += 1
            W[p, p] -= 1
            t_sum[p] += width - len(rem)
            t_cnt[p] += 1
    avg = np.where(t_cnt > 0, t_sum / np.maximum(t_cnt, 1), width)  # never defined: last
    start = sorted(range(width), key=lambda p: (avg[p], p))
    order = list(start)
    improved = True
    while improved:
        improved = False
        for p in list(order):
            rest = [q for q in order if q != p]
            # cost of p at slot s: open-before-p pairs W[x, p] for x before, W[p, x] for x after
            before = np.concatenate(([0], np.cumsum([W[x, p] for x in rest])))
            after = np.concatenate((np.cumsum([W[p, x] for x in rest][::-1])[::-1], [0]))
            best = int(np.argmin(before + after))
            cur = order.index(p)
            if before[best] + after[best] < before[cur] + after[cur]:
                order = rest[:best] + [p] + rest[best:]
                improved = True
    return order, W, start


def ordered_index(rem, rank, p) -> int:
    return sum(rank[q] < rank[p] for q in rem)


def permutation_bits(order) -> int:
    width, left, bits = len(order), sorted(order), 0
    for i, p in enumerate(order):
        bits += tb_bits(left.index(p), width - i)
        left.remove(p)
    return bits


def position_costs(pw: PairWriter, nodes):
    """-> {pos_mode: (bits of all split / strait positions + the order, order)}"""
    defs = [(rem, defined_pos(pw, t, rem)) for t, rem, _, k in nodes if k in (SPLIT, STRAIT)]
    order, _, _ = position_order(pw, nodes)
    rank = {p: i for i, p in enumerate(order)}
    return {POS_MSB: (sum(tb_bits(rem.index(p), len(rem)) for rem, p in defs), None),
            POS_ORDER: (permutation_bits(order)
                        + sum(tb_bits(ordered_index(rem, rank, p), len(rem)) for rem, p in defs), order)}


def write_tree(pw: PairWriter, bw: BitWriter, pos_mode=POS_AUTO):
    width = pw.width
    nodes = list(pw.walk())
    order = [k for k, _ in Counter(k for *_, k in nodes).most_common()]
    codes = kind_codes(order)
    costs = position_costs(pw, nodes)
    if pos_mode == POS_AUTO:
        pos_mode = min(costs, key=lambda m: costs[m][0])
    _, pos_order = costs[pos_mode]
    rank = {p: i for i, p in enumerate(pos_order)} if pos_order else None
    stats, table = Counter(), {}
    put_int(bw, width - 1, 6)
    put_int(bw, pw.mask_mode, 1)
    put_int(bw, pos_mode, 1)
    put_int(bw, len(order) - 1, 2)
    for k in order:
        put_int(bw, k, 2)
    stats["header"] += 10 + 2 * len(order)
    if pos_order:
        left = sorted(pos_order)
        for i, p in enumerate(pos_order):
            put_tb(bw, left.index(p), width - i)
            left.remove(p)
        stats["pos_tables"] += permutation_bits(pos_order)
    for t, rem, code, k in nodes:
        name = KIND_NAMES[k]
        if rem:
            put_bits(bw, codes[k])
            stats[f"{name}_type"] += len(codes[k])
        else:
            assert k == LEAF
        if k == LEAF:
            value = int(t.value)
            put_bits(bw, (bit_of(value, p, width) for p in rem))
            stats["leaf"] += len(rem)
            table[value] = code
        elif k == PAIR:
            a, b = leaf_values(t)
            diff, one = pair_split(a, b, rem, width)
            r, d = len(rem), len(diff)
            if pw.mask_mode == RAW:
                put_bits(bw, (i in diff for i in range(r)))
            else:
                put_tb(bw, d - 1, r)
                put_tb(bw, colex_rank(diff), comb(r, d))
            stats["pair_mask"] += mask_bits(r, diff, pw.mask_mode)
            put_bits(bw, (bit_of(one, p, width) for i, p in enumerate(rem) if i != diff[0]))
            stats["pair_value"] += r - 1
            table[one], table[a ^ b ^ one] = code + "1", code + "0"
        else:
            p = defined_pos(pw, t, rem)
            if pos_mode == POS_MSB:
                put_tb(bw, rem.index(p), len(rem))
                stats[f"{name}_pos"] += tb_bits(rem.index(p), len(rem))
            else:
                i = ordered_index(rem, rank, p)
                put_tb(bw, i, len(rem))
                stats[f"{name}_pos"] += tb_bits(i, len(rem))
            if k == STRAIT:
                bw.put(bool(t.op.bit))
                stats["strait_bit"] += 1
    return stats, table


def read_tree(br: BitReader) -> dict:
    """Returns {code: value}."""
    width = read_int(br, 6) + 1
    mask_mode = read_int(br, 1)
    pos_mode = read_int(br, 1)
    order = [read_int(br, 2) for _ in range(read_int(br, 2) + 1)]
    rank = None
    if pos_mode == POS_ORDER:
        left, pos_order = list(range(width)), []
        for i in range(width):
            pos_order.append(left.pop(read_tb(br, width - i)))
        rank = {p: i for i, p in enumerate(pos_order)}
    table = {}
    queue = deque([(list(range(width)), 0, "")])
    while queue:
        rem, value, code = queue.popleft()
        if not rem:
            k = LEAF
        else:
            i = 0
            while i < len(order) - 1 and br.get_bool():
                i += 1
            k = order[i]
        if k == LEAF:
            for p in rem:
                value |= int(br.get_bool()) << (width - 1 - p)
            table[code] = value
            continue
        if k == PAIR:
            r = len(rem)
            if mask_mode == RAW:
                diff = [i for i in range(r) if br.get_bool()]
            else:
                d = read_tb(br, r) + 1
                diff = colex_unrank(read_tb(br, comb(r, d)), d)
            one = value | (1 << (width - 1 - rem[diff[0]]))
            for i, p in enumerate(rem):
                if i != diff[0]:
                    one |= int(br.get_bool()) << (width - 1 - p)
            mask = sum(1 << (width - 1 - rem[i]) for i in diff)
            table[code + "1"], table[code + "0"] = one, one ^ mask
            continue
        if pos_mode == POS_MSB:
            msb = read_tb(br, len(rem))
        else:
            msb = rem.index(sorted(rem, key=rank.get)[read_tb(br, len(rem))])
        shift = width - 1 - rem[msb]
        crem = rem[:msb] + rem[msb + 1:]
        if k == STRAIT:
            queue.append((crem, value | (int(br.get_bool()) << shift), code))
        else:
            queue.append((crem, value | (1 << shift), code + "1"))
            queue.append((crem, value, code + "0"))
    return table


def code_length_bits(lengths) -> int:
    max_len = max(lengths)
    return MAX_LEN_BITS + (sum(tb_bits(l - 1, max_len) for l in lengths) if max_len else 0)


def put_code_lengths(bw: BitWriter, lengths):
    """max length:6, then each length-1 of max length (nothing more when max is 0)"""
    max_len = max(lengths)
    assert max_len < 1 << MAX_LEN_BITS
    put_int(bw, max_len, MAX_LEN_BITS)
    if max_len:
        for l in lengths:
            put_tb(bw, l - 1, max_len)


def read_code_lengths(br: BitReader, count: int) -> list:
    max_len = read_int(br, MAX_LEN_BITS)
    return [read_tb(br, max_len) + 1 if max_len else 0 for _ in range(count)]


def encode(pw: PairWriter, x, stream=AUTO, pos=POS_AUTO):
    """stream: TREE_CODES (tree paths), HUFFMAN (over the leaves, by count) or AUTO (smaller one);
    pos: how split / strait positions are written (POS_*), POS_AUTO = smallest"""
    bw = BitWriter()
    if len(x) >= 1 << 32:
        raise ValueError(f"item count {len(x)} does not fit the 32-bit header field")
    items = np.asarray(x).tolist()
    put_int(bw, len(x), 32)
    stats, table = write_tree(pw, bw, pos)
    stats["header"] += 32

    leaves = list(table)  # leaf order = order the decoder rebuilds them in
    counts = Counter(items)
    lengths = huffman_lengths([counts[v] for v in leaves])
    huff_bits = sum(l * counts[v] for l, v in zip(lengths, leaves)) + code_length_bits(lengths)
    tree_bits = sum(len(table[v]) * counts[v] for v in leaves)
    if stream == AUTO:
        stream = HUFFMAN if huff_bits < tree_bits else TREE_CODES
    put_int(bw, stream, 1)
    stats["header"] += 1
    if stream == HUFFMAN:
        put_code_lengths(bw, lengths)
        stats["code_lengths"] += code_length_bits(lengths)
        table = dict(zip(leaves, canonical_codes(lengths)))
    for v in items:
        put_bits(bw, (c == "1" for c in table[v]))
        stats["stream"] += len(table[v])
    assert sum(stats.values()) == bits_written(bw)
    return bw.get_bytes(), stats


def decode(data) -> np.ndarray:
    br = BitReader(data)
    n = read_int(br, 32)
    table = read_tree(br)
    if read_int(br, 1) == HUFFMAN:
        lengths = read_code_lengths(br, len(table))
        table = dict(zip(canonical_codes(lengths), table.values()))
    out = []
    for _ in range(n):
        code = ""
        while code not in table:
            code += "1" if br.get_bool() else "0"
        out.append(table[code])
    return np.array(out, dtype=np.uint64)


def best_writer(coder: GainCoder, x, stream=AUTO, pos=POS_AUTO):
    """encodes with both mask modes, returns (writer, data, stats) of the smaller one"""
    runs = [(pw, *encode(pw, x, stream, pos)) for pw in (PairWriter(coder, RAW), PairWriter(coder, COMB))]
    return min(runs, key=lambda r: sum(r[2].values()))


def size_report(coder: GainCoder, x, stream=AUTO):
    pw, data, stats = best_writer(coder, x, stream)
    assert np.array_equal(decode(data), np.asarray(x, dtype=np.uint64)), "round trip failed"
    rows = {name: [0, stats[f"{name}_type"], 0] for name in ("split", "strait", "leaf", "pair")}
    rows["split"][2] = stats["split_pos"]
    rows["strait"][2] = stats["strait_pos"] + stats["strait_bit"]
    rows["leaf"][2] = stats["leaf"]
    rows["pair"][2] = stats["pair_mask"] + stats["pair_value"]
    for *_, k in pw.walk():
        rows[KIND_NAMES[k]][0] += 1
    print(f"(round trip OK, {len(data)} bytes) mask_mode={('raw', 'comb')[pw.mask_mode]} "
          f"stream={'huffman' if stats['code_lengths'] else 'tree codes'} "
          f"pair: mask={stats['pair_mask']} value={stats['pair_value']}")
    total = print_type_sizes(rows, {"pos_tables": stats["pos_tables"], "stream": stats["stream"],
                                    "code_lengths": stats["code_lengths"], "header": stats["header"]},
                             len(x), pw.width, "PairWriter")
    return pw, total


if __name__ == "__main__":
    bits_to_take = 16  # bf16 weights
    for name in ("model.layers.0.input_layernorm.weight.bin", "model.layers.0.post_attention_layernorm.weight.bin",
                 "model.layers.4.post_attention_layernorm.weight.bin"):
        x = np.frombuffer(sandbox_path("bins", name).read_bytes(), dtype=np.uint16)
        values, counts = np.unique(x, return_counts=True)
        print(f"{name}: {len(values)} unique of {len(x)}")
        coder = GainCoder(values, counts, bits_to_take)
        _, base = gain_encode(coder, x)
        base_total = print_type_sizes(gain_type_sizes(coder), {"stream": base["stream"], "header": base["header"]},
                                      len(x), bits_to_take, "GainCoder (baseline)")
        pw, total = size_report(coder, x)
        heads = pw._heads()
        cuts = Counter()
        for head, rem in heads:
            t, k = head, 0
            while id(t) not in pw.pairs and kind(t) == STRAIT:
                t, k = t.child, k + 1
            cuts["classic" if id(t) not in pw.pairs else k] += 1
        print(f"2-item subtrees: {len(heads)}, pairs: {len(pw.pairs)}, "
              f"straits kept above the pair: {sorted(cuts.items(), key=str)}")
        print(f"==Summary== PairWriter {total} vs GainCoder {base_total} ({total - base_total:+d}, "
              f"{total / len(x):.3f} vs {base_total / len(x):.3f} bits/item)")
        print("END")
