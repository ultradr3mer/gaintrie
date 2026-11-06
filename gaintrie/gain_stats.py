from collections import Counter

import numpy as np

from clarautils import build_bins_n_print, get_bits


def iter_tree(tree):
    yield tree
    if hasattr(tree, "child"):
        yield from iter_tree(tree.child)
    elif hasattr(tree, "true_node"):
        yield from iter_tree(tree.true_node)
        yield from iter_tree(tree.false_node)


def is_split(n):
    return hasattr(n, "true_node")


def is_strait(n):
    return hasattr(n, "child")


def run_positions(coder, kind):
    return np.array([op.abs_pos for r in coder.runs for op in r if op.kind == kind], dtype=np.uint32)


def run_levels(coder, kind):
    levels = [[] for _ in range(coder.bit_count)]
    for run in coder.runs:
        for op in run:
            if op.kind == kind:
                levels[op.abs_pos].append(op.level)
    return levels


def print_stats(coder):
    nodes = list(iter_tree(coder.tree))
    splits = [n for n in nodes if is_split(n)]
    straits = [n for n in nodes if is_strait(n)]
    leafs = [n for n in iter_tree(coder.node) if not is_split(n)]
    leaf_len = [len(get_bits(leaf.value)) for leaf in leafs]
    flag_len = ([len(get_bits(s.op.idx)) for s in straits]
                + [len(get_bits(n.bit_idx)) for n in splits if n.bit_idx is not None])

    print("==Data==")
    print("Codes:", len(coder.codes), "Nodes:", len(splits),
          "Straits:", len(straits), "Leafs:", len(leafs))
    print("==Leafs==")
    build_bins_n_print(leaf_len, [45, 90, 100])
    print("==Flags==")
    if len(flag_len):
        build_bins_n_print(flag_len, [45, 90, 100])
    print("==AbsStraits==")
    strait_pos = np.array([s.abs_pos for s in coder.straits], dtype=np.uint32)
    if len(strait_pos):
        build_bins_n_print(strait_pos, [45, 90, 100])
    strait_rules = Counter((s.abs_pos, s.bit) for s in coder.straits)
    print(f"Rules: {len(strait_rules)} unique of {len(coder.straits)}")
    print("Top:", ", ".join(f"{p}={b}×{c}" for (p, b), c in strait_rules.most_common(5)))
    print("==AbsSplits==")
    split_pos = run_positions(coder, 'split')
    if len(split_pos):
        build_bins_n_print(split_pos, [45, 90, 100])


def plot_bit_definition_order(coder):
    strait_levels, split_levels = run_levels(coder, 'strait'), run_levels(coder, 'split')
    strait_qs = np.array([np.percentile(p, [25, 50, 75]) if len(p) > 0 else [0, 0, 0]
                          for p in strait_levels])
    split_qs = np.array([np.percentile(p, [25, 50, 75]) if len(p) > 0 else [0, 0, 0]
                         for p in split_levels])
    strait_order = np.argsort(strait_qs.mean(axis=1))
    split_order = np.argsort(split_qs.mean(axis=1))
    print(f"AbsStraits sorted (pos, [q25, q50, q75]): "
          f"{[(int(p), q.tolist()) for p, q in zip(strait_order, strait_qs[strait_order])]}")
    print(f"AbsSplits sorted (pos, [q25, q50, q75]): "
          f"{[(int(p), q.tolist()) for p, q in zip(split_order, split_qs[split_order])]}")

    import matplotlib.pyplot as plt
    fig, (ax_straits, ax_splits) = plt.subplots(2, 1, figsize=(12, 8))
    for ax, levels, order, title in ((ax_straits, strait_levels, strait_order, "Strait levels per abs bit pos"),
                                     (ax_splits, split_levels, split_order, "Split levels per abs bit pos")):
        ax.boxplot([levels[p] for p in order])
        ax.set_xticks(np.arange(1, len(order) + 1), [int(p) for p in order])
        ax.set_title(title)
        ax.set_xlabel("abs bit pos")
        ax.set_ylabel("level")
    fig.tight_layout()
    fig.savefig("levels_boxplot.png")
    plt.show()


def plot_strait_counts(coder):
    bit_count = coder.bit_count
    s_counts = np.zeros((2, bit_count), dtype=np.uint32)
    for s in coder.straits:
        s_counts[s.bit, s.abs_pos] += 1
    print(f"Strait count bins (pos: bit0/bit1): "
          f"{[(p, int(s_counts[0, p]), int(s_counts[1, p])) for p in range(bit_count) if s_counts[:, p].any()]}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(bit_count)
    width = 0.4
    ax.bar(positions - width / 2, s_counts[0], width, label="bit=0")
    ax.bar(positions + width / 2, s_counts[1], width, label="bit=1")
    ax.set_title("Strait counts per abs bit pos and bit value")
    ax.set_xlabel("abs bit pos")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig("strait_counts.png")
    plt.show()
