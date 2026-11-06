"""How much does one global order of the bit positions save on the split / strait positions?

Per bin: sum of all written position indices in MSB order (today), in the order by
average definition time and in the optimized order (pair_writer.position_order);
the position bits (MSB order / stored order, truncated binary) and the bin total with MSB positions vs the
best mode (written). Layer norms: one bin per layer; big matrices: sampled bins.
"""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.GainCoder import GainCoder
from research.matrix_bins import sample_bins
from gaintrie.pair_writer import (PairWriter, best_writer, position_order, position_costs, order_cost, defined_pos,
                                  RAW, AUTO, POS_MSB, POS_ORDER, SPLIT, STRAIT)
from gaintrie.sandbox_paths import sandbox_path

BITS = 16
MODES = {POS_MSB: "msb", POS_ORDER: "order"}


def bin_report(b) -> dict:
    values, counts = np.unique(b, return_counts=True)
    coder = GainCoder(values, counts, BITS)
    pw = PairWriter(coder, RAW)
    nodes = list(pw.walk())
    order, W, start = position_order(pw, nodes)
    msb_sum = sum(rem.index(defined_pos(pw, t, rem)) for t, rem, _, k in nodes if k in (SPLIT, STRAIT))
    assert msb_sum == order_cost(range(BITS), W)
    costs = position_costs(pw, nodes)
    _, _, msb_stats = best_writer(coder, b, AUTO, POS_MSB)
    _, _, best_stats = best_writer(coder, b, AUTO)
    defs = sum(1 for *_, k in nodes if k in (SPLIT, STRAIT))
    return {"defs": defs, "sum_msb": msb_sum, "sum_avg": order_cost(start, W), "sum_opt": order_cost(order, W),
            **{MODES[m]: c[0] for m, c in costs.items()},
            "total_msb": sum(msb_stats.values()), "total_best": sum(best_stats.values()), "n": len(b)}


def print_rows(title, rows):
    print(f"######## {title}")
    print(f"{'bin':<14}{'defs':>6} | {'Σidx msb':>9}{'avg-time':>9}{'optimized':>10} | "
          + "".join(f"{v:>7}" for v in MODES.values()) + f" | {'total msb':>10}{'best':>8}{'saved':>7}{'bits/val':>9}")
    acc = Counter()
    for label, r in rows:
        acc.update(r)
        print(f"{label:<14}{r['defs']:>6} | {r['sum_msb']:>9}{r['sum_avg']:>9}{r['sum_opt']:>10} | "
              + "".join(f"{r[v]:>7}" for v in MODES.values())
              + f" | {r['total_msb']:>10}{r['total_best']:>8}{r['total_msb'] - r['total_best']:>7}"
              f"{r['total_best'] / r['n']:>9.3f}", flush=True)
    r = acc
    print(f"{'sum':<14}{r['defs']:>6} | {r['sum_msb']:>9}{r['sum_avg']:>9}{r['sum_opt']:>10} | "
          + "".join(f"{r[v]:>7}" for v in MODES.values())
          + f" | {r['total_msb']:>10}{r['total_best']:>8}{r['total_msb'] - r['total_best']:>7}"
          f"{r['total_best'] / r['n']:>9.3f}")


if __name__ == "__main__":
    for kind in ("input_layernorm", "post_attention_layernorm"):
        paths = sorted(sandbox_path("bins").glob(f"model.layers.*.{kind}.weight.bin"), key=lambda p: int(p.name.split(".")[2]))
        print_rows(f"{kind}, one bin per layer",
                   [(f"layer {p.name.split('.')[2]}", bin_report(np.fromfile(p, dtype=np.uint16))) for p in paths])
    x = np.memmap(sandbox_path("bins", "model.layers.0.mlp.gate_proj.weight.bin"), dtype=np.uint16, mode="r")
    for size, count in ((4096, 4), (65536, 2), (1048576, 1)):
        print_rows(f"mlp.gate_proj layer 0, {count} sampled bins of {size}",
                   [(f"bin {i}", bin_report(b)) for i, b in enumerate(sample_bins(x, size, count))])
    print("END")
