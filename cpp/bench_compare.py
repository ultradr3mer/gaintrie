"""Python vs C++ (gaintrie CLI): encode / decode time and bits/value on real bins.

Run: python cpp/bench_compare.py   (needs cpp/build/gaintrie)
Times are in-process (Python) vs the CLI's own encode/decode time (file I/O excluded).
"""
import contextlib
import io
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from gaintrie.GainCoder import GainCoder
from gaintrie.matrix_coder import encode_matrix, decode_matrix, GLOBAL, HYBRID
from gaintrie.pair_writer import best_writer, decode as pair_decode
from gaintrie.sandbox_paths import sandbox_path

CLI = ROOT / "cpp" / "build" / "gaintrie"


def py_pair(x):
    t = time.perf_counter()
    values, counts = np.unique(x, return_counts=True)
    with contextlib.redirect_stdout(io.StringIO()):
        data = best_writer(GainCoder(values, counts, 16), x)[1]
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    assert np.array_equal(pair_decode(data), x.astype(np.uint64))
    return len(data), t_enc, time.perf_counter() - t


def py_matrix(x, mode, bin_size):
    t = time.perf_counter()
    data, _ = encode_matrix(x, 16, mode, bin_size)
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    assert np.array_equal(decode_matrix(data), x.astype(np.uint64))
    return len(data), t_enc, time.perf_counter() - t


def cpp(path, args, fmt):
    with tempfile.TemporaryDirectory() as tmp:
        out, back = Path(tmp) / "x.cle", Path(tmp) / "x.out"
        enc = subprocess.run([str(CLI), "encode", "--format", fmt, "--width", "16", *args, "--stats", path, out],
                             capture_output=True, text=True, check=True).stderr
        dec = subprocess.run([str(CLI), "decode", "--format", fmt, "--stats", out, back],
                             capture_output=True, text=True, check=True).stderr
        assert back.read_bytes() == Path(path).read_bytes()
        t = lambda s: float(re.search(r" time=([\d.]+)", s).group(1))
        return out.stat().st_size, t(enc), t(dec)


def main():
    rows = []
    for name, label, fmt, mode, size in (
            ("model.layers.0.input_layernorm.weight.bin", "layers.0.input_layernorm", "pair", None, None),
            ("model.layers.0.self_attn.k_proj.weight.bin", "layers.0.k_proj", "matrix", GLOBAL, 1 << 20),
            ("model.layers.0.self_attn.k_proj.weight.bin", "layers.0.k_proj", "matrix", HYBRID, 1 << 20),
            ("model.layers.0.mlp.gate_proj.weight.bin", "layers.0.gate_proj", "matrix", GLOBAL, 1 << 20)):
        path = sandbox_path("bins", name)
        x = np.fromfile(path, dtype=np.uint16)
        cfg = "pair" if fmt == "pair" else "global" if mode == GLOBAL else "hybrid 1M"
        args = [] if fmt == "pair" else ["--mode", "global" if mode == GLOBAL else "hybrid", "--bin-size", str(size)]
        c = cpp(path, args, fmt)
        p = py_pair(x) if fmt == "pair" else py_matrix(x, mode, size)
        rows.append((label, cfg, len(x), p, c))
        print(f"{label} {cfg}: python {p}, c++ {c}", flush=True)
    print()
    print("| input | config | values | bits/value py | bits/value c++ | enc py s | enc c++ s | dec py s | dec c++ s |"
          " enc speedup | dec speedup |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, cfg, n, p, c in rows:
        print(f"| {label} | {cfg} | {n:,} | {8 * p[0] / n:.4f} | {8 * c[0] / n:.4f} | {p[1]:.2f} | {c[1]:.3f} | "
              f"{p[2]:.2f} | {c[2]:.3f} | {p[1] / c[1]:.0f}x | {p[2] / c[2]:.0f}x |")


if __name__ == "__main__":
    main()
