"""The C++ port (cpp/, CLI `gaintrie`) and the Python writers decode each other's files.

Build: cmake -S cpp -B cpp/build && cmake --build cpp/build   (or set GAINTRIE_CLI)
Run: python tests/test_cpp_compat.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.matrix_coder import encode_matrix, decode_matrix, GLOBAL, HYBRID
from gaintrie.pair_writer import best_writer, decode as pair_decode
from gaintrie.sandbox_paths import sandbox_path
from tests.test_tree_writer import CASES, make

CLI = Path(os.environ.get("GAINTRIE_CLI", Path(__file__).resolve().parent.parent / "cpp" / "build" / "gaintrie"))
SIZES = []  # (label, python bytes, c++ bytes)


def dtype_for(width):
    return np.uint8 if width <= 8 else np.uint16 if width <= 16 else np.uint32 if width <= 32 else np.uint64


def cli(*args):
    assert CLI.exists(), f"{CLI} missing - build it: cmake -S cpp -B cpp/build && cmake --build cpp/build"
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, f"gaintrie {' '.join(map(str, args))}: {r.stderr}"


def pair_encode(x, width):
    with contextlib.redirect_stdout(io.StringIO()):
        return best_writer(make(x, width), x)[1]


def both_ways(label, x, width, fmt, py_encode, py_decode, cli_args=()):
    """Python file -> C++ decode, C++ file -> Python decode; records both sizes"""
    dtype = dtype_for(width)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        x.astype(dtype).tofile(tmp / "x.bin")
        py = np.asarray(py_encode(x), dtype=np.uint8)
        py.tofile(tmp / "py.cle")
        cli("decode", "--format", fmt, tmp / "py.cle", tmp / "py.out")
        assert np.array_equal(np.fromfile(tmp / "py.out", dtype=dtype), x.astype(dtype)), f"{label}: C++ decode"
        cli("encode", "--format", fmt, "--width", width, *cli_args, tmp / "x.bin", tmp / "cpp.cle")
        cpp = np.fromfile(tmp / "cpp.cle", dtype=np.uint8)
        assert np.array_equal(py_decode(cpp), x.astype(np.uint64)), f"{label}: Python decode"
    SIZES.append((label, len(py), len(cpp), np.array_equal(py, cpp)))


def pair_both_ways(label, x, width):
    both_ways(label, x, width, "pair", lambda v: pair_encode(v, width), pair_decode)


def matrix_both_ways(label, x, width, mode, bin_size):
    both_ways(label, x, width, "matrix", lambda v: encode_matrix(v, width, mode, bin_size)[0], decode_matrix,
              ("--mode", "global" if mode == GLOBAL else "hybrid", "--bin-size", bin_size))


def test_cases_pair():
    for i, (x, width) in enumerate(CASES):
        pair_both_ways(f"case{i} w={width} pair", x, width)


def test_cases_matrix():
    for i, (x, width) in enumerate(CASES):
        for mode, size in ((GLOBAL, 1 << 20), (HYBRID, 3), (HYBRID, 64), (HYBRID, 1 << 20)):
            name = "global" if mode == GLOBAL else f"hybrid {size}"
            matrix_both_ways(f"case{i} w={width} {name}", x, width, mode, size)


def test_layernorms():
    paths = sorted(sandbox_path("bins").glob("model.layers.*_layernorm.weight.bin"),
                   key=lambda p: (int(p.name.split(".")[2]), p.name))
    if not paths:
        print("test_layernorms: SKIPPED (bin files missing)")
        return
    for path in paths:
        x = np.fromfile(path, dtype=np.uint16)
        pair_both_ways(f"{path.name.removeprefix('model.').removesuffix('.weight.bin')} pair", x, 16)


def test_k_proj():
    path = sandbox_path("bins", "model.layers.0.self_attn.k_proj.weight.bin")
    if not path.exists():
        print("test_k_proj: SKIPPED (bin file missing)")
        return
    x = np.fromfile(path, dtype=np.uint16)
    matrix_both_ways("layers.0.k_proj global", x, 16, GLOBAL, 1 << 20)
    matrix_both_ways("layers.0.k_proj hybrid 1M", x, 16, HYBRID, 1 << 20)


def print_sizes():
    print(f"{'input':<40}{'python B':>12}{'c++ B':>12}{'diff':>8}  identical")
    for label, py, cpp, same in SIZES:
        print(f"{label:<40}{py:>12}{cpp:>12}{cpp - py:>+8}  {'yes' if same else 'no'}")
    print(f"identical files: {sum(s[3] for s in SIZES)}/{len(SIZES)}")


TESTS = [test_cases_pair, test_cases_matrix, test_layernorms, test_k_proj]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK", flush=True)
    print_sizes()
    print(f"{len(TESTS)} tests passed")
