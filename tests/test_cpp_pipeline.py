"""C++ pipeline (gaintrie compress/decompress/roundtrip) vs gaintrie/pipeline.py: archives decode both ways.

Build: cmake -S cpp -B cpp/build && cmake --build cpp/build
Run: python tests/test_cpp_pipeline.py
"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.pipeline import compress, decompress, read_index, CLI
from tests.test_pipeline import bf16_like, rng, write_shard

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
TIMES = {"enc_s", "bins_s", "matrix_s", "dec_s"}


def make_model(src: Path):
    """the shards of tests/test_pipeline.py plus a BOOL tensor and a larger matrix-friendly one"""
    src.mkdir(parents=True)
    write_shard(src / SHARDS[0], {
        "model.layers.0.input_layernorm.weight": ("BF16", [4096], (1 + 0.1 * rng.standard_normal(4096)).astype(np.float32).view(np.uint32).__rshift__(16).astype(np.uint16)),
        "model.layers.0.self_attn.k_proj.weight": ("BF16", [64, 3000], bf16_like(192000)),
        "model.layers.0.self_attn.q_norm.weight": ("BF16", [128], bf16_like(128)),
        "model.empty": ("BF16", [0], np.zeros(0, dtype=np.uint16)),
    }, metadata={"format": "pt"})
    write_shard(src / SHARDS[1], {
        "model.layers.1.mlp.gate_proj.weight": ("BF16", [300, 1000], bf16_like(300000)),
        "model.scale": ("F32", [5000], rng.normal(0, 1, 5000).astype(np.float32)),
        "model.position_ids": ("I64", [10], np.arange(10, dtype=np.int64)),
        "model.constant": ("BF16", [2000], np.full(2000, 0x3F80, dtype=np.uint16)),
        "model.mask": ("BOOL", [77], rng.integers(0, 2, 77).astype(np.uint8)),
    })
    (src / "sha256.txt").write_text("".join(f"{hashlib.sha256((src / s).read_bytes()).hexdigest()}  {s}\n"
                                            for s in SHARDS))


def cli(*args, expect=0):
    assert CLI.exists(), f"{CLI} missing - build it: cmake -S cpp -B cpp/build && cmake --build cpp/build"
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == expect, f"gaintrie {' '.join(map(str, args))}: exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout


def same_shards(a: Path, b: Path):
    for s in SHARDS:
        assert (a / s).read_bytes() == (b / s).read_bytes(), s


def kinds(archive: Path):
    return {u["kind"] for e in read_index(archive)["shards"] for t in e["tensors"] for u in t["units"]}


def rows(path: Path, key):
    with path.open(newline="") as f:
        r = list(csv.DictReader(f))
    return sorted(({k: v for k, v in row.items() if k not in TIMES} for row in r), key=key)


def test_cpp_compress_py_decompress():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_model(tmp / "src")
        archive = tmp / "run" / "model.clar"
        cli("compress", tmp / "src", archive, "--bin-size", 1 << 16, "--workers", 4)
        assert {"raw", "pair", "matrix"} <= kinds(archive), kinds(archive)
        dec = decompress(archive, tmp / "restored", tmp / "run", workers=4, reference=tmp / "src")
        same_shards(tmp / "src", tmp / "restored")
        assert all(c["matches_archive"] and c["matches_reference"] for c in dec["checks"])
        for f in ("compress.csv", "tensors.csv", "decompress.csv", "summary.json"):
            assert (tmp / "run" / f).exists(), f


def test_py_compress_cpp_decompress():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_model(tmp / "src")
        archive = tmp / "run" / "model.clar"
        compress(tmp / "src", archive, tmp / "run", bin_size=1 << 16, workers=4)
        assert {"raw", "pair", "matrix"} <= kinds(archive), kinds(archive)
        out = cli("decompress", archive, tmp / "restored", "--reference", tmp / "src", "--run-dir", tmp / "run")
        assert "LOSSLESS" in out and "NOT LOSSLESS" not in out and "MISMATCH" not in out, out
        same_shards(tmp / "src", tmp / "restored")


def test_same_archive_content():
    """both pipelines store the same units (same encoder); metrics files have the same columns and values"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_model(tmp / "src")
        compress(tmp / "src", tmp / "py" / "model.clar", tmp / "py", bin_size=1 << 16, workers=4)
        decompress(tmp / "py" / "model.clar", tmp / "py" / "restored", tmp / "py", workers=4)
        out = cli("roundtrip", tmp / "src", tmp / "cpp", "--bin-size", 1 << 16, "--workers", 4)
        assert "LOSSLESS" in out and "NOT LOSSLESS" not in out, out
        summary = json.loads((tmp / "cpp" / "summary.json").read_text())
        assert summary["lossless"] and all(c["matches_reference"] for c in summary["checks"])

        def units(run):
            index = read_index(run / "model.clar")
            with (run / "model.clar").open("rb") as f:
                got = {}
                for e in index["shards"]:
                    for t in e["tensors"]:
                        blobs = []
                        for u in t["units"]:
                            f.seek(u["offset"])
                            blobs.append((u["kind"], u["start"], u["values"], u["width"], f.read(u["length"])))
                        got[t["name"]] = ({k: v for k, v in t.items() if k != "units"}, blobs)
            return index, got

        py_index, py_units = units(tmp / "py")
        cpp_index, cpp_units = units(tmp / "cpp")
        assert py_units == cpp_units
        for p, c in zip(py_index["shards"], cpp_index["shards"]):
            assert {k: v for k, v in p.items() if k != "tensors"} == {k: v for k, v in c.items() if k != "tensors"}
            assert [t["name"] for t in p["tensors"]] == [t["name"] for t in c["tensors"]]
        assert py_index["bin_size"] == cpp_index["bin_size"]
        for f, key in (("compress.csv", lambda r: (r["tensor"], int(r["start"]))), ("tensors.csv", lambda r: r["tensor"]),
                       ("decompress.csv", lambda r: (r["tensor"], int(r["start"])))):
            assert (tmp / "py" / f).read_text().splitlines()[0] == (tmp / "cpp" / f).read_text().splitlines()[0], f
            assert rows(tmp / "py" / f, key) == rows(tmp / "cpp" / f, key), f


def test_cpp_detects_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_model(tmp / "src")
        archive = tmp / "run" / "model.clar"
        cli("compress", tmp / "src", archive, "--bin-size", 1 << 16)
        index = read_index(archive)
        raw = next(u for e in index["shards"] for t in e["tensors"] for u in t["units"] if u["kind"] == "raw" and u["length"])
        data = bytearray(archive.read_bytes())
        data[raw["offset"]] ^= 1
        archive.write_bytes(bytes(data))
        out = cli("decompress", archive, tmp / "restored", "--reference", tmp / "src", expect=3)
        assert "NOT LOSSLESS" in out and "MISMATCH" in out, out


TESTS = [test_cpp_compress_py_decompress, test_py_compress_cpp_decompress, test_same_archive_content,
         test_cpp_detects_corruption]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
