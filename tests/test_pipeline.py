"""End-to-end test of gaintrie/pipeline.py on small hand-built safetensors shards.

Run: python tests\\test_pipeline.py
"""
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gaintrie.pipeline import compress, decompress, read_index, CLI

rng = np.random.default_rng(11)


def bf16_like(n):
    """weight-like bf16 bit patterns: float32 normals truncated to the upper 16 bits"""
    return (rng.normal(0, 0.02, n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_shard(path: Path, tensors: dict, metadata=None):
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name, (dtype, shape, data) in tensors.items():
        raw = data.tobytes()
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, run = tmp / "src", tmp / "run"
        src.mkdir()
        write_shard(src / "model-00001-of-00002.safetensors", {
            "model.layers.0.input_layernorm.weight": ("BF16", [4096], (1 + 0.1 * rng.standard_normal(4096)).astype(np.float32).view(np.uint32).__rshift__(16).astype(np.uint16)),
            "model.layers.0.self_attn.k_proj.weight": ("BF16", [64, 3000], bf16_like(192000)),
            "model.layers.0.self_attn.q_norm.weight": ("BF16", [128], bf16_like(128)),
            "model.empty": ("BF16", [0], np.zeros(0, dtype=np.uint16)),
        }, metadata={"format": "pt"})
        write_shard(src / "model-00002-of-00002.safetensors", {
            "model.layers.1.mlp.gate_proj.weight": ("BF16", [300, 1000], bf16_like(300000)),
            "model.scale": ("F32", [5000], rng.normal(0, 1, 5000).astype(np.float32)),
            "model.position_ids": ("I64", [10], np.arange(10, dtype=np.int64)),
            "model.constant": ("BF16", [2000], np.full(2000, 0x3F80, dtype=np.uint16)),
        })
        archive = run / "model.clar"
        comp = compress(src, archive, run, bin_size=1 << 16, workers=4)
        dec = decompress(archive, run / "restored", run, workers=4)
        for shard in src.glob("*.safetensors"):
            assert (run / "restored" / shard.name).read_bytes() == shard.read_bytes(), shard.name
        assert all(c["matches_archive"] for c in dec["checks"])
        kinds = {u["kind"] for e in read_index(archive)["shards"] for t in e["tensors"] for u in t["units"]}
        assert "raw" in kinds and kinds & {"pair", "matrix"}
        assert comp["archive_bytes"] < comp["input_bytes"]
        for f in ("compress.csv", "tensors.csv", "decompress.csv"):
            assert (run / f).exists()


TESTS = [test_roundtrip]

if __name__ == "__main__":
    assert CLI.exists(), f"build the C++ CLI first: {CLI}"
    for t in TESTS:
        t()
        print(f"{t.__name__}: OK")
    print(f"{len(TESTS)} tests passed")
