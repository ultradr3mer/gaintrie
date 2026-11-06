"""Baselines on the same safetensors shards as the gaintrie pipeline run.

measured   general-purpose compressors on the shard files (headers included):
           zstd -3 / -19, xz -6, gzip -9 (pigz), bzip2 -9 — compressed bytes counted from stdout
computed   per bf16 tensor, from value counts (no codec written):
             exponent Huffman  the DFloat11 / ZipNN idea: Huffman over the 8 exponent bits,
                               sign + mantissa (8 bits) raw, + a 256-entry length table
             entropy           order-0 entropy of the 16-bit values, no table (lower bound for
                               any coder that codes values independently)
Usage: python research/compare_baselines.py [shard dir] [run dir]   (results -> run dir/baselines.json)
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import struct
import subprocess
import sys
import time

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.code_utils import huffman_lengths
from gaintrie.sandbox_paths import sandbox_path

TOOLS = {
    "zstd -3": ["zstd", "-3", "-T4", "-c"],
    "zstd -19": ["zstd", "-19", "-T4", "-c"],
    "xz -6": ["xz", "-6", "-T4", "-c"],
    "gzip -9": ["pigz", "-9", "-p", "4", "-c"],
    "bzip2 -9": ["bzip2", "-9", "-c"],
}


def compressed_size(cmd, path: Path) -> int:
    with path.open("rb") as f:
        p = subprocess.Popen(cmd, stdin=f, stdout=subprocess.PIPE)
        size = 0
        for block in iter(lambda: p.stdout.read(1 << 22), b""):
            size += len(block)
        assert p.wait() == 0, cmd
    return size


def measure_tool(name, shards):
    t = time.time()
    with ThreadPoolExecutor(len(shards)) as pool:
        size = sum(pool.map(lambda s: compressed_size(TOOLS[name], s), shards))
    return size, time.time() - t


def computed(shards):
    exp_bits = ent_bits = values = other = 0
    for shard in shards:
        with shard.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        other += base
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            if info["dtype"] != "BF16":
                other += end - start
                continue
            x = np.memmap(shard, dtype=np.uint16, mode="r", offset=base + start, shape=((end - start) // 2,))
            counts = np.bincount(x, minlength=1 << 16)
            exp = np.bincount(np.arange(1 << 16) >> 7 & 0xFF, weights=counts, minlength=256).astype(np.int64)
            used = exp[exp > 0]
            lengths = huffman_lengths(used.tolist())
            exp_bits += int(sum(l * c for l, c in zip(lengths, used.tolist()))) + 8 * len(x) + 256 * 5
            c = counts[counts > 0].astype(np.float64)
            ent_bits += float((c * np.log2(len(x) / c)).sum())
            values += len(x)
    return {"exponent Huffman": other + exp_bits / 8, "entropy": other + ent_bits / 8, "values": values}


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else sandbox_path("qwen3-8b")
    run = Path(sys.argv[2]) if len(sys.argv) > 2 else sandbox_path("qwen3-8b-run")
    shards = sorted(src.glob("*.safetensors"))
    raw = sum(s.stat().st_size for s in shards)
    results = {"input_bytes": raw}
    c = computed(shards)
    values = c.pop("values")
    for name, size in c.items():
        results[name] = {"bytes": int(size), "seconds": None}
        print(f"{name:<18} {size / 1e9:8.3f} GB  {size / raw:.4f}  {size * 8 / values:6.3f} bits/weight (computed)", flush=True)
    for name in TOOLS:
        size, secs = measure_tool(name, shards)
        results[name] = {"bytes": size, "seconds": round(secs, 1)}
        print(f"{name:<18} {size / 1e9:8.3f} GB  {size / raw:.4f}  {size * 8 / values:6.3f} bits/weight  {secs:.0f}s",
              flush=True)
        (run / "baselines.json").write_text(json.dumps(results, indent=1))
    print("END")
