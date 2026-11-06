"""Splits .safetensors shards into one raw .bin per tensor (bytes as stored, e.g.
bf16 little-endian), named like the existing test data: <tensor name>.bin.

Usage: python split_safetensors.py <dir with *.safetensors> <out dir>
Standalone (no gaintrie imports); the header is JSON, the tensor names are
checked before they become file names.
"""
from pathlib import Path
import json
import re
import struct
import sys

NAME = re.compile(r"^[A-Za-z0-9_.]+$")
CHUNK = 1 << 24


def split(shard: Path, out: Path) -> list:
    rows = []
    with shard.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not NAME.match(name) or ".." in name:
                raise ValueError(f"unexpected tensor name {name!r}")
            start, end = info["data_offsets"]
            f.seek(base + start)
            left = end - start
            with (out / f"{name}.bin").open("wb") as o:
                while left:
                    block = f.read(min(CHUNK, left))
                    if not block:
                        raise ValueError(f"{shard.name}: {name} is truncated")
                    o.write(block)
                    left -= len(block)
            rows.append((name, info["dtype"], info["shape"], end - start))
    return rows


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for shard in sorted(src.glob("*.safetensors")) for r in split(shard, out)]
    dtypes = {}
    for _, dtype, _, size in rows:
        dtypes[dtype] = dtypes.get(dtype, 0) + size
    print(f"{len(rows)} tensors -> {out}, " + ", ".join(f"{d}: {s / 1e9:.2f} GB" for d, s in dtypes.items()))
    kinds = {}
    for name, dtype, shape, _ in rows:
        kind = re.sub(r"^model\.layers\.\d+\.", "model.layers.N.", name)
        kinds.setdefault(kind, (dtype, shape, 0))
        kinds[kind] = (dtype, shape, kinds[kind][2] + 1)
    for kind, (dtype, shape, count) in sorted(kinds.items()):
        print(f"  {count:>3} x {kind:<50} {dtype} {shape}")
