"""End-to-end model compression: safetensors shards -> one archive -> identical shards.

compress    read every shard's header (kept verbatim) and tensors; per tensor two
            candidates, the smaller one is stored:
              bins    the tensor cut into bins of `bin_size` values, each its own
                      pair_writer file (own tree + Huffman table)
              matrix  one matrix_coder container (global Huffman table)
            dtypes without a codec width are stored raw. Units are encoded by the
            C++ CLI (cpp/build/gaintrie) in parallel and appended to the archive.
decompress  decode every unit, write it at its offset into the rebuilt shard,
            verify the shards' SHA-256 (recorded at compression + sha256.txt if present).

Archive: b"CLAR1" | unit blobs | index JSON | index length:8 (little endian).
Metrics: <run dir>/compress.csv (one row per unit), tensors.csv (both candidates),
decompress.csv, summary.json; summaries per tensor type, per layer and total.

Usage: python pipeline.py [roundtrip|compress|decompress] ... (no args: Qwen3-8B round trip)
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import base64
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.sandbox_paths import sandbox_path

CLI = Path(__file__).resolve().parent.parent / "cpp" / "build" / "gaintrie"
MAGIC = b"CLAR1"
WIDTHS = {"BF16": 16, "F16": 16, "F32": 32, "U8": 8, "I8": 8, "U16": 16, "I16": 16, "U32": 32, "I32": 32}
ITEM = {16: np.uint16, 32: np.uint32, 8: np.uint8}


def read_header(shard: Path):
    with shard.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return raw, json.loads(raw), 8 + n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def tensor_type(name: str):
    m = re.match(r"^model\.layers\.(\d+)\.(.*)$", name)
    return (m.group(2), int(m.group(1))) if m else (name, None)


class CliError(RuntimeError):
    pass


def run_cli(*args):
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise CliError(r.stderr.strip() or f"gaintrie exit {r.returncode}")


def encode_unit(values: np.ndarray, width: int, fmt: str, work: Path) -> bytes:
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.cle"
        values.tofile(src)
        extra = ["--mode", "global"] if fmt == "matrix" else []
        run_cli("encode", "--format", fmt, "--width", width, *extra, src, dst)
        return dst.read_bytes()


def decode_unit(blob: bytes, fmt: str, width: int, work: Path) -> bytes:
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        src, dst = Path(tmp) / "in.cle", Path(tmp) / "out.bin"
        src.write_bytes(blob)
        run_cli("decode", "--format", fmt, "--out-bytes", width // 8, src, dst)
        return dst.read_bytes()


def compress_tensor(shard: Path, base: int, name: str, info: dict, bin_size: int, work: Path):
    """-> (tensor record without blob offsets, [(unit record, blob)], candidate sizes)"""
    start, end = info["data_offsets"]
    width = WIDTHS.get(info["dtype"])
    raw = np.memmap(shard, dtype=np.uint8, mode="r", offset=base + start, shape=(end - start,)) if end > start else None
    record = {"name": name, "dtype": info["dtype"], "shape": info["shape"], "data_offsets": [start, end]}
    if raw is None or width is None or (end - start) % (width // 8):
        return record, [({"kind": "raw", "start": 0, "values": end - start, "width": 8},
                         bytes(raw) if raw is not None else b"", 0.0)], {}
    values = np.asarray(raw).view(ITEM[width])
    candidates = {}
    t = time.time()
    bins = []
    for s in range(0, len(values), bin_size):
        tb = time.time()
        blob = encode_unit(values[s:s + bin_size], width, "pair", work)
        bins.append(({"kind": "pair", "start": s, "values": min(bin_size, len(values) - s), "width": width},
                     blob, time.time() - tb))
    candidates["bins"] = (sum(len(b) for _, b, _ in bins), time.time() - t, bins)
    t = time.time()
    try:  # one stream segment must stay below 2^32 bits (matrix_coder LEN_BITS)
        blob = encode_unit(values, width, "matrix", work)
        dt = time.time() - t
        candidates["matrix"] = (len(blob), dt, [({"kind": "matrix", "start": 0, "values": len(values),
                                                  "width": width}, blob, dt)])
    except CliError as e:
        record["matrix_error"] = str(e)
    best = min(candidates, key=lambda c: candidates[c][0])
    record["choice"] = best
    sizes = {c: (v[0], v[1]) for c, v in candidates.items()}
    return record, candidates[best][2], sizes


def compress(shard_dir: Path, archive: Path, run_dir: Path, bin_size=1 << 20, workers=8):
    run_dir.mkdir(parents=True, exist_ok=True)
    work = run_dir / "work"
    work.mkdir(exist_ok=True)
    shards = sorted(shard_dir.glob("*.safetensors"))
    t0 = time.time()
    index = {"bin_size": bin_size, "shards": []}
    jobs = []
    for shard in shards:
        raw, header, base = read_header(shard)
        entry = {"file": shard.name, "size": shard.stat().st_size, "header": base64.b64encode(raw).decode(),
                 "tensors": []}
        index["shards"].append(entry)
        for name, info in header.items():
            if name != "__metadata__":
                jobs.append((entry, shard, base, name, info))
    hashes = {}
    hasher = threading.Thread(target=lambda: hashes.update({s.name: sha256_file(s) for s in shards}))
    hasher.start()

    unit_rows, tensor_rows = [], []
    lock = threading.Lock()
    with archive.open("wb") as out, ThreadPoolExecutor(workers) as pool:
        out.write(MAGIC)
        offset = len(MAGIC)
        futures = {pool.submit(compress_tensor, shard, base, name, info, bin_size, work): (entry, name)
                   for entry, shard, base, name, info in jobs}
        for i, fut in enumerate(as_completed(futures)):
            entry, name = futures[fut]
            record, units, sizes = fut.result()
            kind, layer = tensor_type(name)
            record["units"] = []
            with lock:
                for u, blob, enc_s in units:
                    out.write(blob)
                    record["units"].append({**u, "offset": offset, "length": len(blob)})
                    offset += len(blob)
                    raw_bytes = u["values"] * u["width"] // 8
                    unit_rows.append({"tensor": name, "type": kind, "layer": layer, "kind": u["kind"],
                                      "start": u["start"], "values": u["values"], "raw_bytes": raw_bytes,
                                      "comp_bytes": len(blob),
                                      "bits_per_value": round(len(blob) * 8 / u["values"], 4) if u["values"] else 0,
                                      "ratio": len(blob) / raw_bytes if raw_bytes else 0, "enc_s": round(enc_s, 4)})
                entry["tensors"].append(record)
            raw_bytes = record["data_offsets"][1] - record["data_offsets"][0]
            tensor_rows.append({"tensor": name, "type": kind, "layer": layer, "dtype": record["dtype"],
                                "values": raw_bytes // max(1, (WIDTHS.get(record["dtype"], 8) // 8)),
                                "raw_bytes": raw_bytes, "choice": record.get("choice", "raw"),
                                "bins_bytes": sizes.get("bins", (0, 0))[0], "bins_s": round(sizes.get("bins", (0, 0))[1], 3),
                                "matrix_bytes": sizes.get("matrix", (0, 0))[0],
                                "matrix_s": round(sizes.get("matrix", (0, 0))[1], 3),
                                "stored_bytes": sum(u["length"] for u in record["units"]),
                                "note": record.get("matrix_error", "")})
            if (i + 1) % 25 == 0 or i + 1 == len(futures):
                done = sum(r["raw_bytes"] for r in tensor_rows)
                stored = sum(r["stored_bytes"] for r in tensor_rows)
                print(f"  compressed {i + 1}/{len(futures)} tensors, {done / 1e9:.2f} GB -> {stored / 1e9:.2f} GB "
                      f"({stored / max(done, 1):.4f}), {time.time() - t0:.0f}s", flush=True)
        hasher.join()
        for entry in index["shards"]:
            entry["sha256"] = hashes[entry["file"]]
            entry["tensors"].sort(key=lambda r: r["data_offsets"][0])
        data = json.dumps(index).encode()
        out.write(data)
        out.write(struct.pack("<Q", len(data)))
    t_compress = time.time() - t0
    work.rmdir()

    write_csv(run_dir / "compress.csv", unit_rows)
    write_csv(run_dir / "tensors.csv", sorted(tensor_rows, key=lambda r: r["tensor"]))
    return {"compress_s": t_compress, "input_bytes": sum(s.stat().st_size for s in shards),
            "archive_bytes": archive.stat().st_size, "units": unit_rows, "tensors": tensor_rows}


def read_index(archive: Path) -> dict:
    with archive.open("rb") as f:
        assert f.read(len(MAGIC)) == MAGIC, "not a CLAR1 archive"
        f.seek(-8, os.SEEK_END)
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(-8 - n, os.SEEK_END)
        return json.loads(f.read(n))


def decompress(archive: Path, out_dir: Path, run_dir: Path, workers=8, reference: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    index = read_index(archive)
    jobs = []
    for entry in index["shards"]:
        path = out_dir / entry["file"]
        header = base64.b64decode(entry["header"])
        with path.open("wb") as f:
            f.write(struct.pack("<Q", len(header)) + header)
            f.truncate(entry["size"])
        base = 8 + len(header)
        for record in entry["tensors"]:
            for u in record["units"]:
                jobs.append((path, base + record["data_offsets"][0], record["name"], u))

    def work_unit(path, tensor_base, name, u):
        t = time.time()
        with archive.open("rb") as f:
            f.seek(u["offset"])
            blob = f.read(u["length"])
        data = blob if u["kind"] == "raw" else decode_unit(blob, u["kind"], u["width"], work)
        assert len(data) == u["values"] * u["width"] // 8, f"{name}: wrong decoded length"
        with path.open("r+b") as f:
            f.seek(tensor_base + u["start"] * u["width"] // 8)
            f.write(data)
        return {"tensor": name, "kind": u["kind"], "start": u["start"], "values": u["values"],
                "comp_bytes": u["length"], "dec_s": round(time.time() - t, 4)}

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(work_unit, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i + 1) % 500 == 0 or i + 1 == len(futures):
                print(f"  decoded {i + 1}/{len(futures)} units, {time.time() - t0:.0f}s", flush=True)
    t_decompress = time.time() - t0
    work.rmdir()

    t0 = time.time()
    expected = {}
    if reference is not None and (reference / "sha256.txt").exists():
        for line in (reference / "sha256.txt").read_text().splitlines():
            digest, file = line.split()
            expected[file] = digest
    checks = []
    for entry in index["shards"]:
        got = sha256_file(out_dir / entry["file"])
        ref = expected.get(entry["file"])
        checks.append({"file": entry["file"], "sha256": got, "matches_archive": got == entry["sha256"],
                       "matches_reference": None if ref is None else ref == got})
    write_csv(run_dir / "decompress.csv", rows)
    return {"decompress_s": t_decompress, "verify_s": time.time() - t0, "checks": checks}


def write_csv(path: Path, rows: list):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(comp: dict, dec: dict | None, run_dir: Path):
    units, tensors = comp["units"], comp["tensors"]
    groups = defaultdict(lambda: [0, 0, 0, 0, 0])  # tensors, values, raw, stored, bins
    for r in tensors:
        g = groups[r["type"]]
        g[0] += 1
        g[1] += r["values"]
        g[2] += r["raw_bytes"]
        g[3] += r["stored_bytes"]
        g[4] += r["choice"] == "bins"
    print("==Per tensor type== (stored = smaller candidate)")
    print(f"{'type':<34}{'tensors':>8}{'values':>15}{'raw GB':>9}{'stored GB':>11}{'ratio':>8}{'bits/v':>8}"
          f"{'bins won':>10}")
    for kind, (n, v, raw, stored, bins) in sorted(groups.items(), key=lambda kv: -kv[1][2]):
        print(f"{kind:<34}{n:>8}{v:>15,}{raw / 1e9:>9.3f}{stored / 1e9:>11.3f}{stored / raw:>8.4f}"
              f"{stored * 8 / v:>8.3f}{bins:>7}/{n}")
    layers = defaultdict(lambda: [0, 0])
    for r in tensors:
        if r["layer"] is not None:
            layers[r["layer"]][0] += r["raw_bytes"]
            layers[r["layer"]][1] += r["stored_bytes"]
    print("==Per layer== ratio: " + "  ".join(f"{l}:{s / raw:.4f}" for l, (raw, s) in sorted(layers.items())))
    bpv = np.array([u["bits_per_value"] for u in units if u["kind"] != "raw"])
    print(f"==Per unit== {len(units)} units ({sum(u['kind'] == 'pair' for u in units)} bins, "
          f"{sum(u['kind'] == 'matrix' for u in units)} matrices, {sum(u['kind'] == 'raw' for u in units)} raw); "
          f"bits/value min {bpv.min():.3f}, median {np.median(bpv):.3f}, max {bpv.max():.3f}")
    total = {"input_bytes": comp["input_bytes"], "archive_bytes": comp["archive_bytes"],
             "ratio": comp["archive_bytes"] / comp["input_bytes"],
             "tensor_bytes": sum(r["raw_bytes"] for r in tensors),
             "tensor_stored_bytes": sum(r["stored_bytes"] for r in tensors),
             "candidate_bins_bytes": sum(r["bins_bytes"] or r["stored_bytes"] for r in tensors),
             "candidate_matrix_bytes": sum(r["matrix_bytes"] or r["stored_bytes"] for r in tensors),
             "compress_s": round(comp["compress_s"], 1)}
    if dec:
        total.update({"decompress_s": round(dec["decompress_s"], 1), "verify_s": round(dec["verify_s"], 1),
                      "lossless": all(c["matches_archive"] and c["matches_reference"] is not False
                                      for c in dec["checks"]), "checks": dec["checks"]})
    print("==Total==")
    n_values = sum(r["values"] for r in tensors)
    print(f"input {total['input_bytes'] / 1e9:.3f} GB -> archive {total['archive_bytes'] / 1e9:.3f} GB, "
          f"ratio {total['ratio']:.4f} ({total['archive_bytes'] * 8 / n_values:.3f} bits per value incl. "
          f"headers + index, raw {total['input_bytes'] * 8 / n_values:.3f})")
    print(f"only bins: {total['candidate_bins_bytes'] / 1e9:.3f} GB, only matrix: "
          f"{total['candidate_matrix_bytes'] / 1e9:.3f} GB, best per tensor: {total['tensor_stored_bytes'] / 1e9:.3f} GB")
    print(f"time: compress {total['compress_s']}s" + (f", decompress {total['decompress_s']}s, verify "
                                                      f"{total['verify_s']}s" if dec else ""))
    if dec:
        for c in dec["checks"]:
            print(f"  {c['file']}: archive sha256 {'OK' if c['matches_archive'] else 'MISMATCH'}, "
                  f"reference {'-' if c['matches_reference'] is None else 'OK' if c['matches_reference'] else 'MISMATCH'}")
        print("LOSSLESS" if total["lossless"] else "NOT LOSSLESS")
    (run_dir / "summary.json").write_text(json.dumps(total, indent=1))
    return total


def roundtrip(shard_dir: Path, run_dir: Path, bin_size=1 << 20, workers=8):
    archive, restored = run_dir / "model.clar", run_dir / "restored"
    print(f"######## compress {shard_dir} -> {archive}")
    comp = compress(shard_dir, archive, run_dir, bin_size, workers)
    print(f"######## decompress {archive} -> {restored}")
    dec = decompress(archive, restored, run_dir, workers, reference=shard_dir)
    return summarize(comp, dec, run_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "roundtrip":
        src = Path(args[1]) if len(args) > 1 else sandbox_path("qwen3-8b")
        run = Path(args[2]) if len(args) > 2 else sandbox_path("qwen3-8b-run")
        roundtrip(src, run)
    elif args[0] == "compress":
        run = Path(args[3]) if len(args) > 3 else Path(args[2]).parent
        summarize(compress(Path(args[1]), Path(args[2]), run), None, run)
    elif args[0] == "decompress":
        run = Path(args[3]) if len(args) > 3 else Path(args[2])
        print(decompress(Path(args[1]), Path(args[2]), run)["checks"])
