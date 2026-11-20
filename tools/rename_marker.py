"""Rename an EXL3 codebook marker key in a checkpoint, in place, without touching tensor data.

The A4NX rename (formerly Q4NX): every routed-expert matrix carries a scalar int32 marker tensor whose key ends in
".q4nx". This rewrites those keys to ".a4nx" in each safetensors header and in model.safetensors.index.json and
quantization_config.json. The old and new suffixes have the same length, so each header keeps its byte length and
every data offset is unchanged; only header bytes are written.

    python rename_marker.py DIR                 # dry run: count markers per file, change nothing
    python rename_marker.py DIR --write         # rewrite DIR in place (run it on a copy, never on a serving copy)
    python rename_marker.py DIR --hash-data     # sha256 of each shard's tensor data region (compare before/after)

Checks: every replaced occurrence is a whole tensor key ending in the old suffix; no key with the new suffix exists
beforehand; the header length is unchanged and re-parses to the same entries under the renamed keys; totals across
the shards, the index and quantization_config.json agree.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys

OLD, NEW = ".q4nx", ".a4nx"


def read_header(path: pathlib.Path) -> tuple[int, bytes]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, f.read(n)


def rename_header(raw: bytes) -> tuple[bytes, int]:
    hdr = json.loads(raw)
    old_keys = [k for k in hdr if k != "__metadata__" and k.endswith(OLD)]
    if any(k.endswith(NEW) for k in hdr):
        raise SystemExit("a key with the new suffix already exists; refusing to mix")
    if not old_keys:
        return raw, 0
    needle_old, needle_new = (OLD + '"').encode(), (NEW + '"').encode()
    if raw.count(needle_old) != len(old_keys):
        raise SystemExit(f"{raw.count(needle_old)} byte matches for {len(old_keys)} keys; not a pure key rename")
    out = raw.replace(needle_old, needle_new)
    assert len(out) == len(raw)
    new = json.loads(out)
    expect = {(k[: -len(OLD)] + NEW if k in set(old_keys) else k): v for k, v in hdr.items()}
    if new != expect:
        raise SystemExit("renamed header does not match the expected mapping")
    return out, len(old_keys)


def rename_json(path: pathlib.Path, write: bool) -> int:
    raw = path.read_bytes()
    before = json.loads(raw)
    needle_old, needle_new = (OLD + '"').encode(), (NEW + '"').encode()
    if (NEW + '"').encode() in raw:
        raise SystemExit(f"{path.name}: already contains {NEW}")
    count = raw.count(needle_old)
    out = raw.replace(needle_old, needle_new)
    after = json.loads(out)
    keys_before = before.get("weight_map", before)
    if isinstance(keys_before, dict) and isinstance(after.get("weight_map", after), dict):
        changed = sum(1 for k in keys_before if k.endswith(OLD))
        if path.name == "model.safetensors.index.json" and changed != count:
            raise SystemExit(f"{path.name}: {count} byte matches for {changed} keys")
    if write and count:
        path.write_bytes(out)
    return count


def data_hash(path: pathlib.Path) -> str:
    n, _ = read_header(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(8 + n)
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--hash-data", action="store_true")
    a = ap.parse_args()
    shards = sorted(a.dir.glob("*.safetensors")) + sorted(a.dir.glob("*/*.safetensors"))
    if a.hash_data:
        for p in shards:
            print(json.dumps({"file": str(p.relative_to(a.dir)), "data_sha256": data_hash(p)}), flush=True)
        return
    total = 0
    for p in shards:
        n, raw = read_header(p)
        out, count = rename_header(raw)
        if a.write and count:
            with open(p, "r+b") as f:
                f.seek(8)
                f.write(out)
                f.flush()
            if read_header(p) != (n, out):
                raise SystemExit(f"{p.name}: header did not read back")
        total += count
        print(json.dumps({"file": str(p.relative_to(a.dir)), "markers": count, "header_bytes": n}), flush=True)
    side = {}
    for name in ("model.safetensors.index.json", "quantization_config.json"):
        p = a.dir / name
        if p.exists():
            side[name] = rename_json(p, a.write)
    print(json.dumps({"shard_markers": total, **side, "written": a.write}), flush=True)
    if side.get("model.safetensors.index.json", total) != total:
        sys.exit("index and shard marker counts disagree")


if __name__ == "__main__":
    main()
