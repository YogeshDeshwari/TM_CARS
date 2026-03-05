#!/usr/bin/env python3
"""
Pack skin zip files into multi-part packs with size limits.

Default behavior (matches current workflow request):
  - Source: CH_all_skins/*.zip
  - Output: CH_Jan_pack_01.zip, CH_Jan_pack_02.zip, ...
  - Max 5 skin-zips per pack
  - Max 100 MB per pack (hard cap)

Notes:
  - Uses ZIP_STORED to avoid recompressing already-compressed .zip files.
  - The size cap is enforced based on input sizes + a small overhead estimate.
"""

from __future__ import annotations

import argparse
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    path: Path
    size: int


def _mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def _iter_skin_zips(src_dir: Path) -> list[Entry]:
    items: list[Entry] = []
    for p in sorted(src_dir.glob("*.zip")):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if not p.is_file():
            continue
        items.append(Entry(path=p, size=int(st.st_size)))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="CH_all_skins", help="Directory containing per-skin .zip files.")
    ap.add_argument("--out-dir", default=".", help="Where to write pack zips.")
    ap.add_argument("--prefix", default="CH_Jan_pack", help="Output pack prefix (adds _01, _02...).")
    ap.add_argument("--max-per-pack", type=int, default=5, help="Maximum number of skin zips per pack.")
    ap.add_argument("--max-mb", type=float, default=100.0, help="Maximum pack size (MB).")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; do not write zips.")
    args = ap.parse_args(argv)

    src_dir = Path(args.src_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_per = max(1, int(args.max_per_pack))
    max_bytes = int(float(args.max_mb) * 1024 * 1024)

    skins = _iter_skin_zips(src_dir)
    if not skins:
        print(f"ERROR: no .zip files found in {src_dir}")
        return 2

    # ZIP overhead estimate: local header (~30 + name) + central dir (~46 + name) per entry + EOCD.
    # We keep a small safety margin so we don't accidentally exceed 100MB by a few KB.
    def est_overhead(name_len: int) -> int:
        return (30 + name_len) + (46 + name_len) + 8

    packs: list[list[Entry]] = []
    cur: list[Entry] = []
    cur_bytes = 0

    for e in skins:
        name_len = len(e.path.name.encode("utf-8", errors="ignore"))
        add = e.size + est_overhead(name_len)
        # If a single file exceeds max, we still have to pack it alone (and it will exceed).
        if e.size > max_bytes:
            if cur:
                packs.append(cur)
                cur = []
                cur_bytes = 0
            packs.append([e])
            continue

        if cur and (len(cur) >= max_per or (cur_bytes + add) > max_bytes):
            packs.append(cur)
            cur = []
            cur_bytes = 0

        cur.append(e)
        cur_bytes += add

    if cur:
        packs.append(cur)

    # Print plan
    total = 0
    for i, pk in enumerate(packs, start=1):
        sz = sum(x.size for x in pk)
        total += len(pk)
        out_name = f"{args.prefix}_{i:02d}.zip"
        print(f"{out_name}: {len(pk)} files, ~{_mb(sz):.2f} MB")
        for x in pk:
            print(f"  - {x.path.name} ({_mb(x.size):.2f} MB)")

    if total != len(skins):
        print(f"WARNING: planned {total} files, expected {len(skins)}")

    # Write packs
    if args.dry_run:
        return 0

    for i, pk in enumerate(packs, start=1):
        out_path = out_dir / f"{args.prefix}_{i:02d}.zip"
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
            for e in pk:
                z.write(e.path, arcname=e.path.name)
        out_size = out_path.stat().st_size
        if out_size > max_bytes:
            print(f"WARNING: {out_path.name} is {_mb(out_size):.2f} MB (exceeds {args.max_mb:.2f} MB cap)")
        else:
            print(f"Wrote {out_path.name} ({_mb(out_size):.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

