#!/usr/bin/env python3
"""
Pack CH_all_skins/*.zip into multiple Discord-friendly pack zips.

Goals:
  - Each output pack zip < 100 MB (configurable)
  - Fit as many skin zips as possible per pack (no fixed count)
  - Avoid macOS metadata folders like __MACOSX (by not zipping directories)
  - Do NOT include previously-made pack zips (e.g. CH_Jan_Skins.zip, CH_Jan_pack_*.zip)

Usage:
  python3 tools/pack_ch_all_skins_jan.py

Outputs (by default):
  CH_Jan_pack_01.zip, CH_Jan_pack_02.zip, ...
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Item:
    path: Path
    size: int


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def _gather(src_dir: Path, *, exclude_prefixes: tuple[str, ...]) -> list[Item]:
    out: list[Item] = []
    for p in sorted(src_dir.glob("*.zip")):
        if any(p.name.startswith(pref) for pref in exclude_prefixes):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if not p.is_file():
            continue
        out.append(Item(path=p, size=int(st.st_size)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="CH_all_skins", help="Source directory containing skin zip files.")
    ap.add_argument("--out-dir", default=".", help="Where to write pack zips.")
    ap.add_argument("--prefix", default="CH_Jan_pack", help="Pack file prefix (adds _01, _02...).")
    ap.add_argument("--cap-mb", type=float, default=100.0, help="Max size per pack zip (MB).")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; do not write pack zips.")
    args = ap.parse_args(argv)

    src_dir = (ROOT / str(args.src)).resolve()
    out_dir = (ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_bytes = int(float(args.cap_mb) * 1024 * 1024)
    # Exclude old packs by name prefix; adjust here if you use different naming.
    exclude_prefixes = ("CH_Jan_pack_", "CH_Jan_Skins", "__MACOSX")

    items = _gather(src_dir, exclude_prefixes=exclude_prefixes)
    if not items:
        print(f"ERROR: no skin zips found in {src_dir}")
        return 2

    # First-fit decreasing bin packing to minimize number of packs while staying under cap.
    items.sort(key=lambda x: x.size, reverse=True)

    OVERHEAD = 2048  # safety bytes per entry (zip headers + slack)
    bins_used: list[int] = []
    packs: list[list[Item]] = []

    for it in items:
        need = it.size + OVERHEAD
        if it.size > cap_bytes:
            # Cannot fit under cap; still pack alone and warn.
            packs.append([it])
            bins_used.append(need)
            continue

        placed = False
        for i in range(len(packs)):
            if bins_used[i] + need <= cap_bytes:
                packs[i].append(it)
                bins_used[i] += need
                placed = True
                break
        if not placed:
            packs.append([it])
            bins_used.append(need)

    # Print plan
    for i, pk in enumerate(packs, start=1):
        raw = sum(x.size for x in pk)
        print(f"{args.prefix}_{i:02d}.zip: files={len(pk)}, raw={_mb(raw):.2f} MB")

    if args.dry_run:
        return 0

    # Write packs. Use ZIP_STORED to avoid recompressing .zip files.
    written: list[Path] = []
    for i, pk in enumerate(packs, start=1):
        out_path = out_dir / f"{args.prefix}_{i:02d}.zip"
        if out_path.exists():
            out_path.unlink()
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
            for it in pk:
                # arcname is just the filename (no directories => no __MACOSX folder creation).
                z.write(it.path, arcname=it.path.name)
        sz = out_path.stat().st_size
        if sz > cap_bytes:
            print(f"WARNING: {out_path.name} is {_mb(sz):.2f} MB (cap {args.cap_mb:.2f} MB)")
        else:
            print(f"Wrote {out_path.name} ({_mb(sz):.2f} MB)")
        written.append(out_path)

    print(f"Done. Packs written: {len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

