#!/usr/bin/env python3
"""
Create ONE zip pack containing all per-skin zips in CH_all_skins/.

This avoids macOS metadata (__MACOSX/.DS_Store) by packing only the *.zip files.
Note: a single pack will typically exceed Discord's upload limit.

Usage:
  python3 tools/pack_ch_all_skins_single.py --out CH_Jan_Skins.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="CH_all_skins", help="Directory containing per-skin .zip files.")
    ap.add_argument("--out", default="CH_Jan_Skins.zip", help="Output zip filename.")
    ap.add_argument(
        "--exclude-prefix",
        action="append",
        default=["CH_Jan_pack_", "CH_Jan_Skins"],
        help="Exclude input zips whose filename starts with this prefix (repeatable).",
    )
    args = ap.parse_args(argv)

    src_dir = (ROOT / str(args.src)).resolve()
    out_path = (ROOT / str(args.out)).resolve()

    items = []
    for p in sorted(src_dir.glob("*.zip")):
        if any(p.name.startswith(pref) for pref in (args.exclude_prefix or [])):
            continue
        if not p.is_file():
            continue
        items.append(p)

    if not items:
        print(f"ERROR: no input zips found in {src_dir}")
        return 2

    if out_path.exists():
        out_path.unlink()

    # Store-only (fast, no double compression).
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
        for p in items:
            z.write(p, arcname=p.name)

    sz = out_path.stat().st_size
    print(f"Wrote {out_path.name}: files={len(items)}, size={_mb(sz):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

