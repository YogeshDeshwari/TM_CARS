#!/usr/bin/env python3
"""
Cleanup script for this repo (removes re-creatable output artefacts).

By default, this is conservative and only targets obvious generated folders/files:
- out/ (generated skins, previews, debug images)
- base_dds/*.png (converted previews)
- assets/generated_logos/ (procedural logo demo outputs)

Usage:
  python3 tools/cleanup_repo.py --dry-run
  python3 tools/cleanup_repo.py
  python3 tools/cleanup_repo.py --keep-zips
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _rm_tree(p: Path, *, dry_run: bool) -> None:
    if not p.exists():
        return
    if dry_run:
        print(f"[dry-run] rm -rf {p}")
        return
    shutil.rmtree(p, ignore_errors=True)
    print(f"removed: {p}")


def _rm_file(p: Path, *, dry_run: bool) -> None:
    if not p.exists():
        return
    if dry_run:
        print(f"[dry-run] rm {p}")
        return
    try:
        p.unlink()
        print(f"removed: {p}")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print what would be removed, but don't delete anything.")
    ap.add_argument("--keep-zips", action="store_true", help="If set, keep any *.zip files under out/.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    out_dir = root / "out"
    if out_dir.exists():
        if args.keep_zips:
            # remove everything except *.zip
            for child in out_dir.iterdir():
                if child.is_file() and child.suffix.lower() == ".zip":
                    continue
                if child.is_dir():
                    _rm_tree(child, dry_run=args.dry_run)
                else:
                    _rm_file(child, dry_run=args.dry_run)
        else:
            _rm_tree(out_dir, dry_run=args.dry_run)

    # Converted DDS previews
    for p in (root / "base_dds").glob("*.png"):
        _rm_file(p, dry_run=args.dry_run)

    # Generated logos
    _rm_tree(root / "assets" / "generated_logos", dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

