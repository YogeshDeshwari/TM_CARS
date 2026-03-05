#!/usr/bin/env python3
"""
Remove macOS metadata entries from a zip:
  - __MACOSX/...
  - .DS_Store

This is useful when a Finder-created archive contains only metadata or bloats/annoys importers.

Usage:
  python3 tools/clean_zip_macos.py input.zip --out output_clean.zip
  python3 tools/clean_zip_macos.py input.zip --in-place
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _is_junk(name: str) -> bool:
    n = name.replace("\\", "/")
    if n.startswith("__MACOSX/") or n == "__MACOSX":
        return True
    if n.endswith("/.DS_Store") or n == ".DS_Store":
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="Input zip file.")
    ap.add_argument("--out", default=None, help="Output zip path (default: <input>_clean.zip).")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input zip (writes a temp file then replaces).")
    args = ap.parse_args(argv)

    in_path = Path(args.zip).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: not found: {in_path}")
        return 2

    if bool(args.in_place):
        out_path = in_path.with_suffix(in_path.suffix + ".tmp_clean")
    else:
        out_path = Path(args.out).expanduser().resolve() if args.out else in_path.with_name(in_path.stem + "_clean.zip")

    kept = 0
    removed = 0

    with zipfile.ZipFile(in_path, "r") as zin, zipfile.ZipFile(out_path, "w") as zout:
        for info in zin.infolist():
            name = info.filename
            if _is_junk(name):
                removed += 1
                continue
            # Preserve metadata where possible.
            zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
            zi.external_attr = info.external_attr
            zi.compress_type = info.compress_type
            zi.comment = info.comment
            zi.extra = info.extra
            # Stream copy.
            if hasattr(info, "is_dir") and info.is_dir():
                zout.writestr(zi, b"")
            else:
                with zin.open(info, "r") as src, zout.open(zi, "w") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            kept += 1

    if bool(args.in_place):
        bak = in_path.with_suffix(in_path.suffix + ".bak")
        try:
            if bak.exists():
                bak.unlink()
        except Exception:
            pass
        in_path.rename(bak)
        out_path.rename(in_path)
        print(f"Cleaned in-place: {in_path.name} (backup: {bak.name}) kept={kept} removed={removed}")
    else:
        print(f"Wrote: {out_path} kept={kept} removed={removed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

