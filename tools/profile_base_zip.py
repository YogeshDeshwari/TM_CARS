#!/usr/bin/env python3
"""
Profile a StadiumCar base zip for consistent reskinning defaults.

This produces a JSON profile capturing:
- texture names + DDS sizes + compression (FourCC) + mipmaps
- whether the pack looks like a standard Stadium mod pack (MainBody*.Gbx present)
- whether Diffuse alpha is constant (common in TMNF packs)

The profile can later be used by generators/validators to avoid pack-specific surprises.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

import sys

# Allow running from anywhere while importing repo-local modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmnf_dds import (  # type: ignore  # noqa: E402
    read_dds_dimensions_from_bytes,
    read_dds_fourcc_from_bytes,
    read_dds_mipmap_count_from_bytes,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_dds_info(z: zipfile.ZipFile, name: str) -> Dict[str, Any]:
    with z.open(name, "r") as f:
        hdr = f.read(128)
    w, h = read_dds_dimensions_from_bytes(hdr)
    fourcc = read_dds_fourcc_from_bytes(hdr) or "RGBA8"
    mip_count = int(read_dds_mipmap_count_from_bytes(hdr))
    return {
        "width": int(w),
        "height": int(h),
        "fourcc": str(fourcc),
        "mipmaps": bool(mip_count > 1),
        "mip_count": int(mip_count),
    }


def _safe_rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p.resolve())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create a JSON profile for a TMNF/TMUF Stadium base zip.")
    ap.add_argument("zip_path", help="Path to a base skin zip (e.g. CH_all_skins/CH_2026.zip).")
    ap.add_argument("--out-dir", default="profiles", help="Output directory for profiles (default: profiles/).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing profile file if present.")
    ap.add_argument("--pretty", action="store_true", help="Write pretty-printed JSON.")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = (root / zip_path).resolve()
    if not zip_path.exists():
        raise SystemExit(f"ERROR: zip not found: {zip_path}")

    zip_sha = _sha256_file(zip_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{zip_sha}.json"
    if out_path.exists() and (not bool(args.force)):
        raise SystemExit(f"ERROR: profile exists: {out_path} (use --force to overwrite)")

    textures: Dict[str, Any] = {}
    base_diffuse_alpha: Optional[Dict[str, Any]] = None
    is_standard_stadium = False

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        is_standard_stadium = ("MainBody.Solid.Gbx" in names) and ("MainBodyHigh.Solid.Gbx" in names)

        for name in sorted(n for n in names if n.lower().endswith(".dds")):
            try:
                textures[name] = _read_dds_info(z, name)
            except Exception as e:
                textures[name] = {"error": str(e)}

        if "Diffuse.dds" in names:
            try:
                raw = z.read("Diffuse.dds")
                img = Image.open(io.BytesIO(raw)).convert("RGBA")
                mn, mx = img.getchannel("A").getextrema()
                base_diffuse_alpha = {"min": int(mn), "max": int(mx), "constant": bool(mn == mx)}
            except Exception as e:
                base_diffuse_alpha = {"error": str(e)}

    profile: Dict[str, Any] = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zip_path": _safe_rel(root, zip_path),
        "zip_sha256": zip_sha,
        "is_standard_stadium": bool(is_standard_stadium),
        "textures": textures,
        "base_diffuse_alpha": base_diffuse_alpha,
        "recommended": {
            # Conservative defaults; override manually if a base pack is known to differ.
            "finish_alpha": "auto",
            "finish_neutral": 142,  # 0x8E
            "finish_invert": False,
            "sanitize": True,
        },
    }

    out_path.write_text(json.dumps(profile, indent=2 if args.pretty else None, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

