#!/usr/bin/env python3
"""
Validate a TMNF/TMUF StadiumCar skin zip for packaging + DDS sanity.

This is meant to catch the common causes of "looks wrong in game":
- missing Dirty / Illum / ProjShad files causing fallback to defaults
- wrong DDS dimensions vs the donor pack
- missing mipmaps (shimmering / ugly transitions)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


CORE_TEXTURES = (
    "Diffuse.dds",
    "Details.dds",
    "Icon.dds",
    "ProjShad.dds",
    "DiffuseDirty.dds",
    "DetailsDirty.dds",
    "Illum.dds",
)


def _iter_zip_paths(root: Path, paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p0 in paths:
        p = Path(p0)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            out.extend(sorted(p.glob("*.zip")))
        else:
            out.append(p)
    # de-dupe preserving order
    seen: set[Path] = set()
    uniq: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def _read_dds_info(z: zipfile.ZipFile, name: str) -> Dict[str, Any]:
    with z.open(name, "r") as f:
        hdr = f.read(128)
    w, h = read_dds_dimensions_from_bytes(hdr)
    fourcc = read_dds_fourcc_from_bytes(hdr) or "RGBA8"
    mip_count = int(read_dds_mipmap_count_from_bytes(hdr))
    return {"width": int(w), "height": int(h), "fourcc": str(fourcc), "mip_count": int(mip_count)}


def _load_profile(root: Path, profile_path: str) -> Dict[str, Any]:
    p = Path(profile_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _build_reference_from_base_zip(base_zip: Path) -> Tuple[set[str], Dict[str, Dict[str, Any]]]:
    refs: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(base_zip, "r") as z:
        names = set(z.namelist())
        for t in CORE_TEXTURES:
            if t in names:
                try:
                    refs[t] = _read_dds_info(z, t)
                except Exception:
                    continue
    return names, refs


def _fmt_dims(info: Dict[str, Any]) -> str:
    return f'{info.get("width")}x{info.get("height")} {info.get("fourcc")} mips:{info.get("mip_count")}'


def validate_one(
    zip_path: Path,
    *,
    base_names: Optional[set[str]],
    base_refs: Optional[Dict[str, Dict[str, Any]]],
    require_aux: bool,
    require_gbx: bool,
    strict: bool,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not zip_path.exists():
        return ([f"missing zip: {zip_path}"], [])

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())

        # Required models (for mod-pack zips)
        if require_gbx:
            for gbx in ("MainBody.Solid.Gbx", "MainBodyHigh.Solid.Gbx"):
                if gbx not in names:
                    errors.append(f"missing required model file: {gbx}")

        # Decide required textures
        required_textures: List[str] = ["Diffuse.dds"]
        if base_names is not None:
            # Mirror the donor pack expectations.
            for t in CORE_TEXTURES:
                if t in base_names and t not in required_textures:
                    required_textures.append(t)
        else:
            # Conservative default expectations.
            for t in ("Details.dds", "Icon.dds", "ProjShad.dds"):
                if t not in required_textures:
                    required_textures.append(t)
            if require_aux:
                for t in ("DiffuseDirty.dds", "DetailsDirty.dds", "Illum.dds"):
                    if t not in required_textures:
                        required_textures.append(t)

        for t in required_textures:
            if t not in names:
                errors.append(f"missing required texture: {t}")

        # DDS sanity + size comparisons
        for t in [n for n in names if n.lower().endswith(".dds")]:
            try:
                info = _read_dds_info(z, t)
            except Exception as e:
                errors.append(f"invalid DDS header for {t}: {e}")
                continue

            # Compare to base if available
            if base_refs and t in base_refs:
                ref = base_refs[t]
                if (info["width"], info["height"]) != (ref["width"], ref["height"]):
                    errors.append(
                        f"{t} size mismatch: got {info['width']}x{info['height']} expected {ref['width']}x{ref['height']}"
                    )

            # Mipmaps
            if info["mip_count"] <= 1:
                msg = f"{t} has no mipmaps"
                (errors if strict else warnings).append(msg)

            # Format recommendations (warnings)
            fourcc = str(info["fourcc"]).upper()
            if t == "Diffuse.dds" and fourcc not in ("DXT5",):
                warnings.append(f"Diffuse.dds is {fourcc} (TMNF packs usually use DXT5)")
            if t == "Illum.dds" and fourcc not in ("DXT1",):
                warnings.append(f"Illum.dds is {fourcc} (DXT1 is preferred in TMNF/TMUF)")
            if t == "ProjShad.dds" and fourcc not in ("DXT1",):
                warnings.append(f"ProjShad.dds is {fourcc} (DXT1 is preferred)")
            if t in ("DiffuseDirty.dds", "DetailsDirty.dds"):
                if fourcc == "DXT1":
                    warnings.append(f"{t} is DXT1 (no alpha); Dirty maps normally need alpha (DXT3/DXT5)")
                if fourcc == "RGBA8":
                    warnings.append(f"{t} is RGBA8 (uncompressed); may be heavier than needed")

    return errors, warnings


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate TMNF/TMUF StadiumCar skin zips.")
    ap.add_argument("paths", nargs="+", help="Zip files or directories (directories: validates *.zip inside).")
    ap.add_argument("--base-zip", default=None, help="Optional donor/base zip to compare texture sizes against.")
    ap.add_argument("--profile", default=None, help="Optional JSON profile to compare against (from tools/profile_base_zip.py).")
    ap.add_argument("--require-aux", action=argparse.BooleanOptionalAction, default=True, help="Require Dirty + Illum textures.")
    ap.add_argument("--require-gbx", action=argparse.BooleanOptionalAction, default=True, help="Require MainBody*.Gbx model files.")
    ap.add_argument("--strict", action="store_true", help="Treat missing mipmaps as errors (not warnings).")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]

    base_names: Optional[set[str]] = None
    base_refs: Optional[Dict[str, Dict[str, Any]]] = None
    if args.profile:
        prof = _load_profile(root, str(args.profile))
        tex = prof.get("textures", {}) if isinstance(prof, dict) else {}
        base_refs = {k: v for k, v in tex.items() if isinstance(v, dict) and ("width" in v and "height" in v)}
        base_names = set(base_refs.keys())
    elif args.base_zip:
        bz = Path(args.base_zip)
        if not bz.is_absolute():
            bz = (root / bz).resolve()
        base_names, base_refs = _build_reference_from_base_zip(bz)

    zip_paths = _iter_zip_paths(root, args.paths)
    if not zip_paths:
        print("No zip files found.")
        return 2

    any_errors = False
    for zp in zip_paths:
        errs, warns = validate_one(
            zp,
            base_names=base_names,
            base_refs=base_refs,
            require_aux=bool(args.require_aux),
            require_gbx=bool(args.require_gbx),
            strict=bool(args.strict),
        )
        if errs:
            any_errors = True
        print(f"\n== {zp.name} ==")
        if errs:
            print("ERRORS:")
            for e in errs:
                print(f" - {e}")
        if warns:
            print("WARNINGS:")
            for w in warns:
                print(f" - {w}")
        if (not errs) and (not warns):
            print("OK")

    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

