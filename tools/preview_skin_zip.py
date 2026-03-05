#!/usr/bin/env python3
"""
Create quick PNG previews from a TMNF/TMUF StadiumCar skin zip.

Outputs:
- a contact sheet (Diffuse/Details/Dirty/Illum/ProjShad/Icon + wing crop when present)

This is not a 3D render; it's a fast way to compare variants and spot obvious issues.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _safe_rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p.resolve())


def _stadium_rear_wing_rect(w: int, h: int) -> Tuple[int, int, int, int]:
    # Standard Stadium model reference @ 2048: (224,4)->(780,208)
    x0 = int(round(224 * (w / 2048.0)))
    y0 = int(round(4 * (h / 2048.0)))
    x1 = int(round(780 * (w / 2048.0)))
    y1 = int(round(208 * (h / 2048.0)))
    return (x0, y0, x1, y1)


def _open_dds(z: zipfile.ZipFile, name: str) -> Image.Image:
    raw = z.read(name)
    return Image.open(io.BytesIO(raw))


def _thumb(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    out = img.copy()
    out = out.convert("RGBA")
    out.thumbnail(size, Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", size, (20, 20, 22, 255))
    bg.alpha_composite(out, ((size[0] - out.size[0]) // 2, (size[1] - out.size[1]) // 2))
    return bg


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 6
    # background strip
    d.rectangle((0, 0, out.size[0], 18), fill=(0, 0, 0, 160))
    d.text((pad, 2), text, fill=(255, 255, 255, 235), font=font)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create quick previews from a skin zip.")
    ap.add_argument("zip_path", help="Skin zip path.")
    ap.add_argument("--out-dir", default="out/previews", help="Output directory (default: out/previews).")
    ap.add_argument("--tile", type=int, default=360, help="Tile size for contact sheet (default: 360).")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = (root / zip_path).resolve()
    if not zip_path.exists():
        raise SystemExit(f"ERROR: zip not found: {zip_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tile = max(180, int(args.tile))
    tile_size = (tile, tile)

    # Load what we can
    wanted = [
        "Diffuse.dds",
        "Details.dds",
        "DiffuseDirty.dds",
        "DetailsDirty.dds",
        "Illum.dds",
        "ProjShad.dds",
        "Icon.dds",
    ]

    thumbs: Dict[str, Image.Image] = {}
    wing_thumb: Optional[Image.Image] = None

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())
        for n in wanted:
            if n not in names:
                continue
            try:
                img = _open_dds(z, n)
                thumbs[n] = _label(_thumb(img, tile_size), n)
            except Exception:
                continue

        # Wing crop from Diffuse
        if "Diffuse.dds" in names:
            try:
                diff = _open_dds(z, "Diffuse.dds").convert("RGBA")
                x0, y0, x1, y1 = _stadium_rear_wing_rect(*diff.size)
                wing = diff.crop((x0, y0, x1, y1))
                wing_size = (tile * 2, tile // 2)
                wing_thumb = _label(_thumb(wing, wing_size), "Wing plate (Diffuse crop)")
            except Exception:
                wing_thumb = None

    # Layout: 3 columns, 3 rows + optional wing row.
    cols = 3
    rows = 3
    margin = 12

    canvas_w = cols * tile + (cols + 1) * margin
    canvas_h = rows * tile + (rows + 1) * margin
    if wing_thumb is not None:
        canvas_h += (wing_thumb.size[1] + margin)

    sheet = Image.new("RGBA", (canvas_w, canvas_h), (14, 14, 16, 255))

    order = [
        "Diffuse.dds",
        "Details.dds",
        "Icon.dds",
        "ProjShad.dds",
        "Illum.dds",
        "DiffuseDirty.dds",
        "DetailsDirty.dds",
    ]

    i = 0
    for key in order:
        if key not in thumbs:
            continue
        r = i // cols
        c = i % cols
        x = margin + c * (tile + margin)
        y = margin + r * (tile + margin)
        sheet.alpha_composite(thumbs[key], (x, y))
        i += 1
        if i >= cols * rows:
            break

    if wing_thumb is not None:
        y = margin + rows * (tile + margin)
        sheet.alpha_composite(wing_thumb, (margin, y))

    out_path = out_dir / f"{zip_path.stem}_sheet.png"
    sheet.convert("RGB").save(out_path)
    print(_safe_rel(root, out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

