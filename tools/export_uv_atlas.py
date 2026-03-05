#!/usr/bin/env python3
"""
Export a Stadium UV-island atlas (PNG + JSON) for spatial awareness.

This uses the existing "standard Stadium UV template" technique from generate_tmnf_skin.py
to get stable island ids (ranked ids like 2=nose, 5/6=sidepods for the standard model).

Usage:
  python3 tools/export_uv_atlas.py --size 2048 --out-dir out/uv_atlas
  python3 tools/export_uv_atlas.py --base-zip CH_all_skins/CH_2026.zip --out-dir out/uv_atlas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_tmnf_skin as gen  # type: ignore


def _try_font(px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", px)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", px)
        except Exception:
            return ImageFont.load_default()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-zip", default=None, help="Optional base zip; used to match Diffuse size.")
    ap.add_argument("--size", type=int, default=2048, help="Fallback size if --base-zip not provided.")
    ap.add_argument("--out-dir", default="out/uv_atlas", help="Output directory.")
    ap.add_argument("--max-id", type=int, default=40, help="Max ranked island id to include.")
    args = ap.parse_args(argv)

    out_dir = (ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    size = int(args.size)
    if args.base_zip:
        # Use generator helper to read Diffuse.dds dimensions from base zip.
        import zipfile

        bz = (ROOT / str(args.base_zip)).resolve()
        with zipfile.ZipFile(bz, "r") as z:
            hdr = z.open("Diffuse.dds").read(128)
        w, h = gen._read_dds_dimensions_from_bytes(hdr)  # type: ignore[attr-defined]
        size = int(w)

    # Load the standard template diffuse (near-black background) and compute islands.
    tmpl = gen._load_standard_stadium_uv_template_diffuse(size=(size, size))  # type: ignore[attr-defined]
    if tmpl is None:
        print("ERROR: Could not load standard stadium UV template from examples/.", file=sys.stderr)
        return 2

    if gen.np is None:  # type: ignore[attr-defined]
        print("ERROR: numpy is required for UV island atlas export.", file=sys.stderr)
        return 2

    ranked_map_small, comps_ranked_full, _scale = gen._compute_ranked_uv_islands(tmpl, downscale_to=512)  # type: ignore[attr-defined]

    # Upscale ranked ids to full size for visualization.
    m_small = Image.fromarray((ranked_map_small.astype("uint8")))  # type: ignore[attr-defined]
    m_full = m_small.resize((size, size), Image.Resampling.NEAREST).convert("L")

    # Colorize map: simple palette cycling.
    pal = []
    for i in range(256):
        # deterministic pseudo-colors
        r = (i * 97) % 256
        g = (i * 57) % 256
        b = (i * 23) % 256
        pal.extend([r, g, b])
    m_full_p = m_full.convert("P")
    m_full_p.putpalette(pal)
    colored = m_full_p.convert("RGB")

    # Make background (id=0) dark for readability.
    bg = Image.new("RGB", (size, size), (8, 8, 12))
    mask_used = m_full.point(lambda p: 255 if p > 0 else 0)
    out = Image.composite(colored, bg, mask_used)

    # Draw bounding boxes + labels from comps_ranked_full.
    draw = ImageDraw.Draw(out)
    font = _try_font(max(12, size // 90))

    islands = []
    for rid, comp in enumerate(comps_ranked_full, start=1):
        if rid > int(args.max_id):
            break
        # comp is (x0,y0,x1,y1,area) in full-res coords.
        try:
            x0, y0, x1, y1, area = comp
        except Exception:
            continue
        # bbox returned in full-res coords already (from generator helper).
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=max(1, size // 800))
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        draw.text((cx, cy), str(rid), fill=(255, 255, 255), anchor="mm", font=font)
        islands.append({"id": rid, "bbox": [int(x0), int(y0), int(x1), int(y1)], "area": int(area), "center": [cx, cy]})

    png_path = out_dir / f"standard_stadium_islands_{size}.png"
    json_path = out_dir / f"standard_stadium_islands_{size}.json"
    out.save(png_path)
    json_path.write_text(json.dumps({"size": size, "islands": islands}, indent=2) + "\n", encoding="utf-8")

    print(png_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

