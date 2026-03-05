#!/usr/bin/env python3
"""
Export UV diagnostics for the standard Stadium template:
  - island atlas (reuses export_uv_atlas logic)
  - tiny-island warnings (likely mipmap/detail loss)
  - overlap/stack groups (identical island masks on the small grid)
  - mirror groups (identical after horizontal flip on the small grid)

Outputs:
  out/uv_atlas/diagnostics_<size>.json
  out/uv_atlas/diagnostics_<size>.png

Usage:
  python3 tools/export_uv_diagnostics.py --base-zip CH_all_skins/CH_2026.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_tmnf_skin as gen  # type: ignore


def _try_font(px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", px)
    except Exception:
        return ImageFont.load_default()


def _hash_mask_bytes(mask_small: "gen.np.ndarray") -> str:  # type: ignore[name-defined]
    # Downsample to 32x32 to make signature stable and small.
    m = mask_small.astype("uint8") * 255
    im = Image.fromarray(m, mode="L").resize((32, 32), Image.Resampling.NEAREST)
    b = im.tobytes()
    return hashlib.sha1(b).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-zip", default=None, help="Optional base zip; used to match Diffuse size.")
    ap.add_argument("--size", type=int, default=2048, help="Fallback size if --base-zip not provided.")
    ap.add_argument("--out-dir", default="out/uv_atlas", help="Output directory.")
    ap.add_argument("--max-id", type=int, default=60, help="Max ranked island id to include.")
    ap.add_argument("--tiny-frac", type=float, default=0.0020, help="Islands smaller than this fraction of used area are 'tiny'.")
    args = ap.parse_args(argv)

    out_dir = (ROOT / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    size = int(args.size)
    if args.base_zip:
        import zipfile

        bz = (ROOT / str(args.base_zip)).resolve()
        with zipfile.ZipFile(bz, "r") as z:
            hdr = z.open("Diffuse.dds").read(128)
        w, _h = gen._read_dds_dimensions_from_bytes(hdr)  # type: ignore[attr-defined]
        size = int(w)

    if gen.np is None:  # type: ignore[attr-defined]
        print("ERROR: numpy is required for UV diagnostics.", file=sys.stderr)
        return 2

    tmpl = gen._load_standard_stadium_uv_template_diffuse(size=(size, size))  # type: ignore[attr-defined]
    if tmpl is None:
        print("ERROR: Could not load standard stadium UV template from examples/.", file=sys.stderr)
        return 2

    ranked_small, comps_full, _scale = gen._compute_ranked_uv_islands(tmpl, downscale_to=512)  # type: ignore[attr-defined]

    # Compute used area on small grid.
    used_small = (ranked_small > 0)
    used_area_small = int(used_small.sum())

    # Build per-island small masks and signatures.
    max_id = min(int(args.max_id), int(ranked_small.max()))
    sig_to_ids: dict[str, list[int]] = defaultdict(list)
    sig_flip_to_ids: dict[str, list[int]] = defaultdict(list)
    island_meta: list[dict[str, object]] = []

    for rid in range(1, max_id + 1):
        m = (ranked_small == rid)
        a_small = int(m.sum())
        frac = float(a_small) / float(max(1, used_area_small))
        sig = _hash_mask_bytes(m)  # type: ignore[arg-type]
        sig_to_ids[sig].append(rid)
        sig_flip = _hash_mask_bytes(gen.np.fliplr(m))  # type: ignore[attr-defined]
        sig_flip_to_ids[sig_flip].append(rid)

        try:
            x0, y0, x1, y1, area_full = comps_full[rid - 1]
        except Exception:
            x0 = y0 = x1 = y1 = 0
            area_full = 0
        island_meta.append(
            {
                "id": rid,
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "area_full": int(area_full),
                "area_small": int(a_small),
                "frac_used_small": frac,
                "is_tiny": bool(frac < float(args.tiny_frac)),
                "sig": sig,
                "sig_flipx": sig_flip,
            }
        )

    # Overlap/stack groups: identical signatures with >1 island.
    stacked = [ids for ids in sig_to_ids.values() if len(ids) > 1]
    stacked.sort(key=len, reverse=True)

    # Mirror groups: pairs where one island's sig matches another's flip signature.
    # Build id->sig, id->flip
    id_to_sig = {int(m["id"]): str(m["sig"]) for m in island_meta}
    id_to_flip = {int(m["id"]): str(m["sig_flipx"]) for m in island_meta}
    seen_pairs: set[tuple[int, int]] = set()
    mirror_pairs: list[tuple[int, int]] = []
    sig_to_any_ids = defaultdict(list)
    for rid, sig in id_to_sig.items():
        sig_to_any_ids[sig].append(rid)
    for a in range(1, max_id + 1):
        flip = id_to_flip.get(a)
        if not flip:
            continue
        for b in sig_to_any_ids.get(flip, []):
            if a == b:
                continue
            p = (min(a, b), max(a, b))
            if p in seen_pairs:
                continue
            seen_pairs.add(p)
            mirror_pairs.append(p)

    # Render overlay on top of existing atlas PNG style.
    # Base: colored island map on dark bg.
    m_small_img = Image.fromarray((ranked_small.astype("uint8")))  # type: ignore[attr-defined]
    m_full = m_small_img.resize((size, size), Image.Resampling.NEAREST).convert("L")
    pal = []
    for i in range(256):
        pal.extend([(i * 97) % 256, (i * 57) % 256, (i * 23) % 256])
    mp = m_full.convert("P")
    mp.putpalette(pal)
    colored = mp.convert("RGB")
    bg = Image.new("RGB", (size, size), (8, 8, 12))
    used_mask = m_full.point(lambda p: 255 if p > 0 else 0)
    out = Image.composite(colored, bg, used_mask)

    d = ImageDraw.Draw(out)
    font = _try_font(max(12, size // 90))

    # Draw bbox for tiny islands in red.
    for meta in island_meta:
        if not bool(meta["is_tiny"]):
            continue
        x0, y0, x1, y1 = meta["bbox"]  # type: ignore[assignment]
        d.rectangle((x0, y0, x1, y1), outline=(255, 80, 80), width=max(1, size // 900))
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        d.text((cx, cy), f"{meta['id']}", fill=(255, 120, 120), anchor="mm", font=font)

    # Annotate stacked groups (identical islands) with a cyan frame.
    for g in stacked[:10]:
        for rid in g:
            try:
                x0, y0, x1, y1, _area = comps_full[rid - 1]
            except Exception:
                continue
            d.rectangle((x0, y0, x1, y1), outline=(90, 255, 255), width=max(1, size // 1100))

    png_path = out_dir / f"diagnostics_{size}.png"
    json_path = out_dir / f"diagnostics_{size}.json"
    out.save(png_path)

    summary = {
        "size": size,
        "used_area_small": used_area_small,
        "tiny_frac_threshold": float(args.tiny_frac),
        "islands": island_meta,
        "stacked_groups": stacked,
        "mirror_pairs": mirror_pairs[:2000],
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(png_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

