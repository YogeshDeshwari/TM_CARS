#!/usr/bin/env python3
"""
Generate a TMNF-friendly car skin texture as a DDS file.

This script writes:
- DDS format: uncompressed RGBA8 (A8R8G8B8 masks) OR DXT5 (BC3)
- Optional mipmaps (recommended)

Why RGBA8 vs DXT5?
- **DXT5** is what most TMNF Stadium car mod packs use (small files, fast in-game).
- **RGBA8** is simpler to generate (no block compression), but produces larger files.
"""

from __future__ import annotations

import argparse
import io
import json
import hashlib
import math
import random
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# Spatial awareness module (optional, for --spatial-aware mode)
try:
    from car_geometry import CarGeometry, ColorRole, FinishType
    HAS_CAR_GEOMETRY = True
except ImportError:
    HAS_CAR_GEOMETRY = False
    CarGeometry = None  # type: ignore

# Optional: vibrant palette sampler (used for --palette auto_vibrant).
# This keeps generate_tmnf_skin.py usable as a standalone script.
try:
    from palette_lab import sample_vibrant_palette  # type: ignore
except Exception:  # pragma: no cover
    sample_vibrant_palette = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

# Novel 2026 pattern generators (suminagashi, moire, palimpsest)
try:
    from skin_utils import generate_suminagashi, generate_moire_interference, generate_palimpsest
except Exception:  # pragma: no cover
    generate_suminagashi = None  # type: ignore[assignment]
    generate_moire_interference = None  # type: ignore[assignment]
    generate_palimpsest = None  # type: ignore[assignment]


def _deband_dither_rgba(
    img: Image.Image,
    *,
    seed: int,
    amp: int = 2,
    lum_lo: float = 10.0,
    lum_hi: float = 140.0,
    grad_hi: float = 10.0,
) -> Image.Image:
    """
    Reduce visible gradient banding by adding subtle per-channel noise in low-gradient regions.

    - Only touches RGB, keeps alpha identical.
    - Avoids edges/text by masking out high-gradient pixels.
    """
    if np is None:
        return img
    try:
        arr = np.array(img.convert("RGBA"), dtype=np.int16)
        rgb = arr[..., :3]
        a = arr[..., 3]
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        dy = np.abs(lum - np.roll(lum, 1, axis=0))
        dx = np.abs(lum - np.roll(lum, 1, axis=1))
        grad = dx + dy
        m = (a > 10) & (lum >= float(lum_lo)) & (lum <= float(lum_hi)) & (grad <= float(grad_hi))
        if not bool(m.any()):
            return img
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        noise = rng.integers(-int(amp), int(amp) + 1, size=rgb.shape, dtype=np.int16)
        rgb[m] = np.clip(rgb[m] + noise[m], 0, 255)
        arr[..., :3] = rgb
        # Avoid Pillow's deprecated `mode=` argument by converting after creation.
        return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    except Exception:
        return img


def _contrast_punch_rgb(img: Image.Image, *, contrast: float = 1.22, color: float = 1.08, gamma: float = 0.94) -> Image.Image:
    """
    Increase perceived contrast for dark wraps without shifting hues too much.
    Only affects RGB; alpha is preserved by caller.
    """
    try:
        rgb = img.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(float(contrast))
        rgb = ImageEnhance.Color(rgb).enhance(float(color))
        g = max(0.70, min(1.20, float(gamma)))
        lut = [max(0, min(255, int(round(((i / 255.0) ** g) * 255.0)))) for i in range(256)]
        r, gg, b = rgb.split()
        rgb = Image.merge("RGB", (r.point(lut), gg.point(lut), b.point(lut)))
        return rgb.convert("RGBA")
    except Exception:
        return img.convert("RGBA")


def _vignette_layer(size: int, *, strength: int, rng: Optional["random.Random"] = None) -> Image.Image:
    """
    Dark vignette to make highlights pop (premium photography trick).
    """
    strength_i = max(0, min(255, int(strength)))
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    if rng is None:
        cx = cy = size // 2
    else:
        cx = int(size * (0.50 + rng.uniform(-0.03, 0.03)))
        cy = int(size * (0.52 + rng.uniform(-0.04, 0.02)))
    steps = 10
    for i in range(steps):
        t = i / max(1, steps - 1)
        a = int(strength_i * (t ** 1.6))
        r = int((0.62 + 0.62 * t) * size)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=a, width=max(2, size // 120))
    m = m.filter(ImageFilter.GaussianBlur(radius=max(6, size // 120)))
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.putalpha(m)
    return layer


def _clamp_int(v: Optional[int], lo: int, hi: int) -> Optional[int]:
    if v is None:
        return None
    try:
        iv = int(v)
    except Exception:
        return None
    return max(lo, min(hi, iv))


def _dxt_edge_sharpen_rgba(
    img: Image.Image,
    *,
    strength: float = 0.35,
    radius: float = 1.2,
    percent: int = 140,
    threshold: int = 6,
) -> Image.Image:
    """
    Mild, DXT-friendly sharpening that targets only real edges.

    Rationale: DXT5 + mipmaps can soften thin cutlines/pinstripes. A small, edge-masked unsharp pass
    improves perceived crispness without boosting noise everywhere (which would create banding).
    """
    try:
        s = max(0.0, min(1.0, float(strength)))
        if s <= 0.0:
            return img.convert("RGBA")

        im = img.convert("RGBA")
        a = im.getchannel("A")
        rgb = im.convert("RGB")

        # Edge mask from luminance edges, thresholded and blurred slightly.
        edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p >= 18 else 0)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=max(0.6, float(radius) * 0.55)))

        sharp = rgb.filter(ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=int(threshold)))
        blended = Image.composite(sharp, rgb, edges)
        out = blended.convert("RGBA")
        out.putalpha(a)

        # Blend back toward original for safety.
        if s < 1.0:
            out = Image.blend(im, out, s)
        return out
    except Exception:
        return img.convert("RGBA")


def _apply_finish_design(
    spec: Image.Image,
    *,
    rgb_src: Image.Image,
    mode: str,
    strength: float = 0.35,
) -> Image.Image:
    """
    Optional "finish choreography" layer to make paint feel more premium.

    - edges: make panel cutlines/pinstripe edges glossier (higher alpha)
    - sweep: add a subtle clearcoat sweep (diagonal gloss gradient)
    """
    try:
        m = (mode or "off").strip().lower()
        if m in ("off", "none", ""):
            return spec.convert("L")
        s = max(0.0, min(1.0, float(strength)))
        if s <= 0.0:
            return spec.convert("L")

        base = spec.convert("L")
        w, h = base.size

        if m in ("edges", "edge"):
            rgb = rgb_src.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
            edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p >= 24 else 0)
            edges = edges.filter(ImageFilter.GaussianBlur(radius=max(0.8, min(w, h) / 1400.0)))
            # Add gloss on edges (higher alpha).
            add = Image.new("L", (w, h), 0)
            add = Image.composite(Image.new("L", (w, h), int(40 + 60 * s)), add, edges)
            out = ImageChops.add(base, add)
            return out

        if m in ("sweep", "clearcoat"):
            # Diagonal sweep: stronger gloss around a soft band.
            g = _make_multi_stop_gradient(
                max(w, h),
                stops=[
                    (0.0, (0, 0, 0)),
                    (0.45, (255, 255, 255)),
                    (0.62, (255, 255, 255)),
                    (1.0, (0, 0, 0)),
                ],
                angle_deg=-35.0,
                noise_strength=0.0,
                quant_steps=256,
            ).convert("L").resize((w, h), Image.Resampling.BICUBIC)
            g = g.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(w, h) / 900.0)))
            add = g.point(lambda p: int((p / 255.0) * (55 + 65 * s)))
            out = ImageChops.add(base, add)
            return out

        return base
    except Exception:
        return spec.convert("L")

def _sanitize_common_details_watermarks(details_img: Image.Image) -> Image.Image:
    """
    Best-effort removal of common author/watermark elements from Details textures (e.g. KACKY strap).
    This is intentionally conservative and only triggers when the region looks like a watermark.
    """
    if np is None:
        return details_img
    try:
        img = details_img.convert("RGBA")
        a = img.getchannel("A")
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]
        sx = float(w) / 4096.0
        sy = float(h) / 4096.0

        def _box(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
            X0 = int(round(x0 * sx))
            Y0 = int(round(y0 * sy))
            X1 = int(round(x1 * sx))
            Y1 = int(round(y1 * sy))
            X0 = max(0, min(w, X0))
            X1 = max(0, min(w, X1))
            Y0 = max(0, min(h, Y0))
            Y1 = max(0, min(h, Y1))
            return (X0, Y0, X1, Y1)

        # Strap text region (commonly contains "KACKY").
        strap = _box(2860, 2650, 3130, 3350)
        x0, y0, x1, y1 = strap
        if x1 > x0 and y1 > y0:
            reg = arr[y0:y1, x0:x1, :3].astype(np.float32)
            lum = 0.2126 * reg[..., 0] + 0.7152 * reg[..., 1] + 0.0722 * reg[..., 2]
            bright = (lum > 205.0).mean()
            # If there are enough bright pixels, it's likely a white-on-black watermark.
            if float(bright) > 0.010:
                arr[y0:y1, x0:x1, 0] = 0
                arr[y0:y1, x0:x1, 1] = 0
                arr[y0:y1, x0:x1, 2] = 0

        # Red patch region (some templates have a vivid red badge).
        redp = _box(0, 2450, 620, 2920)
        x0, y0, x1, y1 = redp
        if x1 > x0 and y1 > y0:
            reg = arr[y0:y1, x0:x1, :3].astype(np.float32)
            r = reg[..., 0]
            g = reg[..., 1]
            b = reg[..., 2]
            is_red = (r > 140.0) & (r > g + 35.0) & (r > b + 35.0)
            if float(is_red.mean()) > 0.015:
                arr[y0:y1, x0:x1, 0] = 0
                arr[y0:y1, x0:x1, 1] = 0
                arr[y0:y1, x0:x1, 2] = 0

        out = Image.fromarray(arr).convert("RGBA")
        out.putalpha(a)
        return out
    except Exception:
        return details_img


# Standard Stadium mod model used by many packs (Pink/Kacky examples in this repo).
STANDARD_STADIUM_MAINBODY_SHA256 = "c273680b4b4cfb07d942ec63d56f24a6d12778b6b3fd7b1a7bde133dcc04136c"

# A known-good standard Stadium Diffuse template with near-black unused background.
# This is used to reliably compute UV islands even when the target base zip's Diffuse.dds
# is fully painted (which makes connected-component UV detection collapse into 1 big island).
STANDARD_STADIUM_UV_TEMPLATE_ZIP_CANDIDATES = [
    "examples/KACKIEST-KACKY-9-(black)_by_MINA_TM.zip",
    "examples/KACKIEST-KACKY-10-SKIN-(dark-gray)_by_MINA_TM.zip",
    "examples/KACKIEST-KACKY-9-SKIN-(white)_by_MINA_TM.zip",
    "examples/Pink-Skin-(TMNF_UF)_by_SparkyTM.zip",
]

# Optional font override (set via CLI) for rendering special characters in names/tags.
FONT_OVERRIDE_PATH: Optional[str] = None

# Standard Stadium mudguard/wheel-arch UV island IDs (ranked by area).
# Automatically detected using geometric heuristics:
# - Mirrored pairs (Y coords sum to ~2160 for 2048 texture)
# - Square-ish aspect ratio (0.7 < AR < 1.5) - wheel arches are roughly square
# - Medium size range (300-1500 area)
# Detected: islands 17/18 (AR=0.89) and 26/27 (AR=1.35) are the 4 mudguards.
STADIUM_MUDGUARD_ISLAND_IDS: List[int] = [17, 18, 26, 27]

# Alternative: explicit normalized rects at 2048x2048 reference (x0, y0, x1, y1).
# Used when island masks are not available or for packs with non-standard UV layouts.
# These rects are derived from the UV atlas bounding boxes for the mudguard islands.
STADIUM_MUDGUARD_RECTS_2048: List[Tuple[int, int, int, int]] = [
    (264, 760, 396, 908),    # Island 18: mudguard pair A (upper in UV)
    (264, 1252, 396, 1400),  # Island 17: mudguard pair A (lower in UV, mirrored)
    (408, 784, 516, 864),    # Island 26: mudguard pair B (upper in UV)
    (408, 1296, 516, 1376),  # Island 27: mudguard pair B (lower in UV, mirrored)
]


def _tint_logo(logo_rgba: Image.Image, *, rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Convert a logo into a single-color decal, preserving only alpha.
    Useful for black/white-only skins.
    """
    logo_rgba = logo_rgba.convert("RGBA")
    a = logo_rgba.getchannel("A")
    out = Image.new("RGBA", logo_rgba.size, (rgb[0], rgb[1], rgb[2], 0))
    out.putalpha(a)
    return out


def _prepare_sticker_rgba(img: Image.Image, *, tolerance: int = 32) -> Image.Image:
    """
    Prepare a sticker for compositing:
    - convert to RGBA
    - if it has an opaque flat-ish background, cut it out (works for white/black backgrounds)
    - lightly trim empty border
    """
    im = img.convert("RGBA")
    # Only auto-cutout when the sticker has NO transparency at all (common for flat-background PNGs).
    # If it already has transparency (like our sponsor plates), keep it as-is.
    try:
        mn, mx = im.getchannel("A").getextrema()
        if mn == 255 and mx == 255:
            im = _auto_cutout_logo_background(im, tolerance=tolerance)
    except Exception:
        pass
    # Trim fully transparent border
    bb = im.getchannel("A").getbbox()
    if bb:
        im = im.crop(bb)
    return im


def _load_standard_stadium_uv_template_diffuse(*, size: Tuple[int, int]) -> Optional[Image.Image]:
    """
    Load a known-good Stadium Diffuse template (near-black unused background) to compute UV islands.
    Returns an RGBA image resized to `size`, or None if not found.
    """
    root = Path(__file__).resolve().parent
    for rel in STANDARD_STADIUM_UV_TEMPLATE_ZIP_CANDIDATES:
        zp = root / rel
        try:
            if not zp.exists():
                continue
            with zipfile.ZipFile(zp, "r") as z:
                if "Diffuse.dds" not in z.namelist():
                    continue
                tex = z.read("Diffuse.dds")
            img = Image.open(io.BytesIO(tex)).convert("RGBA").resize(size, Image.Resampling.NEAREST)
            return img
        except Exception:
            continue
    return None


def _sprinkle_stickers(
    base: Image.Image,
    *,
    stickers: Sequence[Image.Image],
    mask_l: Optional[Image.Image],
    exclude_rects: Sequence[Tuple[int, int, int, int]],
    rng: "random.Random",
    count: int,
    min_scale: float,
    max_scale: float,
    rotate: bool,
    mode: str = "random",  # "random" | "grid"
) -> Image.Image:
    """
    Randomly sprinkle sticker images across the texture.
    - mask_l (L): where placement is allowed (255 = allowed). If None, place anywhere.
    - exclude_rects: avoid placing sticker centers in these rects (e.g. wing plate)
    """
    if not stickers or count <= 0:
        return base

    img = base.convert("RGBA")
    W, H = img.size
    shortest = max(1, min(W, H))
    min_s = max(0.005, float(min_scale))
    max_s = max(min_s, float(max_scale))

    mask_arr = None
    if mask_l is not None:
        try:
            mask_arr = np.asarray(mask_l.resize((W, H), Image.Resampling.NEAREST).convert("L"), dtype=np.uint8) if np is not None else None
        except Exception:
            mask_arr = None

    def in_exclude(cx: int, cy: int) -> bool:
        for x0, y0, x1, y1 in exclude_rects:
            if x0 <= cx < x1 and y0 <= cy < y1:
                return True
        return False

    def _place_at(cx: int, cy: int) -> bool:
        nonlocal img
        if in_exclude(cx, cy):
            return False
        if mask_arr is not None and mask_arr[cy, cx] < 16:
            return False

        st = stickers[int(rng.random() * len(stickers))].convert("RGBA")
        sw, sh = st.size
        if sw < 2 or sh < 2:
            return False

        target = int(rng.uniform(min_s, max_s) * shortest)
        target = max(18, min(shortest // 3, target))
        sc = target / float(max(sw, sh))
        st = st.resize((max(1, int(sw * sc)), max(1, int(sh * sc))), Image.Resampling.LANCZOS)
        if rotate:
            ang = float(rng.uniform(-35.0, 35.0))
            st = st.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)

        x = int(cx - st.size[0] // 2)
        y = int(cy - st.size[1] // 2)

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.alpha_composite(st, (x, y))
        if mask_l is not None:
            try:
                a = layer.getchannel("A")
                a = ImageChops.multiply(a, mask_l.resize((W, H), Image.Resampling.NEAREST).convert("L"))
                layer.putalpha(a)
            except Exception:
                pass
        img = Image.alpha_composite(img, layer)
        return True

    placed = 0
    mode = (mode or "random").strip().lower()
    if mode == "grid":
        # Even coverage: grid + jitter. This avoids clusters and keeps stickers all over the car.
        gx = max(6, int(math.sqrt(max(1, count))))
        gy = max(6, int(math.ceil(float(count) / float(gx))))
        step_x = float(W) / float(gx)
        step_y = float(H) / float(gy)
        jitter_x = step_x * 0.38
        jitter_y = step_y * 0.38
        for iy in range(gy):
            for ix in range(gx):
                if placed >= count:
                    break
                # Try a few jitter attempts inside this cell to find a valid masked spot.
                ok = False
                for _ in range(6):
                    cx = int((ix + 0.5) * step_x + rng.uniform(-jitter_x, jitter_x))
                    cy = int((iy + 0.5) * step_y + rng.uniform(-jitter_y, jitter_y))
                    cx = max(0, min(W - 1, cx))
                    cy = max(0, min(H - 1, cy))
                    if _place_at(cx, cy):
                        ok = True
                        break
                if ok:
                    placed += 1
            if placed >= count:
                break
    else:
        # Random sprinkle (legacy behavior)
        attempts = 0
        max_attempts = max(500, count * 45)
        while placed < count and attempts < max_attempts:
            attempts += 1
            cx = int(rng.random() * W)
            cy = int(rng.random() * H)
            if _place_at(cx, cy):
                placed += 1

    return img


def _tile_sticker_on_island(
    base: Image.Image,
    *,
    sticker: Image.Image,
    island_mask_l: Image.Image,
    island_bbox: Tuple[int, int, int, int],
    tile_scale: float,
    rng: "random.Random",
) -> Image.Image:
    """
    Tile a sticker image inside a specific UV island.
    """
    img = base.convert("RGBA")
    W, H = img.size
    x0, y0, x1, y1 = island_bbox
    rw = max(1, x1 - x0)
    rh = max(1, y1 - y0)
    tile_scale = max(0.10, min(1.0, float(tile_scale)))

    st = sticker.convert("RGBA")
    sw, sh = st.size
    if sw < 2 or sh < 2:
        return img

    # Determine tile size relative to island.
    t = int(min(rw, rh) * tile_scale)
    t = max(10, min(min(rw, rh), t))
    sc = t / float(max(sw, sh))
    st = st.resize((max(1, int(sw * sc)), max(1, int(sh * sc))), Image.Resampling.LANCZOS)

    # Small random rotation per-tile looks fun.
    step = max(8, int(max(st.size) * 0.85))
    patt = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    for yy in range(-step, rh + step, step):
        for xx in range(-step, rw + step, step):
            ang = float(rng.uniform(-18.0, 18.0))
            tile = st.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
            patt.alpha_composite(tile, (xx, yy))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(patt, (x0, y0))
    a = layer.getchannel("A")
    a = ImageChops.multiply(a, island_mask_l.resize((W, H), Image.Resampling.NEAREST).convert("L"))
    layer.putalpha(a)
    return Image.alpha_composite(img, layer)


def _place_decal_on_island(
    base: Image.Image,
    *,
    decal: Image.Image,
    island_mask_l: Image.Image,
    island_bbox: Tuple[int, int, int, int],
    center_xy: Optional[Tuple[int, int]] = None,
    max_size: Optional[Tuple[int, int]] = None,
    rotate_deg: float = 0.0,
    flip: Optional[str] = None,  # None | "lr" | "tb"
    opacity: float = 1.0,
) -> Image.Image:
    """
    Place a single decal inside an island bbox, clipped to the island mask.
    flip: "lr" = left-right, "tb" = top-bottom (useful for mirrored UVs).
    """
    img = base.convert("RGBA")
    W, H = img.size
    x0, y0, x1, y1 = island_bbox
    rw = max(1, x1 - x0)
    rh = max(1, y1 - y0)
    if rw <= 1 or rh <= 1:
        return img

    dec = decal.convert("RGBA")
    if flip == "lr":
        dec = dec.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip == "tb":
        dec = dec.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Fit decal to a target region (contain).
    pad = max(6, int(min(rw, rh) * 0.04))
    tw = max(1, rw - 2 * pad)
    th = max(1, rh - 2 * pad)
    if max_size is not None:
        tw = min(tw, max(1, int(max_size[0])))
        th = min(th, max(1, int(max_size[1])))
    dw, dh = dec.size
    if dw < 2 or dh < 2:
        return img
    s = min(tw / float(dw), th / float(dh))
    s = max(0.05, min(10.0, s))
    nw = max(1, int(round(dw * s)))
    nh = max(1, int(round(dh * s)))
    dec = dec.resize((nw, nh), Image.Resampling.LANCZOS)

    if abs(float(rotate_deg)) > 0.01:
        dec = dec.rotate(float(rotate_deg), expand=True, resample=Image.Resampling.BICUBIC)

    op = float(max(0.0, min(1.0, opacity)))
    if op < 0.999:
        try:
            a = dec.getchannel("A").point(lambda p: int(p * op))
            dec.putalpha(a)
        except Exception:
            pass

    if center_xy is None:
        cx = x0 + rw // 2
        cy = y0 + rh // 2
    else:
        cx, cy = center_xy

    # If the requested center is outside the island mask, snap it to a nearby valid pixel.
    try:
        m = np.asarray(island_mask_l.resize((W, H), Image.Resampling.NEAREST).convert("L"), dtype=np.uint8) if np is not None else None
        if m is not None:
            cx = max(x0, min(x1 - 1, int(cx)))
            cy = max(y0, min(y1 - 1, int(cy)))
            if m[cy, cx] < 8:
                best = None
                # Search a small radius for any mask pixel.
                for r in [6, 10, 14, 18, 24, 32]:
                    ylo = max(y0, cy - r)
                    yhi = min(y1, cy + r + 1)
                    xlo = max(x0, cx - r)
                    xhi = min(x1, cx + r + 1)
                    window = m[ylo:yhi, xlo:xhi]
                    ys, xs = np.where(window >= 8)
                    if xs.size > 0:
                        # Choose closest
                        gx = xs + xlo
                        gy = ys + ylo
                        dx = gx - cx
                        dy = gy - cy
                        i = int(np.argmin(dx * dx + dy * dy))
                        best = (int(gx[i]), int(gy[i]))
                        break
                if best is not None:
                    cx, cy = best
    except Exception:
        pass

    px = int(cx - dec.size[0] // 2)
    py = int(cy - dec.size[1] // 2)

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(dec, (px, py))
    a = layer.getchannel("A")
    a = ImageChops.multiply(a, island_mask_l.resize((W, H), Image.Resampling.NEAREST).convert("L"))
    layer.putalpha(a)
    return Image.alpha_composite(img, layer)


def _make_polish_flag_rgba(*, w: int, h: int) -> Image.Image:
    """
    Poland flag (PL): white over red.
    """
    w = max(24, int(w))
    h = max(18, int(h))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Border
    bw = max(2, min(w, h) // 18)
    d.rectangle((0, 0, w - 1, h - 1), outline=(0, 0, 0, 220), width=bw)
    # Inner bands
    ix0, iy0, ix1, iy1 = bw, bw, w - bw, h - bw
    mid = iy0 + (iy1 - iy0) // 2
    d.rectangle((ix0, iy0, ix1, mid), fill=(245, 245, 245, 255))
    d.rectangle((ix0, mid, ix1, iy1), fill=(220, 20, 60, 255))  # crimson-ish red
    return img


def _convex_hull(points: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    2D convex hull using the monotonic chain algorithm.
    Returns hull points in CCW order.
    """
    pts = sorted(set((int(x), int(y)) for x, y in points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _auto_cutout_logo_background(
    logo_rgba: Image.Image,
    *,
    tolerance: int = 28,
) -> Image.Image:
    """
    If a logo has an opaque flat-ish background (common when exported without alpha),
    remove ONLY the edge-connected background via flood fill from the image borders.

    This keeps interior dark shapes (e.g. cave entrance) intact as long as they are not connected
    to the outer background.
    """
    im = logo_rgba.convert("RGBA")
    w, h = im.size
    if w < 2 or h < 2:
        return im

    a = im.getchannel("A")
    px = im.load()

    # Determine the opaque bbox; handles logos with transparent margins but an opaque plate inside.
    # We use alpha>0 because some logos have antialiased edges with partial alpha.
    bbox = a.getbbox()
    if bbox is None:
        return im
    x0, y0, x1, y1 = bbox  # right/bottom are exclusive
    x1i = x1 - 1
    y1i = y1 - 1

    # Choose background color:
    # - If the bbox touches the image border, use corner sampling (classic case: full opaque background).
    # - Otherwise, sample the border of the opaque bbox (common case: transparent padding around a solid plate).
    samples: List[Tuple[int, int, int]] = []
    if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:
        corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
        for r, g, b, _a in corners:
            samples.append((r, g, b))
    else:
        # Sample border pixels of bbox where alpha is present (avoid transparent padding).
        for x in range(x0, x1):
            for y in (y0, y1i):
                r, g, b, _a = px[x, y]
                if _a > 0:
                    samples.append((r, g, b))
        for y in range(y0, y1):
            for x in (x0, x1i):
                r, g, b, _a = px[x, y]
                if _a > 0:
                    samples.append((r, g, b))

    if not samples:
        # Fallback: assume black background
        br, bgc, bb = (0, 0, 0)
    else:
        # Use median to be robust to a few non-background border pixels.
        rs = sorted([s[0] for s in samples])
        gs = sorted([s[1] for s in samples])
        bs = sorted([s[2] for s in samples])
        mid = len(samples) // 2
        br, bgc, bb = (rs[mid], gs[mid], bs[mid])
    tol2 = int(tolerance) * int(tolerance)

    def close_to_bg(r: int, g: int, b: int) -> bool:
        dr = r - br
        dg = g - bgc
        db = b - bb
        return (dr * dr + dg * dg + db * db) <= tol2

    # Build a candidate background mask inside the opaque bbox and flood-fill it from the bbox border.
    visited = [[False] * w for _ in range(h)]
    stack: List[Tuple[int, int]] = []

    # Seed from bbox border positions (treat transparency outside bbox as "outside world").
    for x in range(x0, x1):
        for y in (y0, y1i):
            r, g, b, _a = px[x, y]
            if _a > 0 and close_to_bg(r, g, b):
                stack.append((x, y))
    for y in range(y0, y1):
        for x in (x0, x1i):
            r, g, b, _a = px[x, y]
            if _a > 0 and close_to_bg(r, g, b):
                stack.append((x, y))

    # If very few candidates on the border, likely no plate; keep as-is.
    if len(stack) < 8:
        return im

    out = im.copy()
    out_px = out.load()
    removed = 0
    while stack:
        x, y = stack.pop()
        if x < x0 or y < y0 or x > x1i or y > y1i:
            continue
        if visited[y][x]:
            continue
        visited[y][x] = True
        r, g, b, _a = px[x, y]
        if _a <= 0 or not close_to_bg(r, g, b):
            continue
        out_px[x, y] = (r, g, b, 0)
        removed += 1
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    # If we didn't remove much, assume false positive.
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    if removed < int(0.03 * bbox_area):
        return im

    # Optional: preserve a "cave opening" / dark interior that is part of the logo design.
    # Many logos (including cavern_final_logo.png) are drawn on a dark plate; removing the plate also
    # makes the interior opening transparent, which looks wrong on-car. We reconstruct a dark interior
    # by filling the convex hull of the remaining foreground and painting only the missing interior pixels.
    try:
        if np is not None:
            bg_lum = 0.2126 * br + 0.7152 * bgc + 0.0722 * bb
            if bg_lum < 60.0:
                out_a = np.asarray(out.getchannel("A"), dtype=np.uint8)
                fg = (out_a > 0)
                ys2, xs2 = np.where(fg)
                if xs2.size >= 40:
                    pts = list(zip(xs2.tolist(), ys2.tolist()))
                    hull = _convex_hull(pts)
                    if len(hull) >= 3:
                        hull_mask = Image.new("L", out.size, 0)
                        ImageDraw.Draw(hull_mask).polygon(hull, fill=255)
                        hull_arr = np.asarray(hull_mask, dtype=np.uint8) > 0
                        fill = hull_arr & (out_a == 0)
                        hull_area = int(hull_arr.sum())
                        fill_area = int(fill.sum())
                        # Only fill when it looks like a true interior (avoid turning into a big blob).
                        if hull_area > 0 and (0.05 * hull_area) <= fill_area <= (0.65 * hull_area):
                            out_arr = np.asarray(out, dtype=np.uint8).copy()
                            out_arr[fill, 0] = br
                            out_arr[fill, 1] = bgc
                            out_arr[fill, 2] = bb
                            out_arr[fill, 3] = 255
                            out = Image.fromarray(out_arr)
    except Exception:
        pass

    return out


def _mean_luminance_rgba(img_rgba: Image.Image) -> float:
    """
    Mean luminance of non-transparent pixels (0..255). Used to pick a contrasting decal plate color.
    """
    im = img_rgba.convert("RGBA")
    if np is not None:
        arr = np.asarray(im, dtype=np.uint8)
        a = arr[..., 3]
        m = a > 24
        if not np.any(m):
            return 255.0
        rgb = arr[..., :3][m].astype(np.float32)
        lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        return float(lum.mean())
    # Fallback: downsample and iterate
    sm = im.resize((64, 64), Image.Resampling.LANCZOS)
    px = list(sm.getdata())
    vals = []
    for r, g, b, a in px:
        if a <= 24:
            continue
        vals.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return float(sum(vals) / max(1, len(vals)))


def _make_tmnf_finish_alpha_from_rgb(
    img_rgb: Image.Image,
    *,
    neutral: int = 0x8E,
) -> Image.Image:
    """
    Approximate the classic TMN/TMNF alpha-channel workflow described in community tutorials:
    - Alpha is NOT transparency; it's a finish/material channel.
    - BLACK alpha => brightest color, no shine.
    - WHITE alpha => duller color, highest reflection.

    We do NOT try to replicate anyone's exact alpha artwork; we generate a plausible, readable finish map:
    - Dark base areas get higher reflection (higher alpha) for "glossy black"
    - Bright, saturated accents get lower alpha to keep them vibrant and readable
    - Near-white highlights keep lower alpha to avoid turning whites into greys
    """
    base = img_rgb.convert("RGB")
    w, h = base.size

    if np is None:
        return Image.new("L", (w, h), int(neutral))

    arr = np.asarray(base, dtype=np.uint8)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx / 255.0
    # Avoid runtime warnings on near-black pixels and keep values finite.
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)
    sat = np.nan_to_num(sat, nan=0.0, posinf=0.0, neginf=0.0)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    a = np.full((h, w), float(int(neutral)), dtype=np.float32)

    # Dark glossy base: push alpha up
    dark = v < 0.18
    a[dark] = 210.0 + (0.18 - v[dark]) * 180.0  # up to ~242

    # Bright saturated accents: lower alpha so they stay bright (matte-ish)
    accent = (sat > 0.35) & (v > 0.50)
    a[accent] = 35.0 + (1.0 - v[accent]) * 60.0  # ~35..95

    # Near-white highlights: keep low-to-mid alpha so white doesn't go grey
    whiteish = (v > 0.90) & (sat < 0.25)
    a[whiteish] = 65.0 + (1.0 - v[whiteish]) * 40.0  # ~65..75

    # Midtone structure: slight reflectivity, but avoid washing out colors
    mid = (~dark) & (~accent) & (~whiteish)
    a[mid] = float(int(neutral)) + (0.55 - lum[mid]) * 35.0

    a = np.clip(a, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(a).convert("L")
    out = out.filter(ImageFilter.GaussianBlur(radius=max(0.5, min(w, h) / 1024.0)))
    return out


def _palette_map_from_luminance(
    lum: "np.ndarray",
    *,
    c0: Tuple[int, int, int],
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    mid: float = 0.52,
    gamma: float = 1.0,
    contrast: float = 1.0,
) -> "np.ndarray":
    """
    Map a luminance field (0..1) into 3 colors with a mid stop.
    Produces an RGB uint8 array (H,W,3).
    """
    if np is None:
        raise RuntimeError("Numpy required for palette mapping.")
    t = lum.astype(np.float32)
    # Contrast around 0.5
    if contrast != 1.0:
        t = np.clip((t - 0.5) * contrast + 0.5, 0.0, 1.0)
    # Gamma
    if gamma != 1.0:
        t = np.clip(t, 0.0, 1.0) ** float(gamma)

    mid = float(np.clip(mid, 0.05, 0.95))
    out = np.zeros((*t.shape, 3), dtype=np.float32)

    c0a = np.array(c0, dtype=np.float32)
    c1a = np.array(c1, dtype=np.float32)
    c2a = np.array(c2, dtype=np.float32)

    m0 = t <= mid
    m1 = ~m0

    # 0..mid
    if np.any(m0):
        tt = t[m0] / max(1e-6, mid)
        out[m0] = c0a * (1.0 - tt[:, None]) + c1a * tt[:, None]
    # mid..1
    if np.any(m1):
        tt = (t[m1] - mid) / max(1e-6, (1.0 - mid))
        out[m1] = c1a * (1.0 - tt[:, None]) + c2a * tt[:, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def _build_inspired_layer_from_zip(
    *,
    inspire_zip_path: Path,
    out_size: Tuple[int, int],
    source: str = "auto",  # "auto"|"diffuse"|"details"
    c0: Tuple[int, int, int],
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    rng: "random.Random",
) -> Image.Image:
    """
    Load an example skin zip and build an RGBA layer by palette-mapping its luminance.
    This preserves the handmade composition (shapes/texture placement) but applies our colors.
    """
    if np is None:
        raise RuntimeError("Numpy required for inspire pipeline.")

    with zipfile.ZipFile(inspire_zip_path, "r") as z:
        names = set(z.namelist())
        if "Diffuse.dds" not in names:
            raise RuntimeError("Inspire zip missing Diffuse.dds")
        diff = Image.open(io.BytesIO(z.read("Diffuse.dds"))).convert("RGBA")
        det = None
        if "Details.dds" in names:
            try:
                det = Image.open(io.BytesIO(z.read("Details.dds"))).convert("RGBA")
            except Exception:
                det = None

    # Decide source
    chosen: Image.Image
    chosen_kind = "diffuse"
    if source == "diffuse":
        chosen = diff
        chosen_kind = "diffuse"
    elif source == "details":
        chosen = det if det is not None else diff
        chosen_kind = "details" if det is not None else "diffuse"
    else:
        # auto: prefer Diffuse if it has variance; otherwise Details
        sm = diff.resize((256, 256), Image.Resampling.BILINEAR)
        arr = np.asarray(sm, dtype=np.uint8)
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        if float(lum.std()) > 1.2:
            chosen = diff
            chosen_kind = "diffuse"
        else:
            chosen = det if det is not None else diff
            chosen_kind = "details" if det is not None else "diffuse"

    chosen = chosen.resize(out_size, Image.Resampling.BILINEAR)

    # IMPORTANT: sanitize inspiration to avoid importing watermarks / author credits.
    # We only want the *large-scale composition* (shapes/placement), not fine details like
    # "made by ..." text, signatures, or logos. We do this by heavy downsample+upsample.
    w, h = out_size
    ds = max(96, min(w, h) // 8)  # 256 on 2048 skins; removes small text well.
    smooth = chosen.resize((ds, ds), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    smooth = smooth.filter(ImageFilter.GaussianBlur(radius=max(1, min(w, h) // 512)))

    arr = np.asarray(smooth.convert("RGB"), dtype=np.float32) / 255.0
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    # Boost visibility like the examples:
    # - Diffuse-based inspirations already have bold shapes -> moderate contrast.
    # - Details-based inspirations can be subtle -> stronger contrast + quantization to make shapes readable.
    if chosen_kind == "details":
        contrast = float(rng.uniform(1.35, 1.95))
        gamma = float(rng.uniform(0.80, 1.00))
        mid = float(rng.uniform(0.44, 0.58))
        # Quantize luminance into steps to avoid “flat tint” look.
        levels = int(rng.choice([10, 12, 14, 16, 18]))
        lum_q = np.round(lum * levels) / float(levels)
        # Slight blur to avoid harsh banding after quantization.
        lum = lum_q
    else:
        contrast = float(rng.uniform(1.15, 1.55))
        gamma = float(rng.uniform(0.85, 1.05))
        mid = float(rng.uniform(0.46, 0.58))

    rgb = _palette_map_from_luminance(lum, c0=c0, c1=c1, c2=c2, mid=mid, gamma=gamma, contrast=contrast)
    out = Image.fromarray(rgb).convert("RGBA")
    out.putalpha(255)

    # Add a subtle edge highlight extracted from the *sanitized* source for “handmade crispness”.
    # Using the sanitized luminance prevents small watermark text from showing up as edges.
    # Pillow 11 warns that 'mode=' will be removed; infer from array shape.
    src_l = Image.fromarray((lum * 255).astype(np.uint8)).convert("L")
    edges = src_l.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 24 else 0)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=1))
    edge_layer = Image.new("RGBA", out.size, (c2[0], c2[1], c2[2], 0))
    edge_layer.putalpha(edges.point(lambda p: int(p * rng.uniform(0.10, 0.22))))
    out = Image.alpha_composite(out, edge_layer)

    return out


def _estimate_nose_logo_placement(
    mask_l: Image.Image,
    *,
    mode: str,
    rng: "random.Random",
) -> Tuple[int, int, int, float]:
    """
    Estimate a good placement for a logo on the nose/top island (rank 2) by analyzing the island shape.

    Returns (cx, cy, target_size, rotate_deg) in FULL-res pixels relative to mask image size.

    mode:
    - 'front'   : near the narrow tip (between front wheels / nose tip)
    - 'cockpit' : closer to the wider end (in front of driver)
    """
    if np is None:
        # Fallback: center-ish with simple offsets
        w, h = mask_l.size
        if mode == "front":
            return (int(w * 0.52), int(h * 0.20), int(min(w, h) * 0.14), 0.0)
        return (int(w * 0.52), int(h * 0.48), int(min(w, h) * 0.11), 0.0)

    W, H = mask_l.size
    # Downsample for faster PCA
    scale = max(1, int(max(W, H) / 512))
    sm = mask_l.resize((max(1, W // scale), max(1, H // scale)), Image.Resampling.NEAREST)
    m = np.asarray(sm, dtype=np.uint8) > 0
    ys, xs = np.where(m)
    if xs.size < 300:
        # Too small / weird mask
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)

    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = coords.mean(axis=0)
    c = coords - mean
    cov = (c.T @ c) / float(c.shape[0])
    if not np.isfinite(cov).all():
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)

    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    if not np.isfinite(eigvals).all() or not np.isfinite(eigvecs).all():
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)

    v = eigvecs[:, int(np.argmax(eigvals))].astype(np.float64)  # major axis
    v = v / (np.linalg.norm(v) + 1e-9)
    if not np.isfinite(v).all():
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)

    vp = np.array([-v[1], v[0]], dtype=np.float64)  # perpendicular

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        proj = c @ v
        perp = c @ vp
    if not np.isfinite(proj).all() or not np.isfinite(perp).all():
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)
    tmin = float(proj.min())
    tmax = float(proj.max())
    tr = tmax - tmin
    if tr < 1e-3:
        if mode == "front":
            return (int(W * 0.52), int(H * 0.20), int(min(W, H) * 0.14), 0.0)
        return (int(W * 0.52), int(H * 0.48), int(min(W, H) * 0.11), 0.0)

    # Identify which end is the "tip" by looking at perpendicular width near the extremes.
    win = 0.08 * tr
    m0 = proj < (tmin + win)
    m1 = proj > (tmax - win)
    w0 = float(np.percentile(np.abs(perp[m0]), 90)) if np.any(m0) else 1e9
    w1 = float(np.percentile(np.abs(perp[m1]), 90)) if np.any(m1) else 1e9
    tip_is_min = w0 < w1
    tip_t = tmin if tip_is_min else tmax
    direction = 1.0 if tip_is_min else -1.0  # from tip towards rear along +proj

    # Rotation: align logo "up" to point toward tip (front direction) along the major axis.
    # front_dir is from rear -> tip in texture coords.
    front_dir = (-direction) * v  # type: ignore
    # Angle of front_dir in image coords (x right, y down); logo's "up" points to (0,-1) i.e. -90 deg.
    front_angle = math.degrees(math.atan2(float(front_dir[1]), float(front_dir[0])))
    rotate_deg = front_angle + 90.0

    # Choose a normalized position from tip
    if mode == "front":
        u = float(0.12 + rng.uniform(-0.015, 0.015))
    elif mode == "cockpit":
        # Slightly closer to center than before; tends to land on the visible "in front of driver" panel.
        u = float(0.50 + rng.uniform(-0.03, 0.03))
    else:
        u = float(0.35 + rng.uniform(-0.02, 0.02))
    t = tip_t + direction * (u * tr)

    # Pick points near t and average to get centerline location.
    band = 0.04 * tr
    sel = np.abs(proj - t) < band
    if np.count_nonzero(sel) < 120:
        band = 0.07 * tr
        sel = np.abs(proj - t) < band
    if np.count_nonzero(sel) < 60:
        sel = np.abs(proj - t) < (0.10 * tr)

    cx_s = float(coords[sel, 0].mean()) if np.any(sel) else float(mean[0])
    cy_s = float(coords[sel, 1].mean()) if np.any(sel) else float(mean[1])

    # Local half-width -> logo size suggestion
    hw = float(np.percentile(np.abs(perp[sel]), 92)) if np.any(sel) else float(np.percentile(np.abs(perp), 75))
    # Convert to full-res pixels
    cx = int(round(cx_s * scale))
    cy = int(round(cy_s * scale))
    # For safety: cap size and keep it reasonable
    target = int(round(max(18, min(hw * 1.45, 0.42 * min(sm.size))) * scale))
    # Slight clamp
    target = max(24, min(int(min(W, H) * 0.22), target))

    return (cx, cy, target, float(rotate_deg))


def _parse_hex_color(s: str) -> Tuple[int, int, int, int]:
    """
    Accepts:
    - '#RRGGBB'
    - '#RRGGBBAA'
    - 'RRGGBB'
    - 'RRGGBBAA'
    """
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) not in (6, 8):
        raise argparse.ArgumentTypeError("Color must be RRGGBB or RRGGBBAA (optionally prefixed with #).")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    a = int(s[6:8], 16) if len(s) == 8 else 255
    return (r, g, b, a)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _iter_mip_sizes(width: int, height: int) -> Iterable[Tuple[int, int]]:
    w, h = width, height
    yield (w, h)
    while w > 1 or h > 1:
        w = max(1, w // 2)
        h = max(1, h // 2)
        yield (w, h)


def _generate_mipmaps(img: Image.Image) -> List[Image.Image]:
    """
    Downscale with BOX filter (good for mipmaps).
    """
    img = img.convert("RGBA")
    levels = [img]
    for (w, h) in list(_iter_mip_sizes(*img.size))[1:]:
        levels.append(levels[-1].resize((w, h), Image.Resampling.BOX))
    return levels


def _rgba_to_bgra_bytes(img: Image.Image) -> bytes:
    """
    DDS A8R8G8B8 masks correspond to little-endian BGRA byte order.
    """
    rgba = img.convert("RGBA")
    return rgba.tobytes("raw", "BGRA")


def build_dds_rgba8_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    """
    Build an uncompressed DDS (RGBA8) as bytes with optional mipmaps.
    """
    base = img.convert("RGBA")
    width, height = base.size

    if mipmaps:
        levels = _generate_mipmaps(base)
        mip_count = len(levels)
    else:
        levels = [base]
        mip_count = 0  # when DDSD_MIPMAPCOUNT not set

    # DDS constants
    DDS_MAGIC = b"DDS "

    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000

    DDPF_ALPHAPIXELS = 0x1
    DDPF_RGB = 0x40

    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = width * 4
    dwDepth = 0
    dwMipMapCount = mip_count
    dwReserved1 = (0,) * 11

    # Pixel format: RGBA8 (A8R8G8B8 masks)
    pfSize = 32
    pfFlags = DDPF_RGB | DDPF_ALPHAPIXELS
    pfFourCC = 0
    pfRGBBitCount = 32
    pfRBitMask = 0x00FF0000
    pfGBitMask = 0x0000FF00
    pfBBitMask = 0x000000FF
    pfABitMask = 0xFF000000

    dwCaps = DDSCAPS_TEXTURE
    if mipmaps:
        dwCaps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        pfSize,
        pfFlags,
        pfFourCC,  # FourCC (0 = none)
        pfRGBBitCount,
        pfRBitMask,
        pfGBitMask,
        pfBBitMask,
        pfABitMask,
        dwCaps,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    if len(header) != 124:
        raise RuntimeError(f"Internal error: DDS header length is {len(header)}, expected 124.")

    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_rgba_to_bgra_bytes(level))
    return b"".join(chunks)


def _rgb_to_565(r: int, g: int, b: int) -> int:
    # Quantize 0..255 -> 5/6/5 with rounding.
    r5 = (r * 31 + 127) // 255
    g6 = (g * 63 + 127) // 255
    b5 = (b * 31 + 127) // 255
    return (r5 << 11) | (g6 << 5) | b5


def _rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (r, g, b)


def _compress_dxt5_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """
    Compress a single 4x4 RGBA block to DXT5 (16 bytes).

    pixels: 16 tuples in row-major order.
    """
    if len(pixels) != 16:
        raise ValueError("DXT5 block must have exactly 16 pixels.")

    # --- Alpha block ---
    alphas = [p[3] for p in pixels]
    a0 = max(alphas)
    a1 = min(alphas)

    alpha_palette: List[int]
    if a0 > a1:
        alpha_palette = [
            a0,
            a1,
            (6 * a0 + 1 * a1) // 7,
            (5 * a0 + 2 * a1) // 7,
            (4 * a0 + 3 * a1) // 7,
            (3 * a0 + 4 * a1) // 7,
            (2 * a0 + 5 * a1) // 7,
            (1 * a0 + 6 * a1) // 7,
        ]
    else:
        alpha_palette = [
            a0,
            a1,
            (4 * a0 + 1 * a1) // 5 if a0 != a1 else a0,
            (3 * a0 + 2 * a1) // 5 if a0 != a1 else a0,
            (2 * a0 + 3 * a1) // 5 if a0 != a1 else a0,
            (1 * a0 + 4 * a1) // 5 if a0 != a1 else a0,
            0,
            255,
        ]

    alpha_bits = 0
    for i, a in enumerate(alphas):
        # nearest palette entry
        best_idx = 0
        best_err = 10**9
        for idx, pa in enumerate(alpha_palette):
            err = abs(a - pa)
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        alpha_bits |= (best_idx & 0x7) << (3 * i)

    alpha_bytes = bytes((a0, a1)) + alpha_bits.to_bytes(6, "little")

    # --- Color block (DXT1-style, always 4-color mode) ---
    # Pick endpoints based on luminance extremes (fast, decent quality).
    best_min = pixels[0]
    best_max = pixels[0]
    lum_min = 77 * best_min[0] + 150 * best_min[1] + 29 * best_min[2]
    lum_max = lum_min
    for p in pixels[1:]:
        lum = 77 * p[0] + 150 * p[1] + 29 * p[2]
        if lum < lum_min:
            lum_min = lum
            best_min = p
        elif lum > lum_max:
            lum_max = lum
            best_max = p

    c0 = _rgb_to_565(best_max[0], best_max[1], best_max[2])
    c1 = _rgb_to_565(best_min[0], best_min[1], best_min[2])

    if c0 == c1:
        # Force a 4-color palette by making endpoints different.
        c1 = c0 - 1 if c0 > 0 else 1

    if c0 < c1:
        c0, c1 = c1, c0

    r0, g0, b0 = _rgb565_to_rgb888(c0)
    r1, g1, b1 = _rgb565_to_rgb888(c1)

    palette = [
        (r0, g0, b0),
        (r1, g1, b1),
        ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3),
        ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3),
    ]

    color_bits = 0
    for i, p in enumerate(pixels):
        pr, pg, pb = p[0], p[1], p[2]
        best_idx = 0
        best_err = 10**18
        for idx, (cr, cg, cb) in enumerate(palette):
            dr = pr - cr
            dg = pg - cg
            db = pb - cb
            err = dr * dr + dg * dg + db * db
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        color_bits |= (best_idx & 0x3) << (2 * i)

    color_bytes = struct.pack("<HHI", c0, c1, color_bits)
    return alpha_bytes + color_bytes


def _compress_image_to_dxt5(img: Image.Image) -> bytes:
    """
    Compress a full RGBA image into DXT5 blocks.
    """
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4

    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt5_block(block))
    return bytes(out)


def _compress_dxt3_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """
    Compress a single 4x4 RGBA block to DXT3 (16 bytes).

    DXT3 stores explicit alpha as 16 x 4-bit values (8 bytes), followed by a DXT1-style color block (8 bytes).
    """
    if len(pixels) != 16:
        raise ValueError("DXT3 block must have exactly 16 pixels.")

    # --- Alpha block: 16 x 4-bit values packed little-endian ---
    alpha_bits = 0
    for i, p in enumerate(pixels):
        a = int(p[3])
        # 0..255 -> 0..15 with rounding
        a4 = (a * 15 + 127) // 255
        alpha_bits |= (a4 & 0xF) << (4 * i)
    alpha_bytes = alpha_bits.to_bytes(8, "little")

    # --- Color block: identical layout to DXT1 (8 bytes) ---
    color_bytes = _compress_dxt1_block(pixels)
    return alpha_bytes + color_bytes


def _compress_image_to_dxt3(img: Image.Image) -> bytes:
    """
    Compress a full RGBA image into DXT3 blocks.
    """
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4

    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt3_block(block))
    return bytes(out)


def build_dds_dxt5_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    """
    Build a DXT5-compressed DDS as bytes with optional mipmaps.
    """
    base = img.convert("RGBA")
    width, height = base.size

    if mipmaps:
        levels = _generate_mipmaps(base)
        mip_count = len(levels)
    else:
        levels = [base]
        mip_count = 0

    DDS_MAGIC = b"DDS "

    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000

    DDPF_FOURCC = 0x4

    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    # Top-level linear size for DXT5.
    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 16

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = top_linear
    dwDepth = 0
    dwMipMapCount = mip_count
    dwReserved1 = (0,) * 11

    pfSize = 32
    pfFlags = DDPF_FOURCC
    pfFourCC = struct.unpack("<I", b"DXT5")[0]
    pfRGBBitCount = 0
    pfRBitMask = 0
    pfGBitMask = 0
    pfBBitMask = 0
    pfABitMask = 0

    dwCaps = DDSCAPS_TEXTURE
    if mipmaps:
        dwCaps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        pfSize,
        pfFlags,
        pfFourCC,
        pfRGBBitCount,
        pfRBitMask,
        pfGBitMask,
        pfBBitMask,
        pfABitMask,
        dwCaps,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    if len(header) != 124:
        raise RuntimeError(f"Internal error: DDS header length is {len(header)}, expected 124.")

    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt5(level))
    return b"".join(chunks)


def build_dds_dxt3_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    """
    Build a DXT3-compressed DDS as bytes with optional mipmaps.
    """
    base = img.convert("RGBA")
    width, height = base.size

    if mipmaps:
        levels = _generate_mipmaps(base)
        mip_count = len(levels)
    else:
        levels = [base]
        mip_count = 0

    DDS_MAGIC = b"DDS "

    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000

    DDPF_FOURCC = 0x4

    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 16

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = top_linear
    dwDepth = 0
    dwMipMapCount = mip_count
    dwReserved1 = (0,) * 11

    pfSize = 32
    pfFlags = DDPF_FOURCC
    pfFourCC = struct.unpack("<I", b"DXT3")[0]
    pfRGBBitCount = 0
    pfRBitMask = 0
    pfGBitMask = 0
    pfBBitMask = 0
    pfABitMask = 0

    dwCaps = DDSCAPS_TEXTURE
    if mipmaps:
        dwCaps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        pfSize,
        pfFlags,
        pfFourCC,
        pfRGBBitCount,
        pfRBitMask,
        pfGBitMask,
        pfBBitMask,
        pfABitMask,
        dwCaps,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    if len(header) != 124:
        raise RuntimeError(f"Internal error: DDS header length is {len(header)}, expected 124.")

    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt3(level))
    return b"".join(chunks)


def save_dds_dxt5(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(build_dds_dxt5_bytes(img, mipmaps=mipmaps))


def _compress_dxt1_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    """
    Compress a single 4x4 RGBA block to DXT1 (8 bytes).
    """
    if len(pixels) != 16:
        raise ValueError("DXT1 block must have exactly 16 pixels.")

    # Pick endpoints based on luminance extremes (fast, decent).
    best_min = pixels[0]
    best_max = pixels[0]
    lum_min = 77 * best_min[0] + 150 * best_min[1] + 29 * best_min[2]
    lum_max = lum_min
    for p in pixels[1:]:
        lum = 77 * p[0] + 150 * p[1] + 29 * p[2]
        if lum < lum_min:
            lum_min = lum
            best_min = p
        elif lum > lum_max:
            lum_max = lum
            best_max = p

    c0 = _rgb_to_565(best_max[0], best_max[1], best_max[2])
    c1 = _rgb_to_565(best_min[0], best_min[1], best_min[2])

    if c0 == c1:
        c1 = c0 - 1 if c0 > 0 else 1

    # Force 4-color mode (no transparency).
    if c0 < c1:
        c0, c1 = c1, c0

    r0, g0, b0 = _rgb565_to_rgb888(c0)
    r1, g1, b1 = _rgb565_to_rgb888(c1)
    palette = [
        (r0, g0, b0),
        (r1, g1, b1),
        ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3),
        ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3),
    ]

    bits = 0
    for i, p in enumerate(pixels):
        pr, pg, pb = p[0], p[1], p[2]
        best_idx = 0
        best_err = 10**18
        for idx, (cr, cg, cb) in enumerate(palette):
            dr = pr - cr
            dg = pg - cg
            db = pb - cb
            err = dr * dr + dg * dg + db * db
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        bits |= (best_idx & 0x3) << (2 * i)

    return struct.pack("<HHI", c0, c1, bits)


def _compress_image_to_dxt1(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4

    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt1_block(block))
    return bytes(out)


def build_dds_dxt1_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    """
    Build a DXT1-compressed DDS as bytes with optional mipmaps.
    """
    base = img.convert("RGBA")
    width, height = base.size

    if mipmaps:
        levels = _generate_mipmaps(base)
        mip_count = len(levels)
    else:
        levels = [base]
        mip_count = 0

    DDS_MAGIC = b"DDS "

    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000

    DDPF_FOURCC = 0x4

    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 8

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = top_linear
    dwDepth = 0
    dwMipMapCount = mip_count
    dwReserved1 = (0,) * 11

    pfSize = 32
    pfFlags = DDPF_FOURCC
    pfFourCC = struct.unpack("<I", b"DXT1")[0]
    pfRGBBitCount = 0
    pfRBitMask = 0
    pfGBitMask = 0
    pfBBitMask = 0
    pfABitMask = 0

    dwCaps = DDSCAPS_TEXTURE
    if mipmaps:
        dwCaps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        pfSize,
        pfFlags,
        pfFourCC,
        pfRGBBitCount,
        pfRBitMask,
        pfGBitMask,
        pfBBitMask,
        pfABitMask,
        dwCaps,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    if len(header) != 124:
        raise RuntimeError(f"Internal error: DDS header length is {len(header)}, expected 124.")

    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt1(level))
    return b"".join(chunks)


def save_dds_dxt1(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(build_dds_dxt1_bytes(img, mipmaps=mipmaps))


def _read_dds_dimensions_from_bytes(buf: bytes) -> Tuple[int, int]:
    """
    Read DDS width/height from the first 128 bytes (magic + header).
    """
    if len(buf) < 128:
        raise ValueError("DDS buffer too small to contain header.")
    if buf[:4] != b"DDS ":
        raise ValueError("Not a DDS file (missing magic).")
    hdr = buf[4 : 4 + 124]
    dwHeight = struct.unpack("<I", hdr[8:12])[0]
    dwWidth = struct.unpack("<I", hdr[12:16])[0]
    return (dwWidth, dwHeight)


def _read_dds_fourcc_from_bytes(buf: bytes) -> Optional[str]:
    """
    Return FourCC (e.g. 'DXT5') if DDS uses a FourCC pixel format, else None.
    """
    if len(buf) < 128 or buf[:4] != b"DDS ":
        return None
    hdr = buf[4 : 4 + 124]
    pf = hdr[72 : 72 + 32]
    pfFlags = struct.unpack("<I", pf[4:8])[0]
    fourCC = pf[8:12]
    if (pfFlags & 0x4) == 0:
        return None
    try:
        return fourCC.decode("ascii", errors="replace")
    except Exception:
        return None


def save_dds_rgba8(
    out_path: Path,
    img: Image.Image,
    *,
    mipmaps: bool = True,
) -> None:
    """
    Write an uncompressed DDS (RGBA8) with optional mipmaps.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dds_bytes = build_dds_rgba8_bytes(img, mipmaps=mipmaps)
    with open(out_path, "wb") as f:
        f.write(dds_bytes)


def _apply_prelight(base: Image.Image, prelight: Image.Image, *, strength: float) -> Image.Image:
    """
    Apply prelight as a multiply on RGB, preserving base alpha.
    strength: 0..1
    """
    strength = max(0.0, min(1.0, strength))

    base = base.convert("RGBA")
    prelight = prelight.convert("RGBA").resize(base.size, Image.Resampling.LANCZOS)

    base_rgb = base.convert("RGB")
    # Prelight is typically a baked shading/lightmap; treat it as grayscale to avoid accidental color tint.
    pre_a = prelight.getchannel("A")
    pre_l = prelight.convert("L")
    pre_rgb = Image.merge("RGB", (pre_l, pre_l, pre_l))
    mul = ImageChops.multiply(base_rgb, pre_rgb)

    if strength <= 0.001:
        out_rgb = base_rgb
    else:
        # If the prelight has alpha (e.g., baked maps saved as TGA), use it as a mask to limit where
        # the shading applies (prevents darkening unused UV/background areas).
        try:
            a_ext = pre_a.getextrema()
        except Exception:
            a_ext = (255, 255)

        if a_ext != (255, 255):
            # Per-pixel mask: alpha * strength.
            m = pre_a.point(lambda p: int((p * strength)))
            out_rgb = Image.composite(mul, base_rgb, m)
        else:
            # No useful alpha: uniform strength blend.
            if strength >= 0.999:
                out_rgb = mul
            else:
                out_rgb = Image.blend(base_rgb, mul, strength)

    r, g, b = out_rgb.split()
    a = base.split()[3]
    return Image.merge("RGBA", (r, g, b, a))


def _draw_diagonal_stripes(img: Image.Image, *, stripe_color: Tuple[int, int, int, int], stripe_width: int) -> None:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Draw a set of diagonal stripes (top-left to bottom-right).
    # We draw lines slightly beyond bounds to avoid gaps.
    step = max(8, stripe_width * 3)
    for offset in range(-h, w + h, step):
        draw.line((offset, 0, offset + h, h), fill=stripe_color, width=stripe_width)


def generate_demo_skin(
    *,
    size: int,
    name: str,
    base_color: Tuple[int, int, int, int],
    accent_color: Tuple[int, int, int, int],
    stripe_color: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Generates a simple procedural livery (NOT a UV-accurate livery by itself).

    For real TMNF skins, you generally paint using a Stadium UV template. This demo is useful
    as a pipeline proof (DDS + alpha + mipmaps).
    """
    img = Image.new("RGBA", (size, size), base_color)

    # Add a subtle vertical gradient.
    grad = Image.new("L", (1, size))
    for y in range(size):
        # 0..255
        v = int(255 * (0.25 + 0.75 * (y / max(1, size - 1))))
        grad.putpixel((0, y), v)
    grad = grad.resize((size, size))

    tint = Image.new("RGBA", (size, size), accent_color)
    tint.putalpha(grad)
    img = Image.alpha_composite(img, tint)

    # Stripes
    stripe_width = max(8, size // 64)
    _draw_diagonal_stripes(img, stripe_color=stripe_color, stripe_width=stripe_width)

    # Title text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, size // 18))
    except Exception:
        font = ImageFont.load_default()

    text = name
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(12, size // 48)
    x = pad
    y = size - text_h - pad
    # Shadow
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 160))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 220))

    return img


def _try_load_font(point_size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    if FONT_OVERRIDE_PATH:
        try:
            return ImageFont.truetype(FONT_OVERRIDE_PATH, point_size)
        except Exception:
            # Fall back to bundled fonts.
            pass
    candidates = []
    # macOS: prefer a very wide Unicode font when available (handles Cyrillic/Japanese/etc for player nicks).
    # We only add these if the files exist to keep this portable.
    mac_paths = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for p in mac_paths:
        try:
            if Path(p).exists():
                candidates.append(p)
        except Exception:
            continue
    if bold:
        candidates.extend(["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"])
    else:
        candidates.extend(["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, point_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _composite_logo_with_glow(
    base: Image.Image,
    *,
    logo: Image.Image,
    xy: Tuple[int, int],
    size_px: int,
    glow_color: Tuple[int, int, int, int],
    glow_radius: int,
) -> None:
    """
    Alpha-composite a resized logo + a soft glow onto base in-place.
    """
    base_rgba = base.convert("RGBA")
    logo_rgba = logo.convert("RGBA")

    w, h = logo_rgba.size
    if w == 0 or h == 0:
        return

    # Keep aspect ratio.
    scale = size_px / max(w, h)
    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    logo_resized = logo_rgba.resize(target, Image.Resampling.LANCZOS)

    # Glow uses logo alpha as mask, blurred.
    alpha = logo_resized.getchannel("A")
    glow = Image.new("RGBA", logo_resized.size, (glow_color[0], glow_color[1], glow_color[2], 0))
    glow.putalpha(alpha)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    # Boost glow alpha a bit.
    glow_alpha = glow.getchannel("A").point(lambda a: min(255, int(a * 1.35)))
    glow.putalpha(glow_alpha)

    base_rgba.alpha_composite(glow, xy)
    base_rgba.alpha_composite(logo_resized, xy)

    # Write back
    base.paste(base_rgba)


def _composite_logo_decal(
    base: Image.Image,
    *,
    logo: Image.Image,
    xy: Tuple[int, int],
    size_px: int,
    glow_color: Tuple[int, int, int, int],
    glow_radius: int,
    outline_color: Tuple[int, int, int, int],
    outline_px: int = 2,
    rotate_deg: float = 0.0,
    anchor: str = "topleft",
    enable_glow: bool = True,
    enable_outline: bool = True,
) -> None:
    """
    Logo composite with glow + crisp outline (more readable on busy backgrounds).
    """
    base_rgba = base.convert("RGBA")
    logo_rgba = logo.convert("RGBA")
    w, h = logo_rgba.size
    if w == 0 or h == 0:
        return

    # If we rotate with expand=True, the bounding box grows; treat size_px as the FINAL max dimension
    # after rotation, so the logo doesn't become unexpectedly huge/clip on small UV islands.
    rdeg = float(rotate_deg)
    if abs(rdeg) > 0.01:
        th = math.radians(rdeg)
        c = abs(math.cos(th))
        s = abs(math.sin(th))
        denom = max(w * c + h * s, w * s + h * c)
        scale = (size_px / denom) if denom > 1e-6 else (size_px / max(w, h))
    else:
        scale = size_px / max(w, h)

    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    logo_resized = logo_rgba.resize(target, Image.Resampling.LANCZOS)
    if abs(rdeg) > 0.01:
        logo_resized = logo_resized.rotate(rdeg, expand=True, resample=Image.Resampling.BICUBIC)
    if anchor == "center":
        xy = (int(xy[0] - logo_resized.size[0] // 2), int(xy[1] - logo_resized.size[1] // 2))
    alpha = logo_resized.getchannel("A")

    if enable_glow:
        # Glow uses alpha as mask, blurred.
        glow = Image.new("RGBA", logo_resized.size, (glow_color[0], glow_color[1], glow_color[2], 0))
        glow.putalpha(alpha)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(0, int(glow_radius))))
        glow_alpha = glow.getchannel("A").point(lambda a: min(255, int(a * 1.55)))
        glow.putalpha(glow_alpha)
        base_rgba.alpha_composite(glow, xy)

    if enable_outline and outline_px > 0 and outline_color[3] > 0:
        # Outline: dilate alpha and subtract original alpha.
        k = max(3, outline_px * 2 + 1)
        dil = alpha.filter(ImageFilter.MaxFilter(size=k))
        outline_a = ImageChops.subtract(dil, alpha)
        outline = Image.new("RGBA", logo_resized.size, (outline_color[0], outline_color[1], outline_color[2], 0))
        outline.putalpha(outline_a.point(lambda p: min(255, int(p * (outline_color[3] / 255.0)))))
        base_rgba.alpha_composite(outline, xy)

    base_rgba.alpha_composite(logo_resized, xy)
    base.paste(base_rgba)


def _make_carbon_fiber_overlay(size: int, *, color: Tuple[int, int, int, int]) -> Image.Image:
    """
    Simple carbon-fiber weave illusion using a tiled 64x64 pattern.
    """
    tile = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    # Two alternating diagonal "weave" bands.
    for i in range(-64, 128, 16):
        d.line((i, 0, i + 64, 64), fill=(255, 255, 255, 20), width=10)
        d.line((i, 64, i + 64, 0), fill=(255, 255, 255, 14), width=6)
    # Tint the tile
    tint = Image.new("RGBA", tile.size, color)
    overlay = Image.alpha_composite(tint, tile)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for y in range(0, size, 64):
        for x in range(0, size, 64):
            out.alpha_composite(overlay, (x, y))
    return out


def _make_diagonal_band_layer(
    size: int,
    *,
    color: Tuple[int, int, int, int],
    highlight_color: Tuple[int, int, int, int],
    band_width: float = 0.32,
    angle_deg: float = -18.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = -0.08,
) -> Image.Image:
    """
    A large diagonal "swoosh" band to break up flat patterns (more aesthetic than stripes).
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Create a wide rectangle and rotate it.
    bw = int(size * (0.55 + band_width))
    bh = int(size * band_width)
    band = Image.new("RGBA", (bw, bh), color)

    # Add a highlight edge.
    hd = ImageDraw.Draw(band)
    edge_h = max(2, bh // 10)
    hd.rectangle((0, 0, bw, edge_h), fill=highlight_color)
    hd.rectangle((0, bh - edge_h, bw, bh), fill=(highlight_color[0], highlight_color[1], highlight_color[2], max(0, highlight_color[3] - 70)))

    band = band.filter(ImageFilter.GaussianBlur(radius=max(2, size // 420)))
    band = band.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)

    # Center, with optional offsets to vary composition.
    x = (size - band.size[0]) // 2 + int(size * offset_x_frac)
    y = (size - band.size[1]) // 2 + int(size * offset_y_frac)
    layer.alpha_composite(band, (x, y))
    return layer


def _recolor_teal_glow_like_tron(img: Image.Image, *, target_rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Recolor "Tron teal" glow pixels (high G/B, lower R) toward a target RGB, preserving intensity.
    Uses numpy if available; otherwise returns original.
    """
    if np is None:
        return img
    arr = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    a = arr[..., 3].astype(np.int16)

    # Heuristic "teal glow" mask: bright-ish, saturated, teal-ish.
    teal = (a > 10) & (g > 80) & (b > 80) & (r < 170) & ((g + b) > (r + 110)) & ((g - r) > 20) & ((b - r) > 20)

    if not np.any(teal):
        return img

    intensity = np.maximum.reduce([r, g, b]).astype(np.int16)  # 0..255
    tr, tg, tb = target_rgb
    arr[..., 0][teal] = np.clip((tr * intensity[teal]) // 255, 0, 255).astype(np.uint8)
    arr[..., 1][teal] = np.clip((tg * intensity[teal]) // 255, 0, 255).astype(np.uint8)
    arr[..., 2][teal] = np.clip((tb * intensity[teal]) // 255, 0, 255).astype(np.uint8)
    # Pillow 11 warns that 'mode=' will be removed; let it infer from array shape.
    return Image.fromarray(arr)


def _recolor_non_gray_accents(img: Image.Image, *, target_rgb: Tuple[int, int, int], strength: float = 1.0) -> Image.Image:
    """
    Recolor pixels that are noticeably non-gray (chromatic accents) toward target_rgb.
    Useful for matching wheel rings / colored details in packs like Deep Galaxy without tinting the whole texture.
    """
    if np is None:
        return img
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.001:
        return img

    arr = np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    a = arr[..., 3].astype(np.int16)

    chroma = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    mask = (a > 10) & (chroma > 40)
    if not np.any(mask):
        return img

    # Preserve intensity (value) while shifting hue toward target.
    intensity = np.maximum.reduce([r, g, b]).astype(np.int16)  # 0..255
    tr, tg, tb = target_rgb

    nr = np.clip((tr * intensity) // 255, 0, 255).astype(np.int16)
    ng = np.clip((tg * intensity) // 255, 0, 255).astype(np.int16)
    nb = np.clip((tb * intensity) // 255, 0, 255).astype(np.int16)

    if strength < 0.999:
        arr[..., 0][mask] = np.clip((r[mask] * (1 - strength) + nr[mask] * strength), 0, 255).astype(np.uint8)
        arr[..., 1][mask] = np.clip((g[mask] * (1 - strength) + ng[mask] * strength), 0, 255).astype(np.uint8)
        arr[..., 2][mask] = np.clip((b[mask] * (1 - strength) + nb[mask] * strength), 0, 255).astype(np.uint8)
    else:
        arr[..., 0][mask] = nr[mask].astype(np.uint8)
        arr[..., 1][mask] = ng[mask].astype(np.uint8)
        arr[..., 2][mask] = nb[mask].astype(np.uint8)

    return Image.fromarray(arr)


def _build_uv_debug_image_from_base(
    base_diffuse: Image.Image,
    *,
    downscale_to: int = 512,
) -> Image.Image:
    """
    Generate a debug Diffuse texture where each UV island bounding box is colored and numbered.
    This is used to identify which parts of the car correspond to which texture regions.
    """
    full = base_diffuse.convert("RGBA")
    W, H = full.size

    if np is None:
        # Numpy-less fallback: simple solid color.
        img = Image.new("RGBA", (W, H), (120, 0, 180, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 20), "UV DEBUG (install + view in 3D)", fill=(255, 255, 255, 255))
        img.putalpha(255)
        return img

    ranked_map_small, comps_ranked, scale = _compute_ranked_uv_islands(full, downscale_to=downscale_to)

    img = Image.new("RGBA", (W, H), (20, 20, 22, 255))
    d = ImageDraw.Draw(img)

    # Title
    title_font = _try_load_font(max(18, W // 80), bold=True)
    d.text((int(W * 0.02), int(H * 0.02)), "UV DEBUG", font=title_font, fill=(255, 255, 255, 220))

    # Colors in HSV-ish wheel via simple trig.
    # IMPORTANT: draw per-island using the *actual island mask* (not just the bbox), and tile labels
    # for ALL islands so you can always spot the id on the car (plates/tyres are often small strips).
    max_islands = min(120, len(comps_ranked))
    for idx in range(1, max_islands + 1):
        fx0, fy0, fx1, fy1, area = comps_ranked[idx - 1]
        rw = max(1, fx1 - fx0)
        rh = max(1, fy1 - fy0)
        if rw < 16 or rh < 10:
            continue

        # Build full-res island mask for this rank id.
        ms = (ranked_map_small == idx).astype(np.uint8) * 255  # type: ignore
        m = Image.fromarray(ms).convert("L").resize((W, H), Image.Resampling.NEAREST)

        hue = (idx * 0.19) % 1.0
        v = 220
        s = 0.85
        h6 = hue * 6.0
        c = v * s
        x = c * (1 - abs((h6 % 2) - 1))
        m0 = v - c
        if 0 <= h6 < 1:
            rr, gg, bb = c, x, 0
        elif 1 <= h6 < 2:
            rr, gg, bb = x, c, 0
        elif 2 <= h6 < 3:
            rr, gg, bb = 0, c, x
        elif 3 <= h6 < 4:
            rr, gg, bb = 0, x, c
        elif 4 <= h6 < 5:
            rr, gg, bb = x, 0, c
        else:
            rr, gg, bb = c, 0, x
        col_rgb = (int(rr + m0), int(gg + m0), int(bb + m0))

        # Fill island
        fill = Image.new("RGBA", (W, H), (col_rgb[0], col_rgb[1], col_rgb[2], 0))
        fill.putalpha(m.point(lambda p: int(p * 0.55)))  # ~140 max
        img = Image.alpha_composite(img, fill)

        # Outline for readability
        try:
            edge = m.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 0 else 0)
            ol = Image.new("RGBA", (W, H), (255, 255, 255, 0))
            ol.putalpha(edge.point(lambda p: int(p * 0.55)))
            img = Image.alpha_composite(img, ol)
        except Exception:
            pass

        # Tile labels across the bbox region, clipped to the island mask.
        text = str(idx)
        fs = max(14, min(120, int(min(rw, rh) * 0.46)))
        font = _try_load_font(fs, bold=True)
        sw = max(1, fs // 9)

        tile = Image.new("RGBA", (fs * 4, fs * 3), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text(
            (fs // 2, fs // 3),
            text,
            font=font,
            fill=(0, 0, 0, 200),
            stroke_width=sw,
            stroke_fill=(255, 255, 255, 240),
        )
        tile = tile.rotate(-18, expand=True, resample=Image.Resampling.BICUBIC)
        step_x = max(28, int(tile.size[0] * 0.85))
        step_y = max(28, int(tile.size[1] * 0.85))

        patt = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
        for py in range(-step_y, rh + step_y, step_y):
            for px in range(-step_x, rw + step_x, step_x):
                patt.alpha_composite(tile, (px, py))

        label_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        label_layer.alpha_composite(patt, (fx0, fy0))
        la = label_layer.getchannel("A")
        la = ImageChops.multiply(la, m)
        label_layer.putalpha(la)
        img = Image.alpha_composite(img, label_layer)

    img.putalpha(255)
    return img


def _make_big_label_debug_image(
    *,
    size: Tuple[int, int],
    label: str,
    bg_rgb: Tuple[int, int, int],
    fg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Simple, ultra-visible debug texture: checkerboard + large label repeated.
    Useful when a part is driven by a different texture than Diffuse (e.g., Details/Dirty).
    """
    w, h = int(size[0]), int(size[1])
    w = max(32, w)
    h = max(32, h)
    img = Image.new("RGBA", (w, h), (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
    d = ImageDraw.Draw(img)

    # Checkerboard for orientation.
    step = max(24, min(w, h) // 16)
    c0 = (max(0, bg_rgb[0] - 35), max(0, bg_rgb[1] - 35), max(0, bg_rgb[2] - 35), 255)
    c1 = (min(255, bg_rgb[0] + 35), min(255, bg_rgb[1] + 35), min(255, bg_rgb[2] + 35), 255)
    for yy in range(0, h, step):
        for xx in range(0, w, step):
            d.rectangle((xx, yy, xx + step, yy + step), fill=c0 if ((xx // step + yy // step) % 2 == 0) else c1)

    text = label.replace(".dds", "").upper()
    # Big center label
    fs = max(28, min(420, int(min(w, h) * 0.22)))
    font = _try_load_font(fs, bold=True)
    sw = max(2, fs // 10)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=sw)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    cy = (h - th) // 2
    d.text((cx, cy), text, font=font, fill=(fg_rgb[0], fg_rgb[1], fg_rgb[2], 255), stroke_width=sw, stroke_fill=(0, 0, 0, 255))

    # Smaller repeats for parts that only show a tiny portion.
    sfs = max(18, min(120, int(min(w, h) * 0.10)))
    sfont = _try_load_font(sfs, bold=True)
    ssw = max(1, sfs // 10)
    sb = d.textbbox((0, 0), text, font=sfont, stroke_width=ssw)
    stw = sb[2] - sb[0]
    sth = sb[3] - sb[1]
    for yy in range(10, h, max(80, sth + 60)):
        for xx in range(10, w, max(120, stw + 80)):
            d.text((xx, yy), text, font=sfont, fill=(fg_rgb[0], fg_rgb[1], fg_rgb[2], 220), stroke_width=ssw, stroke_fill=(0, 0, 0, 235))

    return img


def _compute_ranked_uv_islands(
    base_diffuse_rgba: Image.Image,
    *,
    downscale_to: int = 512,
) -> Tuple["np.ndarray", List[Tuple[int, int, int, int, int]], int]:
    """
    Compute UV island connected components on a downscaled mask, then rank them by area (largest = 1).

    Returns:
    - ranked label map on the SMALL grid (0=background, 1..N=rank id)
    - list of ranked full-res bboxes: (x0,y0,x1,y1,area) in FULL pixels, ordered by rank id
    - scale factor (full/ small)
    """
    if np is None:
        raise RuntimeError("Numpy required for UV island computation.")

    full = base_diffuse_rgba.convert("RGBA")
    W, H = full.size

    scale = max(1, int(max(W, H) / downscale_to))
    small = full.resize((max(1, W // scale), max(1, H // scale)), Image.Resampling.NEAREST)
    rgb = np.asarray(small.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]

    corners = np.array([rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1]], dtype=np.int16)
    bg = np.median(corners, axis=0).astype(np.int16)

    sad = np.abs(rgb - bg).sum(axis=2)
    m = sad > 4

    labels = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, int, int, int, int, int]] = []  # label_id, x0,y0,x1,y1,area
    label_id = 0

    for y in range(h):
        for x in range(w):
            if not m[y, x] or visited[y, x]:
                continue
            label_id += 1
            stack = [(y, x)]
            visited[y, x] = True
            labels[y, x] = label_id
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy

                if cy > 0 and m[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    labels[cy - 1, cx] = label_id
                    stack.append((cy - 1, cx))
                if cy + 1 < h and m[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    labels[cy + 1, cx] = label_id
                    stack.append((cy + 1, cx))
                if cx > 0 and m[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    labels[cy, cx - 1] = label_id
                    stack.append((cy, cx - 1))
                if cx + 1 < w and m[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    labels[cy, cx + 1] = label_id
                    stack.append((cy, cx + 1))

            if area < 120:
                # Remove noise label
                labels[labels == label_id] = 0
                label_id -= 1
                continue
            comps.append((label_id, x0, y0, x1 + 1, y1 + 1, area))

    # Sort by area desc -> rank
    comps_sorted = sorted(comps, key=lambda t: t[5], reverse=True)
    map_arr = np.zeros((label_id + 1,), dtype=np.int32)
    comps_ranked_full: List[Tuple[int, int, int, int, int]] = []
    for rank, (lid, x0, y0, x1, y1, area) in enumerate(comps_sorted, start=1):
        map_arr[lid] = rank
        fx0, fy0, fx1, fy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        comps_ranked_full.append((int(fx0), int(fy0), int(fx1), int(fy1), int(area)))

    ranked = map_arr[labels]
    return ranked, comps_ranked_full, scale


def _guess_number_plate_island_ids(
    comps_ranked_full: List[Tuple[int, int, int, int, int]],
    *,
    tex_w: int,
    tex_h: int,
    max_count: int = 2,
) -> List[int]:
    """
    Best-effort detection of a Stadium "number plate" UV island.

    Many Stadium templates include one or two small wide rectangular islands that map to a
    license/number plate surface. We detect these by bbox aspect ratio + bbox area fraction.

    Returns rank-ids (1-based) into comps_ranked_full.
    """
    if tex_w <= 0 or tex_h <= 0:
        return []
    total = float(tex_w * tex_h)

    def _score(bb: Tuple[int, int, int, int]) -> float:
        x0, y0, x1, y1 = bb
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        ar = float(w) / float(h)
        af = float(w * h) / total
        # Typical plate-ish shapes: wide, not too large.
        if ar < 1.6 or ar > 6.2:
            return -1.0
        if af < 0.0006 or af > 0.03:
            return -1.0
        # Prefer ~3.0 aspect and around ~0.5% of texture area.
        ar_term = 1.0 - abs(ar - 3.05) / 3.05  # peak at ~3.05
        af_term = 1.0 - abs(af - 0.0055) / 0.0055
        return (max(-1.0, ar_term) * 2.2) + (max(-1.0, af_term) * 1.8)

    cands: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
    for rid, (x0, y0, x1, y1, _area) in enumerate(comps_ranked_full, start=1):
        bb = (int(x0), int(y0), int(x1), int(y1))
        s = _score(bb)
        if s > 0.0:
            cands.append((s, int(rid), bb))

    if not cands:
        return []
    cands.sort(key=lambda t: t[0], reverse=True)

    out: List[int] = []
    out_bbs: List[Tuple[int, int, int, int]] = []
    for s, rid, bb in cands:
        if len(out) >= int(max(1, max_count)):
            break
        dup = False
        for obb in out_bbs:
            if abs(bb[0] - obb[0]) <= 12 and abs(bb[1] - obb[1]) <= 12 and abs(bb[2] - obb[2]) <= 12 and abs(bb[3] - obb[3]) <= 12:
                dup = True
                break
        if dup:
            continue
        out.append(int(rid))
        out_bbs.append(bb)

    return out


def _find_details_plate_rects(
    details_rgba: Image.Image,
    *,
    downscale_to: int = 1024,
    luma_threshold: int = 160,
    max_count: int = 2,
) -> List[Tuple[int, int, int, int]]:
    """
    Heuristic: find license-plate-like rectangles in Details.dds.

    In some Stadium packs, the actual license plate surface is driven by Details.dds (often alpha=0),
    and does NOT share the same UV layout as Diffuse.dds. So we detect plate areas directly from the
    Details texture by looking for small bright-edged rectangles in the lower-right region.

    Returns full-res rects (x0,y0,x1,y1) inclusive-exclusive.
    """
    if np is None:
        return []
    try:
        thr = int(luma_threshold)
    except Exception:
        thr = 160
    thr = max(60, min(240, thr))

    rgb = details_rgba.convert("RGB")
    W, H = rgb.size
    if W < 64 or H < 64:
        return []

    # Downscale for speed; keep edges (NEAREST).
    scale = max(1, int(max(W, H) / max(128, int(downscale_to))))
    sw = max(1, W // scale)
    sh = max(1, H // scale)
    small = rgb.resize((sw, sh), Image.Resampling.NEAREST)
    arr = np.asarray(small, dtype=np.uint8)
    # Luma (float) and threshold for bright edges.
    luma = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    mask = luma > float(thr)
    if not np.any(mask):
        return []

    visited = np.zeros(mask.shape, dtype=bool)
    comps: List[Tuple[int, int, int, int, int]] = []  # area, x0,y0,x1,y1

    # Sparse BFS: only iterate candidate pixels in each row.
    for y in range(sh):
        xs = np.flatnonzero(mask[y] & (~visited[y]))
        for x in xs.tolist():
            if visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < x0:
                    x0 = cx
                if cx > x1:
                    x1 = cx
                if cy < y0:
                    y0 = cy
                if cy > y1:
                    y1 = cy

                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < sh and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < sw and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            if area < 40:
                continue
            comps.append((area, x0, y0, x1 + 1, y1 + 1))

    if not comps:
        return []

    # Score candidates: plate-ish bbox, and prefer bottom-right region.
    cands: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for area, x0, y0, x1, y1 in comps:
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        ar = bw / float(bh)
        # plate-ish proportions
        if ar < 1.7 or ar > 8.0:
            continue
        if bw < 60 or bh < 12 or bw > 520 or bh > 220:
            continue
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        # Strong bias to the lower-right quadrant where plates live in common Details layouts.
        if cx < (sw * 0.50) or cy < (sh * 0.45):
            continue
        pos = (cx / max(1.0, float(sw))) * 0.8 + (cy / max(1.0, float(sh))) * 1.2
        ar_pref = 1.0 - min(1.0, abs(ar - 3.2) / 3.2)
        score = pos * 10.0 + ar_pref * 2.5 + min(2.0, area / 5000.0)
        cands.append((score, (x0, y0, x1, y1)))

    if not cands:
        return []
    cands.sort(key=lambda t: t[0], reverse=True)

    out: List[Tuple[int, int, int, int]] = []
    for score, (x0, y0, x1, y1) in cands:
        if len(out) >= int(max(1, max_count)):
            break
        # Scale to full-res and pad a bit.
        fx0 = int(max(0, (x0 * scale) - 2))
        fy0 = int(max(0, (y0 * scale) - 2))
        fx1 = int(min(W, (x1 * scale) + 2))
        fy1 = int(min(H, (y1 * scale) + 2))
        bb = (fx0, fy0, fx1, fy1)
        # Deduplicate near-identical rects.
        dup = False
        for ob in out:
            if abs(bb[0] - ob[0]) <= 20 and abs(bb[1] - ob[1]) <= 20 and abs(bb[2] - ob[2]) <= 20 and abs(bb[3] - ob[3]) <= 20:
                dup = True
                break
        if not dup:
            out.append(bb)

    return out


def _draw_text_block(
    img: Image.Image,
    *,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int, int],
    stroke_width: int = 0,
    stroke_fill: Tuple[int, int, int, int] = (0, 0, 0, 200),
    shadow: bool = True,
) -> None:
    draw = ImageDraw.Draw(img)
    x, y = xy
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 140), stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _mask_used_pixels_from_template(template_rgba: Image.Image) -> "np.ndarray | None":
    """
    Attempt to infer which pixels are "used" UV islands vs unused background, based on a template image.

    Returns a boolean numpy mask (H,W) or None if numpy isn't available.
    """
    if np is None:
        return None
    rgb = np.asarray(template_rgba.convert("RGB"), dtype=np.uint8)
    # Unused areas in many TM templates are near-black. Keep a small threshold to include dark-but-used pixels.
    s = rgb[:, :, 0].astype(np.int16) + rgb[:, :, 1].astype(np.int16) + rgb[:, :, 2].astype(np.int16)
    return s > 18


def _find_long_rectangles_from_mask(mask: "np.ndarray", *, min_width: int, min_height: int, max_height: int) -> List[Tuple[int, int, int, int]]:
    """
    Find axis-aligned rectangles that look like long UV islands (e.g., wings) by run-length clustering.
    Returns list of (x0, y0, x1, y1) inclusive-exclusive.
    """
    h, w = mask.shape
    runs: List[Tuple[int, int, int]] = []
    for y in range(h):
        row = mask[y]
        # Find transitions
        diff = np.diff(row.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        if starts.size == 0:
            continue
        lengths = ends - starts
        # Keep all runs above threshold, not just the longest.
        good = np.flatnonzero(lengths >= min_width)
        for gi in good.tolist():
            x0 = int(starts[gi])
            x1 = int(ends[gi])
            runs.append((y, x0, x1))

    if not runs:
        return []

    runs.sort(key=lambda t: (t[1], t[2], t[0]))
    rects: List[Tuple[int, int, int, int]] = []
    # Cluster by similar x0/x1 across consecutive y.
    tol = 6
    i = 0
    while i < len(runs):
        y0, x0, x1 = runs[i]
        y1 = y0 + 1
        j = i + 1
        while j < len(runs):
            yj, x0j, x1j = runs[j]
            if yj != y1:
                break
            if abs(x0j - x0) <= tol and abs(x1j - x1) <= tol:
                x0 = int(round((x0 + x0j) / 2))
                x1 = int(round((x1 + x1j) / 2))
                y1 += 1
                j += 1
            else:
                break
        height = y1 - y0
        width = x1 - x0
        if min_height <= height <= max_height and width >= min_width and (width / max(1, height)) >= 3.0:
            rects.append((x0, y0, x1, y1))
        i = max(i + 1, j)

    # Deduplicate near-identical rects
    out: List[Tuple[int, int, int, int]] = []
    for r in sorted(rects, key=lambda t: (-(t[2] - t[0]) * (t[3] - t[1]), t[1], t[0])):
        x0, y0, x1, y1 = r
        duplicate = False
        for ox0, oy0, ox1, oy1 in out[:10]:
            if abs(x0 - ox0) <= 10 and abs(y0 - oy0) <= 10 and abs(x1 - ox1) <= 10 and abs(y1 - oy1) <= 10:
                duplicate = True
                break
        if not duplicate:
            out.append(r)
        if len(out) >= 8:
            break
    return out


def _infer_stadium_feature_rects_from_base_diffuse(base_diffuse_rgba: Image.Image) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Heuristically infer useful UV rectangles (like wings) from a base Diffuse.dds template.

    This is best-effort; if it fails, we fall back to a few hardcoded safe regions.
    """
    w, h = base_diffuse_rgba.size
    out: Dict[str, List[Tuple[int, int, int, int]]] = {"long_rects": []}
    mask = _mask_used_pixels_from_template(base_diffuse_rgba)
    if mask is not None:
        rects = _find_long_rectangles_from_mask(mask, min_width=max(260, w // 6), min_height=max(18, h // 120), max_height=max(70, h // 18))
        out["long_rects"] = rects

    if not out["long_rects"]:
        # Fallback rectangles (scaled), meant to hit a couple of long UV islands on many Stadium templates.
        out["long_rects"] = [
            (int(w * 0.08), int(h * 0.06), int(w * 0.62), int(h * 0.12)),
            (int(w * 0.38), int(h * 0.88), int(w * 0.92), int(h * 0.95)),
            (int(w * 0.70), int(h * 0.42), int(w * 0.98), int(h * 0.48)),
        ]
    return out


def _scale_rects(rects: List[Tuple[int, int, int, int]], *, sx: float, sy: float) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in rects:
        out.append((int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))))
    return out


def _standard_stadium_rear_wing_rects(diffuse_w: int, diffuse_h: int) -> List[Tuple[int, int, int, int]]:
    """
    Rear-wing UV island rect for the standard Stadium model (observed in Kacky/Pink packs).
    Base reference is 2048x2048: (224, 4) -> (780, 208)
    """
    x0 = int(round(224 * (diffuse_w / 2048.0)))
    y0 = int(round(4 * (diffuse_h / 2048.0)))
    x1 = int(round(780 * (diffuse_w / 2048.0)))
    y1 = int(round(208 * (diffuse_h / 2048.0)))
    return [(x0, y0, x1, y1)]


def _standard_stadium_mudguard_rects(diffuse_w: int, diffuse_h: int) -> List[Tuple[int, int, int, int]]:
    """
    Mudguard/wheel-arch UV rects for the standard Stadium model.
    Base reference is 2048x2048; scales to actual diffuse size.
    """
    sx = diffuse_w / 2048.0
    sy = diffuse_h / 2048.0
    return _scale_rects(STADIUM_MUDGUARD_RECTS_2048, sx=sx, sy=sy)


def _detect_mudguard_island_ids(
    comps_ranked_full: List[Tuple[int, int, int, int, int]],
    *,
    tex_size: int = 2048,
    max_pairs: int = 2,
) -> List[int]:
    """
    Automatically detect mudguard/wheel-arch UV islands using geometric heuristics.

    Algorithm:
    1. Find mirrored pairs (Y coords sum to ~2160 for 2048 texture)
    2. Filter for square-ish aspect ratio (0.7 < AR < 1.5) - wheel arches are roughly square
    3. Filter for appropriate size range (not too big, not too small)
    4. Return best candidates (2 pairs = 4 islands)

    Args:
        comps_ranked_full: List of (x0, y0, x1, y1, area) tuples, ranked by area (largest=1)
        tex_size: Texture size (default 2048)
        max_pairs: Maximum number of mirrored pairs to return (default 2 = 4 islands)

    Returns:
        List of rank IDs (1-based) for detected mudguard islands
    """
    if not comps_ranked_full:
        return list(STADIUM_MUDGUARD_ISLAND_IDS)  # Fallback to hardcoded

    # Find mirrored pairs
    pairs: List[Tuple[float, Tuple[int, int], float]] = []  # (score, (id1, id2), ar)
    used: set = set()
    y_mirror_sum = tex_size * 1.0547  # ~2160 for 2048

    for rid_a, (ax0, ay0, ax1, ay1, a_area) in enumerate(comps_ranked_full, start=1):
        if rid_a in used:
            continue
        a_cy = (ay0 + ay1) / 2.0

        for rid_b, (bx0, by0, bx1, by1, b_area) in enumerate(comps_ranked_full, start=1):
            if rid_a == rid_b or rid_b in used:
                continue
            b_cy = (by0 + by1) / 2.0

            # Check if mirrored pair
            x_match = abs(ax0 - bx0) < 20 and abs(ax1 - bx1) < 20
            area_match = abs(a_area - b_area) < 100
            y_sum = a_cy + b_cy
            y_mirror = abs(y_sum - y_mirror_sum) < 40

            if x_match and area_match and y_mirror:
                w = ax1 - ax0
                h = ay1 - ay0
                ar = w / h if h > 0 else 999.0
                avg_area = (a_area + b_area) / 2.0

                # Filter for mudguard-like characteristics
                # - Aspect ratio 0.7-1.5 (square-ish, wheel arches are roughly square)
                # - Area 300-1500 (not too big, not too small)
                if 0.7 < ar < 1.5 and 300 < avg_area < 1500:
                    score = 1.0 - abs(ar - 1.0)  # Prefer AR closer to 1.0
                    pairs.append((score, (rid_a, rid_b), ar))

                used.add(rid_a)
                used.add(rid_b)
                break

    if not pairs:
        return list(STADIUM_MUDGUARD_ISLAND_IDS)  # Fallback to hardcoded

    # Sort by score (best match first) and return top pairs
    pairs.sort(key=lambda x: x[0], reverse=True)
    result_ids: List[int] = []
    for score, (id1, id2), ar in pairs[:max_pairs]:
        result_ids.extend([id1, id2])

    return result_ids if result_ids else list(STADIUM_MUDGUARD_ISLAND_IDS)


def _apply_mudguard_harmonization(
    img: Image.Image,
    *,
    island_masks: Dict[int, Image.Image],
    island_bboxes: Dict[int, Tuple[int, int, int, int]],
    mudguard_color: Tuple[int, int, int],
    strength: float = 0.85,
    feather: int = 3,
    fallback_rects: Optional[List[Tuple[int, int, int, int]]] = None,
    comps_ranked_full: Optional[List[Tuple[int, int, int, int, int]]] = None,
) -> Image.Image:
    """
    Apply cohesive color to mudguard/wheel-arch regions.

    Uses automatic detection when comps_ranked_full is provided, otherwise
    falls back to STADIUM_MUDGUARD_ISLAND_IDS, then to explicit rects.

    The blend preserves texture detail by using a luma-preserving tint:
    - Converts target color to HSV
    - Applies hue/saturation to the mudguard region
    - Blends value (lightness) partially to preserve shading
    """
    if strength <= 0.0:
        return img

    size = img.size[0]
    img = img.convert("RGBA")

    # Determine mudguard island IDs - use automatic detection if possible
    if comps_ranked_full is not None and len(comps_ranked_full) > 0:
        mudguard_ids = _detect_mudguard_island_ids(comps_ranked_full, tex_size=size)
    else:
        mudguard_ids = list(STADIUM_MUDGUARD_ISLAND_IDS)

    # Build combined mudguard mask from island masks or fallback rects.
    mask = Image.new("L", img.size, 0)
    used_islands = False

    # Try island masks first (more accurate).
    for mid in mudguard_ids:
        m = island_masks.get(mid)
        if m is not None:
            # Paste the island mask as white onto our combined mask.
            mask.paste(255, (0, 0), m.convert("L"))
            used_islands = True

    # Fallback to rects if no island masks were found.
    if not used_islands and fallback_rects:
        draw = ImageDraw.Draw(mask)
        for (x0, y0, x1, y1) in fallback_rects:
            draw.rectangle((x0, y0, x1, y1), fill=255)

    # Check if mask is empty.
    if mask.getbbox() is None:
        return img

    # Feather the mask edges for smooth blending.
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Clamp strength.
    strength = max(0.0, min(1.0, float(strength)))

    # Create the tinted layer using luma-preserving blend.
    # This keeps texture detail while shifting color.
    try:
        if np is not None:
            # Luma-preserving tint: keep original brightness, shift hue/sat.
            arr = np.array(img, dtype=np.float32)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

            # Compute original luminance.
            lum = 0.299 * r + 0.587 * g + 0.114 * b

            # Target color.
            tr, tg, tb = float(mudguard_color[0]), float(mudguard_color[1]), float(mudguard_color[2])
            target_lum = 0.299 * tr + 0.587 * tg + 0.114 * tb

            # Normalize target to create a tint that preserves relative brightness.
            if target_lum > 1.0:
                scale = lum / target_lum
                nr = np.clip(tr * scale, 0, 255)
                ng = np.clip(tg * scale, 0, 255)
                nb = np.clip(tb * scale, 0, 255)
            else:
                # Target is very dark; just use it directly scaled by original lum.
                nr = np.clip(lum * (tr / 128.0), 0, 255) if tr > 0 else lum * 0.1
                ng = np.clip(lum * (tg / 128.0), 0, 255) if tg > 0 else lum * 0.1
                nb = np.clip(lum * (tb / 128.0), 0, 255) if tb > 0 else lum * 0.1

            # Build tinted array.
            tinted = np.stack([nr, ng, nb, arr[..., 3]], axis=-1).astype(np.uint8)
            tinted_img = Image.fromarray(tinted, mode="RGBA")
        else:
            # Fallback without numpy: simple solid color overlay.
            tinted_img = Image.new("RGBA", img.size, (mudguard_color[0], mudguard_color[1], mudguard_color[2], 255))
    except Exception:
        # Fallback on any error.
        tinted_img = Image.new("RGBA", img.size, (mudguard_color[0], mudguard_color[1], mudguard_color[2], 255))

    # Blend using the mask and strength.
    # Scale mask by strength.
    if strength < 1.0:
        mask = mask.point(lambda p: int(p * strength))

    # Composite: paste tinted image using the mask.
    result = img.copy()
    result.paste(tinted_img, (0, 0), mask)

    return result


def _pick_mudguard_color(
    mode: str,
    base_rgb: Tuple[int, int, int],
    accent_rgb: Tuple[int, int, int],
    secondary_rgb: Tuple[int, int, int],
    custom_rgb: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int]:
    """
    Pick the mudguard color based on the specified mode.
    """
    mode = (mode or "darken").strip().lower()

    if mode == "custom" and custom_rgb is not None:
        return custom_rgb
    elif mode == "match_base":
        return base_rgb
    elif mode == "match_accent":
        return accent_rgb
    elif mode == "match_secondary":
        return secondary_rgb
    elif mode == "darken":
        # Darken the base color by 40%.
        factor = 0.6
        return (
            max(0, int(base_rgb[0] * factor)),
            max(0, int(base_rgb[1] * factor)),
            max(0, int(base_rgb[2] * factor)),
        )
    else:
        # Default: darken.
        factor = 0.6
        return (
            max(0, int(base_rgb[0] * factor)),
            max(0, int(base_rgb[1] * factor)),
            max(0, int(base_rgb[2] * factor)),
        )


def _draw_centered_text_in_rect(
    img: Image.Image,
    *,
    rect: Tuple[int, int, int, int],
    text: str,
    fill: Tuple[int, int, int, int],
    stroke_fill: Tuple[int, int, int, int] = (0, 0, 0, 220),
    max_font_px: int,
    rotate_degrees: int = 0,
) -> None:
    x0, y0, x1, y1 = rect
    rw = max(1, x1 - x0)
    rh = max(1, y1 - y0)
    # Render on its own layer so rotation won't clip.
    layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Fit font size to rect.
    best_font = _try_load_font(max(10, min(max_font_px, rh)), bold=True)
    best_fs = max(10, min(max_font_px, rh))
    # coarse search downwards
    for fs in range(min(max_font_px, 240), 9, -4):
        f = _try_load_font(fs, bold=True)
        bbox = draw.textbbox((0, 0), text, font=f, stroke_width=max(1, int(fs * 0.10)))
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= int(rw * 0.92) and th <= int(rh * 0.88):
            best_font = f
            best_fs = fs
            break

    # Stroke width must scale with font size (NOT rect height), otherwise small sidepod text becomes a blob.
    sw = max(1, int(best_fs * 0.10))
    bbox = draw.textbbox((0, 0), text, font=best_font, stroke_width=sw)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (rw - tw) // 2
    ty = (rh - th) // 2
    _draw_text_block(
        layer,
        xy=(tx, ty),
        text=text,
        font=best_font,
        fill=fill,
        stroke_width=sw,
        stroke_fill=stroke_fill,
        shadow=True,
    )

    if rotate_degrees % 360 != 0:
        layer = layer.rotate(rotate_degrees, expand=True, resample=Image.Resampling.BICUBIC)
        # Paste centered into rect.
        px = x0 + (rw - layer.size[0]) // 2
        py = y0 + (rh - layer.size[1]) // 2
        img.alpha_composite(layer, (px, py))
    else:
        img.alpha_composite(layer, (x0, y0))


def _render_wing_plate(
    img: Image.Image,
    *,
    wing_rect: Tuple[int, int, int, int],
    top_text: str,
    bottom_text: str,
    accent_rgb: Tuple[int, int, int],
    background_rgb: Tuple[int, int, int],
    decal: Optional[Image.Image] = None,
    decal_fit: str = "contain",  # contain | cover | stretch
    decal_scale: float = 1.0,
    decal_opacity: float = 1.0,
) -> None:
    """
    Render a clean rear-wing plate (no logo) with tag + small team name.
    The rear wing on the standard Stadium model maps to a single UV island; we render ONCE to
    avoid double-printing (which looks like the text appears twice).
    """
    x0, y0, x1, y1 = wing_rect
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    # Build plate layer
    plate = Image.new("RGBA", (w, h), (background_rgb[0], background_rgb[1], background_rgb[2], 255))
    d = ImageDraw.Draw(plate)

    # Border lines
    border = max(2, h // 18)
    d.rectangle((0, 0, w, border), fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 255))
    d.rectangle((0, h - border, w, h), fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 255))

    # Subtle diagonal sheen
    sheen = _make_diagonal_band_layer(
        max(w, h),
        color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 40),
        highlight_color=(255, 255, 255, 18),
        band_width=0.26,
        angle_deg=-14.0,
    ).crop((0, 0, w, h))
    plate = Image.alpha_composite(plate, sheen)

    # Content padding and layout
    pad_x = max(10, w // 22)
    pad_y = max(6, h // 10)
    content_rect = (pad_x, pad_y, w - pad_x, h - pad_y)

    cx0, cy0, cx1, cy1 = content_rect
    ch = max(1, cy1 - cy0)

    # Optional decal (e.g. skull) inside the plate content area.
    if decal is not None:
        try:
            dec = decal.convert("RGBA")
            tw = max(1, cx1 - cx0)
            th = max(1, cy1 - cy0)
            if tw > 0 and th > 0 and dec.size[0] > 1 and dec.size[1] > 1:
                fit = (decal_fit or "contain").strip().lower()
                if fit not in ("contain", "cover", "stretch"):
                    fit = "contain"
                scale = float(decal_scale) if decal_scale is not None else 1.0
                scale = max(0.05, min(10.0, scale))
                opacity = float(decal_opacity) if decal_opacity is not None else 1.0
                opacity = max(0.0, min(1.0, opacity))

                if fit == "stretch":
                    nw, nh = tw, th
                else:
                    sx = tw / float(max(1, dec.size[0]))
                    sy = th / float(max(1, dec.size[1]))
                    s = min(sx, sy) if fit == "contain" else max(sx, sy)
                    s *= scale
                    nw = max(1, int(round(dec.size[0] * s)))
                    nh = max(1, int(round(dec.size[1] * s)))

                dec = dec.resize((nw, nh), Image.Resampling.LANCZOS)
                if opacity < 0.999:
                    a = dec.getchannel("A").point(lambda p: int(p * opacity))
                    dec.putalpha(a)

                layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                px = (tw - nw) // 2
                py = (th - nh) // 2
                layer.alpha_composite(dec, (px, py))
                plate.alpha_composite(layer, (cx0, cy0))
        except Exception:
            pass

    has_top = bool(top_text.strip())
    has_bottom = bool(bottom_text.strip())

    if has_top and has_bottom:
        # Top text (bigger) + bottom text (smaller) stacked
        tag_h = int(ch * 0.62)
        tag_rect = (cx0, cy0, cx1, cy0 + tag_h)
        name_rect = (cx0, cy0 + tag_h, cx1, cy1)
        _draw_centered_text_in_rect(
            plate,
            rect=tag_rect,
            text=top_text,
            fill=(245, 245, 245, 245),
            stroke_fill=(0, 0, 0, 245),
            max_font_px=max(16, int((tag_rect[3] - tag_rect[1]) * 0.85)),
            rotate_degrees=0,
        )
        _draw_centered_text_in_rect(
            plate,
            rect=name_rect,
            text=bottom_text.upper(),
            fill=(230, 230, 230, 235),
            stroke_fill=(0, 0, 0, 235),
            max_font_px=max(10, int((name_rect[3] - name_rect[1]) * 0.60)),
            rotate_degrees=0,
        )
    elif has_top:
        # Only top text: center it in the full content area
        _draw_centered_text_in_rect(
            plate,
            rect=content_rect,
            text=top_text,
            fill=(245, 245, 245, 245),
            stroke_fill=(0, 0, 0, 245),
            max_font_px=max(16, int(ch * 0.70)),
            rotate_degrees=0,
        )
    elif has_bottom:
        # Only bottom text: center it
        _draw_centered_text_in_rect(
            plate,
            rect=content_rect,
            text=bottom_text.upper(),
            fill=(230, 230, 230, 235),
            stroke_fill=(0, 0, 0, 235),
            max_font_px=max(10, int(ch * 0.60)),
            rotate_degrees=0,
        )

    # Composite plate onto main image (opaque)
    img.alpha_composite(plate, (x0, y0))


def _make_topographic_lines_layer(size: int, *, color: Tuple[int, int, int, int], opacity: int) -> Image.Image:
    noise = Image.effect_noise((size, size), 28).filter(ImageFilter.GaussianBlur(radius=max(1, size // 320)))
    # Quantize -> contour steps
    step = 18
    quant = noise.point(lambda p: (p // step) * step)
    edges = quant.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda p: 255 if p > 22 else 0).filter(ImageFilter.GaussianBlur(radius=1))
    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    layer.putalpha(edges.point(lambda p: int((p / 255.0) * opacity)))
    return layer


def _make_lava_cracks_layer(size: int, *, glow_color: Tuple[int, int, int, int], opacity: int) -> Image.Image:
    base = Image.effect_noise((size, size), 16).filter(ImageFilter.GaussianBlur(radius=max(2, size // 220)))
    edges = base.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda p: 255 if p > 30 else 0)
    # Thicken + glow
    edges = edges.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=2))
    layer = Image.new("RGBA", (size, size), (glow_color[0], glow_color[1], glow_color[2], 0))
    layer.putalpha(edges.point(lambda p: int((p / 255.0) * opacity)))
    return layer


def _make_fluid_sheen_layer(size: int, *, a: Tuple[int, int, int], b: Tuple[int, int, int], opacity: int) -> Image.Image:
    n = Image.effect_noise((size, size), 10).filter(ImageFilter.GaussianBlur(radius=max(6, size // 110)))
    col = ImageOps.colorize(n, black=a, white=b).convert("RGBA")
    col.putalpha(opacity)
    # Add slight directional blur for "flow"
    col = col.filter(ImageFilter.GaussianBlur(radius=max(1, size // 700)))
    return col


def _make_starfield_layer(
    size: int,
    *,
    density: float = 0.0012,
    color: Tuple[int, int, int] = (255, 255, 255),
    max_alpha: int = 120,
) -> Image.Image:
    """
    Starfield as sparse points with slight blur. Requires numpy for speed; falls back to PIL draw.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    if np is None:
        d = ImageDraw.Draw(layer)
        step = max(1, int(1 / max(1e-6, density) ** 0.5))
        for y in range(0, size, step):
            for x in range(0, size, step):
                if (x * 1315423911 + y * 2654435761) & 1023 == 0:
                    d.point((x, y), fill=(color[0], color[1], color[2], max_alpha))
        return layer.filter(ImageFilter.GaussianBlur(radius=0.6))

    n = int(size * size * density)
    if n <= 0:
        return layer

    arr = np.zeros((size, size, 4), dtype=np.uint8)
    ys = np.random.randint(0, size, size=n, dtype=np.int32)
    xs = np.random.randint(0, size, size=n, dtype=np.int32)
    br = np.random.randint(170, 256, size=n, dtype=np.int32)
    arr[ys, xs, 0] = color[0]
    arr[ys, xs, 1] = color[1]
    arr[ys, xs, 2] = color[2]
    arr[ys, xs, 3] = (br * max_alpha // 255).astype(np.uint8)

    img = Image.fromarray(arr)
    # Add a tiny blur so stars aren't single-pixel harsh in DXT.
    return img.filter(ImageFilter.GaussianBlur(radius=0.6))


def _make_galaxy_nebula_layer(
    size: int,
    *,
    dark: Tuple[int, int, int],
    bright: Tuple[int, int, int],
    opacity: int,
) -> Image.Image:
    noise = Image.effect_noise((size, size), 9).filter(ImageFilter.GaussianBlur(radius=max(6, size // 140)))
    col = ImageOps.colorize(noise, black=dark, white=bright).convert("RGBA")
    col.putalpha(opacity)
    return col


def _make_splatter_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int = 180,
    blob_scale: int = 20,
    dots: bool = True,
    dot_threshold: int = 245,
    blob_threshold: Optional[int] = None,
    slashes: bool = True,
    slash_count: int = 6,
    rng: Optional["random.Random"] = None,
) -> Image.Image:
    """
    High-contrast splatter/blobs layer (good for pink/black style).
    """
    # Deterministic-ish defaults. If rng is not provided, use a stable one so this
    # layer doesn't look identical across runs but also doesn't depend on global state.
    if rng is None:
        seed_src = f"splatter|{size}|{color}|{opacity}|{blob_scale}|{dots}|{dot_threshold}|{blob_threshold}|{slashes}|{slash_count}"
        rng = random.Random(int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16))

    try:
        blob_scale_i = int(blob_scale)
    except Exception:
        blob_scale_i = 20
    blob_scale_i = max(4, min(64, blob_scale_i))

    # Build alpha as a mask first (more control, better-looking droplets).
    a = Image.new("L", (size, size), 0)

    # 1) Big organic blobs (spray “hits”)
    # Pillow's "scale" param: lower = larger blobs.
    freq = max(4, int(round(160.0 / float(blob_scale_i))))
    # Slight per-seed variation so “splatter” doesn't look samey.
    freq = int(max(4, min(64, freq + int(rng.uniform(-2, 3)))))
    blur = float(max(1.0, size / (max(1.0, float(freq)) * 18.0)))
    n = Image.effect_noise((size, size), freq).filter(ImageFilter.GaussianBlur(radius=blur))

    if blob_threshold is not None:
        try:
            thr = int(blob_threshold)
        except Exception:
            thr = 224
    else:
        # Higher threshold -> less fill (better for “dust”), lower -> bigger blobs.
        if blob_scale_i <= 12:
            thr = 236
        elif blob_scale_i <= 18:
            thr = 228
        else:
            thr = 218
        thr += int(rng.uniform(-4, 5))
    thr = max(190, min(246, int(thr)))

    blobs = n.point(lambda p: 255 if p > thr else 0)
    # Slight dilation then soften edges
    blobs = blobs.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=max(1, size // 520)))
    a = ImageChops.lighter(a, blobs.point(lambda p: int(p * 0.82)))

    # 2) Droplet clusters (this is what makes it feel like “paint splatter”, not TV noise)
    # Keep counts moderate for performance on 2048.
    try:
        d = ImageDraw.Draw(a)
        clusters = int(rng.uniform(10, 22))
        for _ in range(clusters):
            cx = rng.uniform(-0.05, 1.05) * size
            cy = rng.uniform(-0.05, 1.05) * size
            # Cluster radius scaled by blob_scale (larger blobs -> wider scatter)
            cr = rng.uniform(0.06, 0.18) * size * (blob_scale_i / 20.0)
            drops = int(rng.uniform(18, 55))
            for _k in range(drops):
                ang = rng.random() * 6.283185307179586
                rr = rng.random() ** 0.55  # bias towards center
                x = cx + (rr * cr) * (math.cos(ang) if "math" in globals() else __import__("math").cos(ang))
                y = cy + (rr * cr) * (math.sin(ang) if "math" in globals() else __import__("math").sin(ang))
                r = rng.uniform(1.2, 6.8) * (size / 2048.0) * (1.0 + (20.0 / max(6.0, float(blob_scale_i))) * 0.15)
                # Randomize alpha per droplet; more variation reads more “natural”.
                val = int(rng.uniform(120, 255))
                d.ellipse((x - r, y - r, x + r, y + r), fill=val)
        # Crisp droplets can look harsh after DXT; soften slightly.
        a = a.filter(ImageFilter.GaussianBlur(radius=max(0.8, size / 2600.0)))
    except Exception:
        pass

    # 3) Mist/spray (very fine dots, helps “razzle” feel)
    if dots:
        # The previous approach created pixel-noise; instead draw tiny droplets + blur.
        try:
            dd = ImageDraw.Draw(a)
            # Density tuned to read without turning into grey mush.
            count = int((size * size) / 4200)
            for _ in range(count):
                x = rng.randrange(0, size)
                y = rng.randrange(0, size)
                r = rng.uniform(0.55, 1.65) * (size / 2048.0)
                val = int(rng.uniform(40, 95))
                dd.ellipse((x - r, y - r, x + r, y + r), fill=val)
            a = a.filter(ImageFilter.GaussianBlur(radius=max(0.7, size / 3000.0)))
        except Exception:
            pass

    # 4) Optional streaks (short, randomized, not evenly spaced)
    if slashes and int(slash_count) > 0:
        try:
            streak = Image.new("L", (size, size), 0)
            sd = ImageDraw.Draw(streak)
            cnt = max(1, min(14, int(slash_count)))
            for _ in range(cnt):
                x0 = rng.uniform(-0.1, 1.1) * size
                y0 = rng.uniform(-0.1, 1.1) * size
                length = rng.uniform(0.12, 0.42) * size
                ang = rng.uniform(-70.0, 70.0)
                ar = math.radians(ang) if "math" in globals() else __import__("math").radians(ang)
                dx = (math.cos(ar) if "math" in globals() else __import__("math").cos(ar)) * length
                dy = (math.sin(ar) if "math" in globals() else __import__("math").sin(ar)) * length
                w = int(rng.uniform(size / 170.0, size / 85.0))
                val = int(rng.uniform(90, 190))
                sd.line((x0, y0, x0 + dx, y0 + dy), fill=val, width=max(1, w))
            streak = streak.filter(ImageFilter.GaussianBlur(radius=max(1.1, size / 1200.0)))
            a = ImageChops.lighter(a, streak)
        except Exception:
            pass

    # Convert alpha mask into a colored RGBA layer
    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    # Clamp + scale to target opacity.
    op = max(0, min(255, int(opacity)))
    layer.putalpha(a.point(lambda p: int((p / 255.0) * op)))
    return layer


def _make_camo_layer(
    size: int,
    *,
    colors: List[Tuple[int, int, int]],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Procedural camo blobs (3-4 colors) using blurred noise + thresholds.
    Non-tiling and varies per seed.
    """
    if np is None:
        # Fallback: use PIL noise and 2-color colorize (keep blobs large).
        n = Image.effect_noise((size, size), int(rng.uniform(4, 10))).filter(ImageFilter.GaussianBlur(radius=max(2, size // 320)))
        cam = ImageOps.colorize(n, black=colors[0], white=colors[-1]).convert("RGBA")
        cam.putalpha(opacity)
        return cam

    # Use PIL noise for speed + determinism from rng by varying parameters only.
    # Lower frequency -> larger blobs; smaller blur -> crisper edges.
    n = Image.effect_noise((size, size), int(rng.uniform(4, 10))).filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.2, 5.2))))
    arr = np.asarray(n, dtype=np.uint8)

    cols = colors[:]
    if len(cols) < 3:
        cols = (cols * 3)[:3]
    cols = cols[:4]

    # Choose thresholds
    t1 = int(rng.uniform(78, 108))
    t2 = int(rng.uniform(140, 172))
    t3 = int(rng.uniform(198, 222)) if len(cols) >= 4 else 255

    out = np.zeros((size, size, 3), dtype=np.uint8)
    m0 = arr < t1
    m1 = (arr >= t1) & (arr < t2)
    m2 = (arr >= t2) & (arr < t3)
    m3 = arr >= t3
    out[m0] = np.array(cols[0], dtype=np.uint8)
    out[m1] = np.array(cols[1], dtype=np.uint8)
    out[m2] = np.array(cols[2], dtype=np.uint8)
    if len(cols) >= 4:
        out[m3] = np.array(cols[3], dtype=np.uint8)
    else:
        out[m3] = np.array(cols[2], dtype=np.uint8)

    img = Image.fromarray(out).convert("RGBA")

    # Edge accent for “printed” look
    edges = n.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 20 else 0).filter(ImageFilter.GaussianBlur(radius=1))
    edge_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    edge_layer.putalpha(edges.point(lambda p: int(p * 0.35)))
    img = Image.alpha_composite(img, edge_layer)

    img.putalpha(opacity)
    return img


def _make_halftone_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Halftone-ish dot overlay: randomized dot grid to avoid obvious repetition.
    """
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    spacing = int(rng.uniform(22, 46))
    jitter = spacing * 0.18
    for y in range(-spacing, size + spacing, spacing):
        for x in range(-spacing, size + spacing, spacing):
            jx = x + rng.uniform(-jitter, jitter)
            jy = y + rng.uniform(-jitter, jitter)
            r = rng.uniform(spacing * 0.18, spacing * 0.46)
            d.ellipse((jx - r, jy - r, jx + r, jy + r), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, spacing // 20)))

    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    layer.putalpha(mask.point(lambda p: int((p / 255.0) * opacity)))
    # Random rotation to break any remaining grid feel
    rot = float(rng.uniform(-18.0, 18.0))
    layer = layer.rotate(rot, expand=False, resample=Image.Resampling.BICUBIC)
    return layer


def _make_shards_layer(
    size: int,
    *,
    colors: List[Tuple[int, int, int]],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Polygon shards overlay (random trapezoids/triangles), good for esports/aggressive wraps.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    n = int(rng.uniform(26, 56))
    cols = colors if colors else [(255, 255, 255)]
    for _ in range(n):
        cx = rng.uniform(-0.1, 1.1) * size
        cy = rng.uniform(-0.1, 1.1) * size
        w = rng.uniform(0.10, 0.42) * size
        h = rng.uniform(0.06, 0.28) * size
        ang = rng.uniform(-80.0, 80.0)
        # Build a small quad and rotate points around center.
        pts = [
            (-w * 0.5, -h * 0.5),
            (w * 0.5, -h * 0.35),
            (w * 0.45, h * 0.5),
            (-w * 0.55, h * 0.35),
        ]
        if np:
            ar = np.deg2rad(ang)
            c = float(np.cos(ar))
            s = float(np.sin(ar))
        else:
            import math
            ar = ang * 3.14159 / 180.0
            c = math.cos(ar)
            s = math.sin(ar)
        rpts = []
        for x, y in pts:
            rx = x * c - y * s
            ry = x * s + y * c
            rpts.append((cx + rx, cy + ry))
        col = cols[int(rng.random() * len(cols))]
        a = int(rng.uniform(opacity * 0.45, opacity))
        d.polygon(rpts, fill=(col[0], col[1], col[2], a))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(1, size // 520)))
    return layer


def _make_brushed_metal_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Brushed metal grain: directional noise + blur (non-tiling).
    """
    n = Image.effect_noise((size, size), int(rng.uniform(8, 18))).filter(ImageFilter.GaussianBlur(radius=1.0))
    # Fake directional blur by rotating, blurring, rotating back.
    ang = float(rng.uniform(-22.0, 22.0))
    n2 = n.rotate(ang, expand=False, resample=Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.0, 4.5))))
    n2 = n2.rotate(-ang, expand=False, resample=Image.Resampling.BICUBIC)
    col = ImageOps.colorize(n2, black=(0, 0, 0), white=color).convert("RGBA")
    col.putalpha(opacity)
    return col


def _roughen_mask_edges(mask: Image.Image, *, rng: "random.Random", strength: float = 0.55) -> Image.Image:
    """
    Add organic jaggedness to a binary-ish L mask edge.
    Keeps large shapes but breaks perfect straight lines so it reads hand-made/taped.
    """
    try:
        m = mask.convert("L")
        # Find edges, then modulate them with noise.
        edges = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.2))
        n = Image.effect_noise(m.size, int(rng.uniform(10, 22))).filter(ImageFilter.GaussianBlur(radius=1.1))
        # Scale noise to [-s..s] around 128
        s = max(0.05, min(1.0, float(strength)))
        n2 = n.point(lambda p: int(max(0, min(255, 128 + (p - 128) * (0.85 * s)))))
        # Darken/brighten edges slightly according to noise.
        mod = ImageChops.multiply(edges, n2)
        out = ImageChops.lighter(m, mod.point(lambda p: int(p * 0.55 * s)))
        # Slight soften to avoid crunchy DXT artifacts.
        return out.filter(ImageFilter.GaussianBlur(radius=0.8))
    except Exception:
        return mask.convert("L")


def _make_tape_block_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Big masked/tape blocks (diagonal polygons) with roughened edges.
    This is a core “handcrafted wrap” signal: intentional blocks, imperfect edges.
    """
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)

    blocks = int(rng.uniform(2, 4))
    for _ in range(blocks):
        # Large diagonal quad
        cx = rng.uniform(0.15, 0.85) * size
        cy = rng.uniform(0.15, 0.85) * size
        w = rng.uniform(0.55, 0.95) * size
        h = rng.uniform(0.16, 0.34) * size
        ang = rng.uniform(-60.0, 60.0)
        ar = math.radians(ang)
        c = math.cos(ar)
        s = math.sin(ar)
        pts = [(-w * 0.5, -h * 0.5), (w * 0.5, -h * 0.45), (w * 0.48, h * 0.5), (-w * 0.52, h * 0.45)]
        rpts = []
        for x, y in pts:
            rx = x * c - y * s
            ry = x * s + y * c
            rpts.append((cx + rx, cy + ry))
        d.polygon(rpts, fill=255)

    m = _roughen_mask_edges(m, rng=rng, strength=0.70)
    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    layer.putalpha(m.point(lambda p: int((p / 255.0) * int(opacity))))
    return layer


def _make_rimlight_from_mask(
    *,
    mask_l: Image.Image,
    color: Tuple[int, int, int],
    opacity: int,
    blur: float = 5.0,
) -> Image.Image:
    """
    “Bottom-light / rimlight” style glow derived from a used-mask edge.
    Reads like premium lighting accents without relying on decals/logos.
    """
    m = mask_l.convert("L")
    edges = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=max(1.0, float(blur))))
    layer = Image.new("RGBA", m.size, (color[0], color[1], color[2], 0))
    op = max(0, min(255, int(opacity)))
    layer.putalpha(edges.point(lambda p: int((p / 255.0) * op)))
    return layer


def _make_micro_hatch_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Very fine hatch/weave texture. Reads as “crafted print” and sharpens surfaces after DXT.
    """
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    step = int(rng.uniform(18, 34))
    w = max(1, int(rng.uniform(1, 2)))
    for y in range(-size, size * 2, step):
        d.line((0, y, size, y + size), fill=255, width=w)
    m = m.filter(ImageFilter.GaussianBlur(radius=0.7))
    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    op = max(0, min(255, int(opacity)))
    layer.putalpha(m.point(lambda p: int((p / 255.0) * op)))
    layer = layer.rotate(float(rng.uniform(-8.0, 8.0)), expand=False, resample=Image.Resampling.BICUBIC)
    return layer


def _make_inkblot_layer(
    size: int,
    *,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Symmetric inkblot / rorschach texture (handmade signal).
    """
    base = Image.effect_noise((max(1, size // 2), size), int(rng.uniform(10, 24))).resize((max(1, size // 2), size), Image.Resampling.BICUBIC)
    base = base.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(1.6, 3.2))))
    thr = int(rng.uniform(130, 190))
    m = base.point(lambda p: 255 if p > thr else 0).filter(ImageFilter.GaussianBlur(radius=1.2))
    left = m
    right = ImageOps.mirror(m)
    full = Image.new("L", (size, size), 0)
    full.paste(left, (0, 0))
    full.paste(right, (size - left.size[0], 0))
    full = _roughen_mask_edges(full, rng=rng, strength=0.55)
    col = ImageOps.colorize(full, black=(0, 0, 0), white=color_a).convert("RGBA")
    try:
        n2 = Image.effect_noise((size, size), int(rng.uniform(16, 34))).filter(ImageFilter.GaussianBlur(radius=2.0))
        m2 = n2.point(lambda p: 255 if p > int(rng.uniform(210, 238)) else 0)
        col2 = Image.new("RGBA", (size, size), (color_b[0], color_b[1], color_b[2], 0))
        col2.putalpha(m2.point(lambda p: int((p / 255.0) * int(opacity * 0.45))))
        col = Image.alpha_composite(col, col2)
    except Exception:
        pass
    col.putalpha(int(opacity))
    return col


def _make_brush_strokes_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Thick hand-painted strokes: curvy polylines with width variation and rough edges.
    Avoids “random noise everywhere” and reads as deliberate handcrafted paint.
    """
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)

    strokes = int(rng.uniform(2, 5))
    for _ in range(strokes):
        # Polyline points
        x = rng.uniform(-0.1, 1.1) * size
        y = rng.uniform(0.05, 0.95) * size
        pts = [(x, y)]
        segs = int(rng.uniform(4, 7))
        for _k in range(segs):
            x += rng.uniform(0.12, 0.32) * size * rng.choice([-1, 1])
            y += rng.uniform(-0.18, 0.18) * size
            pts.append((x, y))

        base_w = rng.uniform(size / 28.0, size / 14.0)
        # Draw multiple passes for width variation
        passes = int(rng.uniform(2, 4))
        for p in range(passes):
            w = int(base_w * rng.uniform(0.65, 1.05))
            a = int(255 * rng.uniform(0.55, 0.95))
            d.line(pts, fill=a, width=max(1, w), joint="curve")
            # Slight offset for “bristle” feel
            ox = rng.uniform(-size / 240.0, size / 240.0)
            oy = rng.uniform(-size / 240.0, size / 240.0)
            pts = [(px + ox, py + oy) for (px, py) in pts]

    # Rough edges: modulate with directional noise
    m = m.filter(ImageFilter.GaussianBlur(radius=max(1.0, size / 1400.0)))
    try:
        n = Image.effect_noise((size, size), int(rng.uniform(7, 16))).filter(ImageFilter.GaussianBlur(radius=1.0))
        n = n.rotate(float(rng.uniform(-25, 25)), expand=False, resample=Image.Resampling.BICUBIC)
        # Multiply keeps the center strong and tears the edges.
        m = ImageChops.multiply(m, n.point(lambda p: int(120 + (p - 120) * 0.9)))
    except Exception:
        pass

    m = _roughen_mask_edges(m, rng=rng, strength=0.45)
    layer = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    layer.putalpha(m.point(lambda p: int((p / 255.0) * int(opacity))))

    # Subtle edge highlight so strokes pop after DXT.
    try:
        edge = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.1))
        hi = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        hi.putalpha(edge.point(lambda p: int(p * 0.22)))
        layer = Image.alpha_composite(layer, hi)
    except Exception:
        pass

    return layer


def _make_glitch_layer(
    size: int,
    *,
    color: Tuple[int, int, int],
    opacity: int,
    rng: "random.Random",
) -> Image.Image:
    """
    Glitch slice overlay: offsets random horizontal strips.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # Start from a noisy band mask
    base = Image.effect_noise((size, size), int(rng.uniform(16, 40))).point(lambda p: 255 if p > int(rng.uniform(195, 225)) else 0)
    base = base.filter(ImageFilter.GaussianBlur(radius=max(1, size // 1600)))
    # Build color layer
    col = Image.new("RGBA", (size, size), (color[0], color[1], color[2], 0))
    col.putalpha(base.point(lambda p: int((p / 255.0) * opacity)))
    layer = Image.alpha_composite(layer, col)
    # Slice-shift
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bands = int(rng.uniform(14, 26))
    for _ in range(bands):
        y0 = int(rng.uniform(0, size - 1))
        h = int(rng.uniform(size * 0.016, size * 0.085))
        y1 = min(size, y0 + h)
        xoff = int(rng.uniform(-size * 0.16, size * 0.16))
        strip = layer.crop((0, y0, size, y1))
        out.alpha_composite(strip, (xoff, y0))
    # Add scanline feel
    scan = Image.new("L", (size, size), 0)
    sd = ImageDraw.Draw(scan)
    step = int(rng.uniform(18, 36))
    for y in range(0, size, step):
        sd.rectangle((0, y, size, y + 1), fill=255)
    scan = scan.filter(ImageFilter.GaussianBlur(radius=0.6))
    scan_rgba = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    scan_rgba.putalpha(scan.point(lambda p: int(p * 0.10 * opacity / 255.0)))
    out = Image.alpha_composite(out, scan_rgba)

    out = out.filter(ImageFilter.GaussianBlur(radius=max(1, size // 1000)))
    return out


def _make_proj_shad_from_logo(
    *,
    size: int,
    logo: Image.Image,
    rgb: Tuple[int, int, int],
    invert: bool = False,
) -> Image.Image:
    """
    Create a simple projection texture for ProjShad.dds:
    - Bright background with a darker logo-shaped imprint (or inverted).
    The actual in-game usage depends on the car mod, but this gives a visible 'projection' style.
    """
    # Base: white background (tutorials often treat white as "transparent / lets light through")
    bg = Image.new("RGB", (size, size), (255, 255, 255))

    # Logo mask
    lg = logo.convert("RGBA")
    # Fit logo into center with margin
    target = int(size * 0.66)
    w, h = lg.size
    scale = target / max(1, max(w, h))
    lg = lg.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    alpha = lg.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(2, size // 120)))

    # Compose: imprint color (keep dark for readable projection; tinted projections can be pack-dependent)
    r, g, b = int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF
    # Force darker imprint (shadow-like) even if the user passed a bright accent.
    imprint_rgb = (min(r, 48), min(g, 48), min(b, 48))
    imprint = Image.new("RGB", (size, size), imprint_rgb)
    # Place at center
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.paste(lg, ((size - lg.size[0]) // 2, (size - lg.size[1]) // 2), lg)
    mask = layer.getchannel("A")

    # Blend imprint with background
    if invert:
        # bright logo on darkened bg
        bg2 = Image.new("RGB", (size, size), (25, 25, 25))
        out = Image.composite(bg, bg2, mask)
    else:
        # dark imprint on bright bg
        out = bg.copy()
        out = Image.composite(imprint, out, mask.point(lambda p: int(p * 0.85)))

    return out


def _finalize_proj_shad_rgb(img: Image.Image) -> Image.Image:
    """
    ProjShad hygiene based on common community tutorials:
    - ProjShad is a projection; flip horizontally so text isn't backward.
    - Keep a white border to avoid edge artifacts.
    - Use RGB (no alpha) when possible (DXT1).
    """
    out = img.convert("RGB")
    out = ImageOps.mirror(out)

    w, h = out.size
    border = max(1, min(6, w // 128))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, w - 1, border - 1), fill=(255, 255, 255))
    d.rectangle((0, h - border, w - 1, h - 1), fill=(255, 255, 255))
    d.rectangle((0, 0, border - 1, h - 1), fill=(255, 255, 255))
    d.rectangle((w - border, 0, w - 1, h - 1), fill=(255, 255, 255))
    return out


def _make_proj_shad_wings(*, size: int, darkness: int = 18) -> Image.Image:
    """
    Procedurally generate a "raven wings" style ProjShad image.

    Design goals from common community practice:
    - White background ("transparent / lets light through")
    - Dark tiremark-like wings behind the car
    - Works well when squashed/projected and at distance

    Returns RGB.
    """
    size = int(size)
    size = max(64, size)
    darkness = max(0, min(90, int(darkness)))
    bg = Image.new("RGB", (size, size), (255, 255, 255))

    # Draw into an RGBA layer so we can feather/blur, then composite onto white.
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    cx = size * 0.50
    cy = size * 0.56

    # Wing tuning (make it long + scary: wide span, sharper feather fan, claw-like tips)
    feather_count = 15
    base_len = size * 0.26
    len_step = size * 0.032
    base_th = max(3, size // 160)
    th_step = 2

    # Primary sweep behind the car (more horizontal, longer span)
    sweep_w = max(9, size // 16)
    sweep_a = 34
    for sign in (-1, 1):
        x0 = int(cx + sign * size * 0.06)
        y0 = int(cy + size * 0.02)
        x1 = int(cx + sign * size * 0.48)
        y1 = int(cy + size * 0.16)
        d.line((x0, y0, x1, y1), fill=(darkness, darkness, darkness, sweep_a), width=sweep_w)

    # Feather lines: progressively longer, more horizontal fan, with jagged "claw" tips.
    for sign in (-1, 1):
        sx = cx + sign * size * 0.06
        sy = cy + size * 0.01
        for i in range(feather_count):
            t = i / max(1, feather_count - 1)
            # Smaller vertical component => longer horizontal wings
            ang = (10 + 46 * t)  # degrees away from center
            ang = ang if sign > 0 else (180 - ang)
            rad = math.radians(ang)
            length = base_len + len_step * i
            thickness = base_th + int(th_step * (feather_count - 1 - i) * 0.30)
            alpha = int(120 * (0.95 - 0.55 * t))

            ex = sx + math.cos(rad) * length
            ey = sy + math.sin(rad) * length

            # Slight "double stroke" for feather texture.
            d.line((int(sx), int(sy), int(ex), int(ey)), fill=(darkness, darkness, darkness, alpha), width=thickness)
            d.line(
                (int(sx), int(sy) + 1, int(ex), int(ey) + 1),
                fill=(darkness, darkness, darkness, int(alpha * 0.55)),
                width=max(1, thickness - 1),
            )

            # Add a tiny fork/jag at the feather tip for a more "scary" silhouette.
            tip_len = max(6, int(size * 0.020))
            tip_ang = rad + (0.28 if sign > 0 else -0.28)
            tx = ex + math.cos(tip_ang) * tip_len
            ty = ey + math.sin(tip_ang) * tip_len
            d.line((int(ex), int(ey), int(tx), int(ty)), fill=(darkness, darkness, darkness, int(alpha * 0.65)), width=max(1, thickness - 1))

    # Center tail marks (subtle), like tyre scuffs.
    tail_w = max(3, size // 120)
    for k in range(5):
        y = int(cy + size * (0.08 + 0.018 * k))
        x0 = int(cx - size * (0.06 + 0.010 * k))
        x1 = int(cx + size * (0.06 + 0.010 * k))
        a = int(95 - k * 10)
        d.line((x0, y, x1, y), fill=(darkness, darkness, darkness, a), width=tail_w)

    # Feather/soften so it projects nicely (keep a bit sharper than the old version).
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, size / 260)))

    # Composite onto white.
    bg_rgba = bg.convert("RGBA")
    out = Image.alpha_composite(bg_rgba, layer).convert("RGB")
    return out


def _read_zip_names(zip_path: Path) -> set[str]:
    """Return all filenames in a zip, or empty set if unreadable."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            return set(z.namelist())
    except Exception:
        return set()


def _ensure_stadium_aux_textures(
    *,
    base_names: set[str],
    additions: Dict[str, bytes],
    mipmaps: bool,
) -> None:
    """
    Ensure Stadium aux textures exist (community rules / compatibility):
    - Dirty maps: if absent, game falls back to default Stadium dirties (can mismatch the pack).
    - Illum: if absent, game falls back to environment default illum.

    We add safe defaults:
    - Dirty: alpha=0 => "no dirt overlay" but overrides defaults.
    - Illum: RGB=0 => "no illum" but overrides defaults (DXT1 preferred).
    """
    if "DiffuseDirty.dds" not in base_names:
        dd = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        additions["DiffuseDirty.dds"] = build_dds_dxt5_bytes(dd, mipmaps=mipmaps)
    if "DetailsDirty.dds" not in base_names:
        dt = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        additions["DetailsDirty.dds"] = build_dds_dxt5_bytes(dt, mipmaps=mipmaps)
    if "Illum.dds" not in base_names:
        il = Image.new("RGB", (1024, 1024), (0, 0, 0))
        additions["Illum.dds"] = build_dds_dxt1_bytes(il, mipmaps=mipmaps)


def _build_illum_override_dds(
    *,
    base_zip_path: Path,
    illum_image_path: str,
    mipmaps: bool,
) -> Tuple[str, bytes]:
    """
    Build an Illum.dds payload from a user image:
    - Resizes to match base zip's Illum.dds if present, else 1024.
    - If source has meaningful alpha, treat it as a glow mask (premultiply into RGB).
    - Prefer DXT1 when possible (TMNF/TMUF illum alpha is generally unused), but preserve DXT5 if the base uses it.

    Returns: (mode, payload) where mode is 'replace' or 'add'.
    """
    base_names = _read_zip_names(base_zip_path)
    iw, ih, ifourcc = (1024, 1024, "DXT1")
    if "Illum.dds" in base_names:
        try:
            with zipfile.ZipFile(base_zip_path, "r") as zin:
                hdr = zin.open("Illum.dds").read(128)
            iw, ih = _read_dds_dimensions_from_bytes(hdr)
            ifourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT1"
        except Exception:
            pass

    src = Image.open(illum_image_path).convert("RGBA").resize((iw, ih), Image.Resampling.LANCZOS)
    a = src.getchannel("A")
    rgb = src.convert("RGB")

    # If alpha is meaningful, use it as a glow mask.
    try:
        a_ext = a.getextrema()
    except Exception:
        a_ext = (255, 255)
    if a_ext != (255, 255):
        black = Image.new("RGB", (iw, ih), (0, 0, 0))
        rgb = Image.composite(rgb, black, a)

    if ifourcc == "DXT5":
        rgba = rgb.convert("RGBA")
        rgba.putalpha(255)
        payload = build_dds_dxt5_bytes(rgba, mipmaps=mipmaps)
    else:
        payload = build_dds_dxt1_bytes(rgb, mipmaps=mipmaps)

    return ("replace" if "Illum.dds" in base_names else "add", payload)


def _warn_if_base_zip_looks_suspicious(base_zip_path: Path, base_names: set[str]) -> None:
    """
    Heuristic warnings only (non-fatal).
    The base zip should be a working Stadium pack; missing core files usually means the zip isn't a Stadium mod pack
    or isn't structured as expected.
    """
    missing = []
    for core in ("Diffuse.dds", "Icon.dds"):
        if core not in base_names:
            missing.append(core)
    if missing:
        print(
            "WARNING: base zip is missing expected files: "
            + ", ".join(missing)
            + ". This may still work (some packs omit Icon), but results can be inconsistent."
        )
        print("WARNING: Recommended workflow: pick a known-good Stadium mod zip with Diffuse/Icon/ProjShad/Details.")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_base_zip_profile(
    base_zip_path: Path,
    *,
    profile_path: Optional[str],
    allow_auto: bool = True,
) -> Optional[Dict[str, object]]:
    """
    Load a base-zip profile JSON (from tools/profile_base_zip.py).

    - If profile_path is provided, loads that file (relative paths are treated as repo-root relative).
    - Otherwise, if allow_auto is true, tries: profiles/<sha256(base_zip)>.json
    """
    root = Path(__file__).resolve().parent

    p: Optional[Path] = None
    if profile_path:
        p = Path(profile_path).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
    elif allow_auto:
        try:
            sha = _sha256_file(base_zip_path)
            p = (root / "profiles" / f"{sha}.json").resolve()
        except Exception:
            p = None

    if p is None or (not p.exists()):
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _make_multi_stop_gradient(
    size: int,
    *,
    stops: List[Tuple[float, Tuple[int, int, int]]],
    angle_deg: float = -18.0,
    noise_strength: float = 0.0,
    quant_steps: int = 0,
) -> Image.Image:
    """
    Create an RGB gradient with multiple color stops along a diagonal direction.
    stops: list of (position 0..1, (r,g,b)), must be sorted by position.
    """
    if np is None:
        # Fallback to a simple vertical gradient between first and last.
        c0 = stops[0][1]
        c1 = stops[-1][1]
        g = Image.new("L", (1, size))
        for y in range(size):
            t = y / max(1, size - 1)
            r = int(c0[0] * (1 - t) + c1[0] * t)
            g.putpixel((0, y), r)
        img = ImageOps.colorize(g.resize((size, size)), black=c0, white=c1).convert("RGBA")
        img.putalpha(255)
        return img

    stops = sorted(stops, key=lambda s: s[0])
    # grid
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    ang = np.deg2rad(angle_deg)
    dx = np.cos(ang).astype(np.float32)
    dy = np.sin(ang).astype(np.float32)
    # projection range normalization
    t = X * dx + Y * dy
    t_min = float(t.min())
    t_max = float(t.max())
    if abs(t_max - t_min) < 1e-6:
        tt = np.zeros_like(t)
    else:
        tt = (t - t_min) / (t_max - t_min)

    if noise_strength > 0.0:
        n = (np.random.rand(size, size).astype(np.float32) - 0.5) * float(noise_strength)
        tt = np.clip(tt + n, 0.0, 1.0)

    if quant_steps and quant_steps > 1:
        qs = float(quant_steps)
        tt = np.round(tt * qs) / qs

    # allocate
    out = np.zeros((size, size, 3), dtype=np.float32)

    # piecewise interpolation
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (tt >= p0) & (tt <= p1)
        if not np.any(m):
            continue
        local = (tt[m] - p0) / max(1e-6, (p1 - p0))
        c0a = np.array(c0, dtype=np.float32)
        c1a = np.array(c1, dtype=np.float32)
        out[m] = c0a * (1.0 - local[:, None]) + c1a * local[:, None]

    # clamp ends
    out[tt < stops[0][0]] = np.array(stops[0][1], dtype=np.float32)
    out[tt > stops[-1][0]] = np.array(stops[-1][1], dtype=np.float32)

    # Add very small RGB dither to reduce banding (without changing brightness like before).
    if noise_strength <= 0.0:
        jitter = (np.random.rand(size, size, 1).astype(np.float32) - 0.5) * 6.0
        out = out + jitter

    rgb = np.clip(out, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb).convert("RGBA")
    img.putalpha(255)
    return img


# =============================================================================
# PRO SKIN DESIGN SYSTEM
# =============================================================================

# Curated color palettes based on color theory (complementary, triadic, analogous)
PRO_COLOR_PALETTES = {
    "cyberpunk": {
        "base": (10, 8, 18),        # Deep purple-black
        "secondary": (25, 15, 45),   # Dark purple
        "accent": (255, 0, 136),     # Hot pink/magenta
        "highlight": (0, 255, 255),  # Cyan
        "text": (255, 255, 255),
    },
    "racing_orange": {
        "base": (15, 15, 18),        # Near black
        "secondary": (35, 30, 28),   # Warm dark grey
        "accent": (255, 106, 0),     # Racing orange
        "highlight": (255, 200, 60), # Gold
        "text": (255, 255, 255),
    },
    "ocean_teal": {
        "base": (4, 24, 36),         # Deep ocean
        "secondary": (8, 45, 58),    # Dark teal
        "accent": (0, 184, 212),     # Bright cyan
        "highlight": (0, 229, 255),  # Electric cyan
        "text": (255, 255, 255),
    },
    "volcanic": {
        "base": (12, 8, 6),          # Charred black
        "secondary": (35, 18, 10),   # Dark ember
        "accent": (255, 69, 0),      # Red-orange
        "highlight": (255, 140, 0),  # Bright orange
        "text": (255, 255, 255),
    },
    "arctic": {
        "base": (10, 24, 40),        # Dark ice blue
        "secondary": (25, 50, 75),   # Steel blue
        "accent": (79, 195, 247),    # Ice blue
        "highlight": (225, 245, 254),# Near white ice
        "text": (255, 255, 255),
    },
    "midnight_gold": {
        "base": (13, 10, 20),        # Deep purple-black
        "secondary": (30, 20, 45),   # Dark violet
        "accent": (155, 89, 182),    # Purple
        "highlight": (255, 215, 0),  # Gold
        "text": (255, 255, 255),
    },
    # Black + Gold (Eror PRO pack) – luxury / JPS-inspired variants.
    "blackgold": {
        "base": (5, 5, 5),             # Deepest Obsidian Black (Matte)
        "secondary": (20, 20, 20),     # Dark Carbon/Graphite
        "accent": (218, 165, 32),      # Metallic Gold Leaf
        "highlight": (255, 223, 100),  # Bright Gold Highlight
        "text": (255, 255, 255),
    },
    "blackgold_amber": {
        "base": (6, 6, 8),           # Deep black
        "secondary": (24, 18, 10),   # Warm graphite
        "accent": (255, 193, 7),     # Amber (#FFC107)
        "highlight": (255, 240, 175),# Soft champagne
        "text": (255, 255, 255),
    },
    "blackgold_stealth": {
        "base": (10, 10, 12),        # Matte black
        "secondary": (22, 22, 26),   # Dark grey
        "accent": (184, 134, 11),    # Dark goldenrod
        "highlight": (235, 212, 140),# Muted champagne
        "text": (255, 255, 255),
    },
    "forest": {
        "base": (8, 18, 12),         # Dark forest
        "secondary": (15, 35, 22),   # Forest green
        "accent": (46, 204, 113),    # Emerald
        "highlight": (180, 255, 180),# Light green
        "text": (255, 255, 255),
    },
    "blood_moon": {
        "base": (18, 8, 10),         # Dark crimson
        "secondary": (40, 15, 18),   # Dark red
        "accent": (192, 57, 43),     # Blood red
        "highlight": (255, 100, 80), # Coral
        "text": (255, 255, 255),
    },
    "monochrome": {
        "base": (15, 15, 18),        # Near black
        "secondary": (45, 45, 50),   # Dark grey
        "accent": (180, 180, 185),   # Light grey
        "highlight": (245, 245, 250),# Near white
        "text": (255, 255, 255),
    },
    "neon_green": {
        "base": (8, 12, 8),          # Dark green-black
        "secondary": (15, 25, 15),   # Dark green
        "accent": (57, 255, 20),     # Neon green
        "highlight": (180, 255, 180),# Light green
        "text": (255, 255, 255),
    },
    # TM2020-inspired “pro” vibes: high-chroma accents on deep bases.
    "synthwave": {
        "base": (6, 5, 14),          # Deeper violet-black (more contrast)
        "secondary": (20, 8, 46),    # Purple shadow (punchier)
        "accent": (255, 0, 190),     # Hot magenta (sharper)
        "highlight": (0, 255, 255),  # Pure cyan (sharper)
        "text": (255, 255, 255),
    },
    "ultraviolet": {
        "base": (6, 6, 14),          # Ink
        "secondary": (18, 10, 28),   # Violet shadow
        "accent": (156, 68, 255),    # UV purple
        "highlight": (255, 245, 110),# Neon lemon
        "text": (255, 255, 255),
    },
    "neon_sunset": {
        "base": (10, 8, 12),         # Near black
        "secondary": (28, 10, 16),   # Warm maroon shadow
        "accent": (255, 106, 0),     # Asiimov orange
        "highlight": (255, 245, 245),# Clean white
        "text": (255, 255, 255),
    },
    "acid_ice": {
        "base": (4, 8, 14),          # Darker base (more contrast)
        "secondary": (8, 20, 30),    # Deep teal shadow
        "accent": (0, 255, 230),     # Neon aqua (sharper)
        "highlight": (120, 255, 60), # Acid green (no yellow)
        "text": (255, 255, 255),
    },
    "neon_koi": {
        "base": (6, 6, 9),           # Ink black (deeper)
        "secondary": (20, 10, 30),   # Purple shadow
        "accent": (0, 255, 255),     # Neon cyan (max)
        "highlight": (255, 50, 150), # Koi pink (sharper)
        "text": (255, 255, 255),
    },
    "aurora": {
        "base": (4, 8, 14),          # Deep night (more contrast)
        "secondary": (8, 16, 30),    # Night teal
        "accent": (0, 255, 170),     # Aurora green
        "highlight": (200, 110, 255),# Aurora violet (brighter)
        "text": (255, 255, 255),
    },
}


def _make_diagonal_slash(
    size: int,
    *,
    color: Tuple[int, int, int, int],
    angle_deg: float = 55.0,
    thickness: float = 0.15,
    position: float = 0.5,
    feather: int = 8,
) -> Image.Image:
    """
    Create a diagonal slash/stripe across the image.
    - angle_deg: angle of the slash (0=horizontal, 90=vertical, 45-60=typical racing)
    - thickness: width as fraction of image size
    - position: center position along perpendicular axis (0-1)
    - feather: blur radius for soft edges
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    
    # Calculate slash geometry
    angle_rad = np.deg2rad(angle_deg) if np else (angle_deg * 3.14159 / 180.0)
    
    # Extend beyond canvas to ensure full coverage
    ext = size * 2
    half_thick = (thickness * size) / 2
    
    # Center point offset by position
    cx = size * position
    cy = size * 0.5
    
    # Direction vectors
    if np:
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        px = -np.sin(angle_rad)  # perpendicular
        py = np.cos(angle_rad)
    else:
        import math
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        px = -math.sin(angle_rad)
        py = math.cos(angle_rad)
    
    # Four corners of the slash rectangle
    pts = [
        (cx - dx * ext + px * half_thick, cy - dy * ext + py * half_thick),
        (cx + dx * ext + px * half_thick, cy + dy * ext + py * half_thick),
        (cx + dx * ext - px * half_thick, cy + dy * ext - py * half_thick),
        (cx - dx * ext - px * half_thick, cy - dy * ext - py * half_thick),
    ]
    
    d.polygon(pts, fill=color)
    
    if feather > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=feather))
    
    return layer


def _make_swoosh_curve(
    size: int,
    *,
    color: Tuple[int, int, int, int],
    thickness: int = 60,
    curve_type: str = "wave",  # "wave", "arc", "slash"
    flip: bool = False,
) -> Image.Image:
    """
    Create an organic swoosh/curve shape.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    
    points = []
    
    if curve_type == "wave":
        # S-curve from bottom-left to top-right
        for i in range(100):
            t = i / 99.0
            x = t * size
            # Sine wave offset
            if np:
                y = size * 0.5 + np.sin(t * 3.14159 * 1.5) * size * 0.25
            else:
                import math
                y = size * 0.5 + math.sin(t * 3.14159 * 1.5) * size * 0.25
            points.append((x, y))
    elif curve_type == "arc":
        # Smooth arc from corner
        for i in range(100):
            t = i / 99.0
            if np:
                angle = t * np.pi * 0.5
                x = size * 0.2 + np.cos(angle) * size * 0.7
                y = size * 0.8 - np.sin(angle) * size * 0.6
            else:
                import math
                angle = t * math.pi * 0.5
                x = size * 0.2 + math.cos(angle) * size * 0.7
                y = size * 0.8 - math.sin(angle) * size * 0.6
            points.append((x, y))
    else:  # slash
        points = [(0, size * 0.7), (size, size * 0.3)]
    
    if flip:
        points = [(size - x, y) for x, y in points]
    
    # Draw as thick line
    if len(points) >= 2:
        d.line(points, fill=color, width=thickness, joint="curve")
    
    # Blur for softer edges
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(2, thickness // 8)))
    
    return layer


def _make_geometric_blocks(
    size: int,
    *,
    color1: Tuple[int, int, int, int],
    color2: Tuple[int, int, int, int],
    block_style: str = "angular",  # "angular", "split", "corner"
) -> Image.Image:
    """
    Create geometric block patterns for aggressive esports look.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    
    if block_style == "angular":
        # Angled block covering ~40% of image
        pts = [
            (0, size * 0.3),
            (size * 0.6, 0),
            (size, 0),
            (size, size * 0.4),
            (size * 0.4, size * 0.7),
            (0, size * 0.5),
        ]
        d.polygon(pts, fill=color1)
        
        # Secondary smaller block
        pts2 = [
            (size * 0.7, size),
            (size, size * 0.6),
            (size, size),
        ]
        d.polygon(pts2, fill=color2)
        
    elif block_style == "split":
        # Diagonal split
        d.polygon([(0, 0), (size, 0), (size, size * 0.4), (0, size * 0.6)], fill=color1)
        
    elif block_style == "corner":
        # Corner accent blocks
        d.polygon([(0, 0), (size * 0.3, 0), (0, size * 0.3)], fill=color1)
        d.polygon([(size, size), (size * 0.7, size), (size, size * 0.7)], fill=color2)
    
    return layer


def _make_edge_highlights(
    size: int,
    *,
    uv_islands: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int, int],
    thickness: int = 3,
) -> Image.Image:
    """
    Draw thin highlight lines along UV island edges.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    
    for (x0, y0, x1, y1) in uv_islands:
        # Draw rectangle outline
        d.rectangle((x0, y0, x1, y1), outline=color, width=thickness)
    
    # Slight blur for glow effect
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(1, thickness // 2)))
    
    return layer


def _make_panel_gradient(
    size: int,
    *,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
    angle_deg: float = 0.0,
) -> Image.Image:
    """
    Create a gradient simulating light from above (lighter top, darker bottom).
    """
    if np is None:
        # Simple vertical gradient fallback
        g = Image.new("L", (1, size))
        for y in range(size):
            t = y / max(1, size - 1)
            v = int(255 * (1 - t * 0.3))  # 30% darker at bottom
            g.putpixel((0, y), v)
        img = ImageOps.colorize(g.resize((size, size)), black=bottom_color, white=top_color)
        return img.convert("RGBA")
    
    # Numpy gradient
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    
    ang = np.deg2rad(angle_deg)
    t = Y * np.cos(ang) + X * np.sin(ang)
    t = (t - t.min()) / max(1e-6, t.max() - t.min())
    
    top = np.array(top_color, dtype=np.float32)
    bot = np.array(bottom_color, dtype=np.float32)
    
    rgb = top * (1 - t[:, :, None]) + bot * t[:, :, None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(rgb).convert("RGBA")
    img.putalpha(255)
    return img


def _apply_texture_overlay(
    base: Image.Image,
    texture_type: str = "noise",
    opacity: float = 0.1,
    color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Apply a subtle texture overlay (noise, grain, carbon, etc.) at given opacity.
    """
    size = base.size[0]
    
    if texture_type == "noise":
        tex = Image.effect_noise((size, size), 25)
    elif texture_type == "grain":
        tex = Image.effect_noise((size, size), 15).filter(ImageFilter.GaussianBlur(radius=1))
    elif texture_type == "carbon":
        # Create carbon fiber pattern
        tex = Image.new("L", (size, size), 128)
        d = ImageDraw.Draw(tex)
        spacing = max(4, size // 128)
        for y in range(0, size, spacing):
            for x in range(0, size, spacing):
                offset = spacing // 2 if (y // spacing) % 2 else 0
                d.rectangle((x + offset, y, x + offset + spacing // 2, y + spacing // 2), fill=100)
        tex = tex.filter(ImageFilter.GaussianBlur(radius=1))
    else:
        tex = Image.effect_noise((size, size), 20)
    
    # Colorize if requested
    if color:
        tex_rgb = ImageOps.colorize(tex, black=(0, 0, 0), white=color).convert("RGBA")
    else:
        tex_rgb = tex.convert("RGBA")
    
    # Set opacity
    alpha = tex_rgb.getchannel("A").point(lambda p: int(p * opacity))
    tex_rgb.putalpha(alpha)
    
    return Image.alpha_composite(base.convert("RGBA"), tex_rgb)


def _generate_pro_skin_layers(
    size: int,
    *,
    palette_name: str = "cyberpunk",
    shape_style: str = "slashes",  # "slashes", "swoosh", "blocks", "minimal", or combined like "slashes_fade"
    custom_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    base_effect: str = "gradient",  # "gradient", "fade", "splatter", "galaxy", "fluid", "carbon", "razzle", "holo", "crafted"
    theme: Optional[str] = None,  # Override to use a specific visual theme
    rng: Optional["random.Random"] = None,
    feature_rects: Optional[Dict[str, object]] = None,
    grade_contrast: Optional[float] = None,
    grade_color: Optional[float] = None,
    grade_gamma: Optional[float] = None,
    vignette_strength: Optional[int] = None,
) -> Image.Image:
    """
    Generate a pro-quality layered skin using the pro design system.
    
    Layer stack:
    1. Base effect (gradient/fade/splatter/galaxy/fluid)
    2. Secondary color zone (optional based on effect)
    3. Accent shapes (slashes/swooshes/blocks)
    4. Texture overlay
    5. Pinstripe highlights
    """
    # RNG (used by some pro motifs to ensure --seed actually changes the design)
    if rng is None:
        seed_src = f"{palette_name or ''}|{shape_style}|{base_effect}|{theme or ''}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)

    # Get palette
    if custom_palette:
        pal = custom_palette
    else:
        pal = PRO_COLOR_PALETTES.get(palette_name, PRO_COLOR_PALETTES["cyberpunk"])
    
    base_rgb = pal["base"]
    secondary_rgb = pal["secondary"]
    accent_rgb = pal["accent"]
    highlight_rgb = pal["highlight"]
    
    # Layer 1: Base effect
    if base_effect == "fade":
        # Multi-stop fade like Kacky
        img = _make_multi_stop_gradient(
            size,
            stops=[
                (0.0, base_rgb),
                (0.5, secondary_rgb),
                (1.0, highlight_rgb),
            ],
            angle_deg=-25.0,
            noise_strength=0.18,
            quant_steps=14,
        )
    elif base_effect == "splatter":
        # Dark base with gold dust/splatter overlay (more premium than flat fill).
        darker_base = tuple(max(0, c - 18) for c in base_rgb)
        img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=darker_base)
        # A subtle secondary zone for depth (kept dark so it stays “black”).
        sec_a = 60 if (int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])) < 70 else 90
        secondary_block = _make_diagonal_slash(
            size,
            color=(secondary_rgb[0], secondary_rgb[1], secondary_rgb[2], sec_a),
            angle_deg=25.0,
            thickness=0.30,
            position=0.65,
            feather=22,
        )
        img = Image.alpha_composite(img, secondary_block)

        # Fine dust (no big slashes) + a softer highlight layer.
        # Keep it “speckle” heavy so we don’t end up with huge khaki panels.
        spl1 = _make_splatter_layer(
            size,
            color=accent_rgb,
            opacity=135,
            blob_scale=10,
            blob_threshold=236,
            dots=True,
            dot_threshold=236,
            slashes=False,
            rng=rng,
        )
        spl2 = _make_splatter_layer(
            size,
            color=highlight_rgb,
            opacity=85,
            blob_scale=12,
            blob_threshold=238,
            dots=True,
            dot_threshold=238,
            slashes=False,
            rng=rng,
        )
        img = Image.alpha_composite(img, spl1)
        img = Image.alpha_composite(img, spl2)

        # Extra micro “sparkles” so gold reads as metallic in-game.
        try:
            dust = _make_starfield_layer(size, density=0.0080, color=(min(255, highlight_rgb[0]), min(255, highlight_rgb[1]), min(255, highlight_rgb[2])), max_alpha=85)
            img = Image.alpha_composite(img, dust)
        except Exception:
            pass

        # Subtle brushed “foil” grain (very low) to help gold read as metallic after DXT.
        try:
            foil = _make_brushed_metal_layer(
                size,
                color=(min(255, highlight_rgb[0] + 10), min(255, highlight_rgb[1] + 10), min(255, highlight_rgb[2] + 10)),
                opacity=22,
                rng=rng,
            )
            img = Image.alpha_composite(img, foil)
        except Exception:
            pass
    elif base_effect == "galaxy":
        # Nebula base
        img = Image.new("RGBA", (size, size), (base_rgb[0], base_rgb[1], base_rgb[2], 255))
        neb1 = _make_galaxy_nebula_layer(size, dark=base_rgb, bright=accent_rgb, opacity=80)
        neb2 = _make_galaxy_nebula_layer(size, dark=secondary_rgb, bright=highlight_rgb, opacity=60)
        stars = _make_starfield_layer(size, density=0.0012, color=(255, 255, 255), max_alpha=80)
        img = Image.alpha_composite(img, neb1)
        img = Image.alpha_composite(img, neb2)
        img = Image.alpha_composite(img, stars)
    elif base_effect == "fluid":
        # Fluid/marble swirl base
        img = Image.new("RGBA", (size, size), (base_rgb[0], base_rgb[1], base_rgb[2], 255))
        sheen = _make_fluid_sheen_layer(size, a=base_rgb, b=accent_rgb, opacity=90)
        img = Image.alpha_composite(img, sheen)
        # Add secondary fluid layer
        sheen2 = _make_fluid_sheen_layer(size, a=secondary_rgb, b=highlight_rgb, opacity=50)
        img = Image.alpha_composite(img, sheen2)
    elif base_effect == "carbon":
        # Carbon fiber base
        darker_base = tuple(max(0, c - 15) for c in base_rgb)
        img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=darker_base)
        # Dark themes (black/gold) can get washed out by bright carbon; keep it subtler and slightly warm.
        base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
        if base_sum < 75:
            carbon = _make_carbon_fiber_overlay(size, color=(255, 236, 190, 10))
        else:
            carbon = _make_carbon_fiber_overlay(size, color=(255, 255, 255, 40))
        img = Image.alpha_composite(img, carbon)
        # Add a dark secondary panel so the design has structure even before shapes.
        sec_a = 28 if base_sum < 70 else 88
        secondary_block = _make_diagonal_slash(
            size,
            color=(secondary_rgb[0], secondary_rgb[1], secondary_rgb[2], sec_a),
            angle_deg=25.0,
            thickness=0.34,
            position=0.62,
            feather=26,
        )
        img = Image.alpha_composite(img, secondary_block)
    elif base_effect == "razzle":
        # RAZZLE: aggressive multi-layer “pro” wrap (fade + shards + splatter + halftone + glitch).
        # The goal is loud but still readable after DXT compression.
        img = _make_multi_stop_gradient(
            size,
            stops=[
                (0.0, base_rgb),
                (0.35, secondary_rgb),
                (0.72, accent_rgb),
                (1.0, highlight_rgb),
            ],
            angle_deg=float(rng.uniform(-55.0, -15.0)),
            noise_strength=0.22,
            quant_steps=12,
        )
        # Big structural shards first (keeps “wrap” feel).
        try:
            sh = _make_shards_layer(size, colors=[accent_rgb, highlight_rgb, secondary_rgb], opacity=150, rng=rng)
            img = Image.alpha_composite(img, sh)
        except Exception:
            pass
        # Heavy splatter (accent + highlight).
        try:
            spl_a = _make_splatter_layer(size, color=accent_rgb, opacity=165, blob_scale=9, blob_threshold=234, dots=True, dot_threshold=236, slashes=True, rng=rng)
            spl_h = _make_splatter_layer(size, color=highlight_rgb, opacity=110, blob_scale=11, blob_threshold=236, dots=True, dot_threshold=238, slashes=False, rng=rng)
            img = Image.alpha_composite(img, spl_a)
            img = Image.alpha_composite(img, spl_h)
        except Exception:
            pass
        # Halftone + glitch for that esports “screen-print + scanline” energy.
        try:
            ht = _make_halftone_layer(size, color=highlight_rgb, opacity=34, rng=rng)
            img = Image.alpha_composite(img, ht)
        except Exception:
            pass
        try:
            gl = _make_glitch_layer(size, color=highlight_rgb, opacity=70, rng=rng)
            img = Image.alpha_composite(img, gl)
        except Exception:
            pass
        # Micro sparkles to prevent “flat dull” look on dark bases.
        try:
            dust = _make_starfield_layer(size, density=0.0045, color=(255, 255, 255), max_alpha=70)
            img = Image.alpha_composite(img, dust)
        except Exception:
            pass
    elif base_effect == "holo":
        # HOLO: iridescent neon gradient + fine prism lines (reads “premium” and vibrant).
        img = _make_multi_stop_gradient(
            size,
            stops=[
                (0.00, base_rgb),
                (0.22, accent_rgb),
                (0.46, highlight_rgb),
                (0.70, secondary_rgb),
                (1.00, accent_rgb),
            ],
            angle_deg=float(rng.uniform(-40.0, 40.0)),
            noise_strength=0.20,
            quant_steps=0,  # keep smoother; debanding happens later via --sanitize
        )
        # Prism pinstripes: thin repeated slashes with alternating highlight/accent.
        try:
            prism = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            # Dense thin bands, but low alpha to avoid muddying.
            for i in range(9):
                pos = 0.12 + i * 0.095 + float(rng.uniform(-0.015, 0.015))
                col = highlight_rgb if (i % 2 == 0) else accent_rgb
                a = 64 if (i % 2 == 0) else 46
                prism = Image.alpha_composite(
                    prism,
                    _make_diagonal_slash(size, color=(col[0], col[1], col[2], a), angle_deg=float(rng.uniform(55.0, 78.0)), thickness=0.010, position=pos, feather=1),
                )
            prism = prism.filter(ImageFilter.GaussianBlur(radius=0.6))
            img = Image.alpha_composite(img, prism)
        except Exception:
            pass
        # Add very subtle foil grain to fight DXT banding and add “sheen”.
        try:
            foil = _make_brushed_metal_layer(size, color=highlight_rgb, opacity=20, rng=rng)
            img = Image.alpha_composite(img, foil)
        except Exception:
            pass
    elif base_effect == "crafted":
        # CRAFTED: “handmade wrap” look with intentional blocks + brush swaths + controlled splatter.
        # Priorities: readability, negative space, organic edges (not random TV noise).
        darker_base = tuple(max(0, c - 18) for c in base_rgb)
        img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=darker_base, angle_deg=float(rng.uniform(-10.0, 12.0)))

        # Very subtle carbon/grain so large flats don’t look dead.
        try:
            base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
            carbon_a = 16 if base_sum < 80 else 28
            carbon = _make_carbon_fiber_overlay(size, color=(255, 255, 255, carbon_a))
            img = Image.alpha_composite(img, carbon)
        except Exception:
            pass

        # Tape blocks: secondary tone to carve structure.
        try:
            tb = _make_tape_block_layer(size, color=secondary_rgb, opacity=int(rng.uniform(70, 120)), rng=rng)
            img = Image.alpha_composite(img, tb)
        except Exception:
            pass

        # Brush strokes: one accent, one highlight.
        try:
            st1 = _make_brush_strokes_layer(size, color=accent_rgb, opacity=int(rng.uniform(120, 175)), rng=rng)
            img = Image.alpha_composite(img, st1)
        except Exception:
            pass
        try:
            st2 = _make_brush_strokes_layer(size, color=highlight_rgb, opacity=int(rng.uniform(70, 120)), rng=rng)
            img = Image.alpha_composite(img, st2)
        except Exception:
            pass

        # Controlled splatter (small + clustered) to add energy without turning grey.
        try:
            spl = _make_splatter_layer(
                size,
                color=highlight_rgb,
                opacity=int(rng.uniform(70, 110)),
                blob_scale=int(rng.uniform(10, 18)),
                dots=True,
                slashes=False,
                rng=rng,
            )
            img = Image.alpha_composite(img, spl)
        except Exception:
            pass

        # Pinstripes: a few thin separators, low alpha.
        try:
            pins = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            for _ in range(int(rng.uniform(3, 6))):
                pins = Image.alpha_composite(
                    pins,
                    _make_diagonal_slash(
                        size,
                        color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], int(rng.uniform(40, 70))),
                        angle_deg=float(rng.uniform(35.0, 78.0)),
                        thickness=0.0065,
                        position=float(rng.uniform(0.12, 0.88)),
                        feather=1,
                    ),
                )
            pins = pins.filter(ImageFilter.GaussianBlur(radius=0.6))
            img = Image.alpha_composite(img, pins)
        except Exception:
            pass
    elif base_effect == "aurora":
        # AURORA: silky neon bands with depth; less “random dots”, more deliberate gradients.
        img = _make_multi_stop_gradient(
            size,
            stops=[
                (0.00, base_rgb),
                (0.22, secondary_rgb),
                (0.52, accent_rgb),
                (0.76, highlight_rgb),
                (1.00, secondary_rgb),
            ],
            angle_deg=float(rng.uniform(-35.0, 15.0)),
            noise_strength=0.18,
            quant_steps=0,
        )
        try:
            sheen = _make_fluid_sheen_layer(size, a=secondary_rgb, b=accent_rgb, opacity=85)
            sheen2 = _make_fluid_sheen_layer(size, a=base_rgb, b=highlight_rgb, opacity=55)
            img = Image.alpha_composite(img, sheen)
            img = Image.alpha_composite(img, sheen2)
        except Exception:
            pass
        try:
            dust = _make_starfield_layer(size, density=0.0028, color=(255, 255, 255), max_alpha=55)
            img = Image.alpha_composite(img, dust)
        except Exception:
            pass
        # Deeper edges for stronger contrast
        try:
            vs = _clamp_int(vignette_strength, 0, 180)
            img = Image.alpha_composite(img, _vignette_layer(size, strength=int(vs if vs is not None else 82), rng=rng))
        except Exception:
            pass
        # Micro hatch to avoid “airbrush flatness”
        try:
            hatch = _make_micro_hatch_layer(size, color=highlight_rgb, opacity=16, rng=rng)
            img = Image.alpha_composite(img, hatch)
        except Exception:
            pass
    elif base_effect == "inkblot":
        # INKBLOT: handcrafted symmetric motif used as a depth layer.
        img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=tuple(max(0, c - 18) for c in base_rgb))
        try:
            ink = _make_inkblot_layer(size, color_a=secondary_rgb, color_b=accent_rgb, opacity=95, rng=rng)
            img = Image.alpha_composite(img, ink)
        except Exception:
            pass
        try:
            vs = _clamp_int(vignette_strength, 0, 180)
            img = Image.alpha_composite(img, _vignette_layer(size, strength=int(vs if vs is not None else 92), rng=rng))
        except Exception:
            pass
    else:  # "gradient" default
        darker_base = tuple(max(0, c - 15) for c in base_rgb)
        img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=darker_base)
        
        # Add secondary zone only for gradient base
        secondary_block = _make_diagonal_slash(
            size,
            color=(secondary_rgb[0], secondary_rgb[1], secondary_rgb[2], 140),
            angle_deg=25.0,
            thickness=0.30,
            position=0.65,
            feather=20,
        )
        img = Image.alpha_composite(img, secondary_block)
    
    # Layer 2: Accent shapes - UNIQUE per palette to avoid samey look
    # Use palette name to determine shape configuration
    palette_key = palette_name if palette_name else "default"
    
    # Define unique shape configs per palette
    PALETTE_SHAPE_CONFIGS = {
        "cyberpunk": {"angles": [65, 70, -20], "positions": [0.2, 0.25, 0.85], "style": "sharp"},
        "racing_orange": {"angles": [45, 40, 48], "positions": [0.15, 0.5, 0.55], "style": "racing"},
        "ocean_teal": {"angles": [-30, -25, 15], "positions": [0.3, 0.35, 0.7], "style": "wave"},
        "volcanic": {"angles": [30, 35, -45], "positions": [0.4, 0.45, 0.1], "style": "crack"},
        "arctic": {"angles": [0, 5, -5], "positions": [0.25, 0.5, 0.75], "style": "horizontal"},
        "midnight_gold": {"angles": [75, 80, -15], "positions": [0.6, 0.65, 0.2], "style": "luxury"},
        "blackgold": {"angles": [78, 82, -12], "positions": [0.62, 0.68, 0.22], "style": "luxury"},
        "blackgold_amber": {"angles": [58, 62, -22], "positions": [0.18, 0.44, 0.84], "style": "racing"},
        "blackgold_stealth": {"angles": [0, 90, 45], "positions": [0.33, 0.66, 0.50], "style": "geometric"},
        "forest": {"angles": [50, 55, 120], "positions": [0.35, 0.4, 0.8], "style": "organic"},
        "blood_moon": {"angles": [-55, -60, 25], "positions": [0.7, 0.75, 0.15], "style": "aggressive"},
        "monochrome": {"angles": [0, 90, 45], "positions": [0.33, 0.66, 0.5], "style": "geometric"},
        "neon_green": {"angles": [40, -40, 0], "positions": [0.2, 0.8, 0.5], "style": "neon"},
        "synthwave": {"angles": [68, 74, -18], "positions": [0.22, 0.52, 0.86], "style": "sharp"},
        "ultraviolet": {"angles": [62, 70, 12], "positions": [0.18, 0.47, 0.76], "style": "luxury"},
        "neon_sunset": {"angles": [46, 42, 52], "positions": [0.14, 0.48, 0.62], "style": "racing"},
        "acid_ice": {"angles": [-34, -26, 18], "positions": [0.30, 0.38, 0.74], "style": "wave"},
        "neon_koi": {"angles": [66, 74, -18], "positions": [0.20, 0.50, 0.86], "style": "sharp"},
        "aurora": {"angles": [-28, -20, 14], "positions": [0.28, 0.36, 0.72], "style": "wave"},
    }
    
    config = PALETTE_SHAPE_CONFIGS.get(palette_key, {"angles": [55, 50, 60], "positions": [0.3, 0.5, 0.7], "style": "default"})
    angles = config["angles"]
    positions = config["positions"]
    style_type = config["style"]
    
    if shape_style == "mixmatch":
        # MixMatch: intentionally combines multiple pattern families (fade/topo/halftone/glitch)
        # while keeping a strong hierarchy: base -> structure -> texture -> micro detail.
        # This is meant to be "inspired by livery design principles", not any single reference skin.
        # Base structure: one strong slash + one arc
        img = Image.alpha_composite(
            img,
            _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 190), angle_deg=angles[0], thickness=0.07, position=positions[0], feather=6),
        )
        img = Image.alpha_composite(
            img,
            _make_swoosh_curve(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 135), thickness=max(24, size // 44), curve_type="arc"),
        )
        # Texture overlays
        try:
            topo = _make_topographic_lines_layer(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255), opacity=22)
            img = Image.alpha_composite(img, topo)
        except Exception:
            pass
        try:
            ht = _make_halftone_layer(size, color=highlight_rgb, opacity=18, rng=rng)
            img = Image.alpha_composite(img, ht)
        except Exception:
            pass
        try:
            gl = _make_glitch_layer(size, color=highlight_rgb, opacity=42, rng=rng)
            img = Image.alpha_composite(img, gl)
        except Exception:
            pass
        # Optional: subtle camo only for brighter palettes (avoid muddying luxury palettes)
        try:
            base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
            if base_sum > 85:
                cam = _make_camo_layer(size, colors=[base_rgb, secondary_rgb, accent_rgb], opacity=22, rng=rng)
                img = Image.alpha_composite(img, cam)
        except Exception:
            pass

    elif shape_style == "slashes":
        if style_type == "sharp":
            # Cyberpunk: Sharp angular cuts
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 210), angle_deg=angles[0], thickness=0.12, position=positions[0], feather=2))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 180), angle_deg=angles[1], thickness=0.04, position=positions[1], feather=1))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 100), angle_deg=angles[2], thickness=0.08, position=positions[2], feather=3))
        elif style_type == "racing":
            # Racing: Classic racing stripes
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 230), angle_deg=angles[0], thickness=0.10, position=positions[0], feather=3))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 200), angle_deg=angles[1], thickness=0.03, position=positions[1], feather=1))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(base_rgb[0]+30, base_rgb[1]+30, base_rgb[2]+30, 150), angle_deg=angles[2], thickness=0.05, position=positions[2], feather=2))
        elif style_type == "horizontal":
            # Arctic: Clean horizontal bands
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 180), angle_deg=angles[0], thickness=0.06, position=positions[0], feather=8))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 220), angle_deg=angles[1], thickness=0.15, position=positions[1], feather=15))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 120), angle_deg=angles[2], thickness=0.04, position=positions[2], feather=5))
        elif style_type == "luxury":
            # Midnight gold: Elegant sweeping lines
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 200), angle_deg=angles[0], thickness=0.025, position=positions[0], feather=1))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 160), angle_deg=angles[1], thickness=0.015, position=positions[1], feather=0))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 140), angle_deg=angles[2], thickness=0.08, position=positions[2], feather=6))
        elif style_type == "aggressive":
            # Blood moon: Aggressive crossing slashes
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 220), angle_deg=angles[0], thickness=0.09, position=positions[0], feather=3))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 170), angle_deg=angles[1], thickness=0.06, position=positions[1], feather=2))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 130), angle_deg=angles[2], thickness=0.04, position=positions[2], feather=4))
        elif style_type == "geometric":
            # Monochrome: Grid-like geometric
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 200), angle_deg=0, thickness=0.02, position=0.33, feather=0))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 200), angle_deg=0, thickness=0.02, position=0.66, feather=0))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 150), angle_deg=90, thickness=0.02, position=0.5, feather=0))
        else:
            # Default varied slashes
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 200), angle_deg=angles[0], thickness=0.07, position=positions[0], feather=5))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 160), angle_deg=angles[1], thickness=0.03, position=positions[1], feather=2))
            img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 140), angle_deg=angles[2], thickness=0.05, position=positions[2], feather=4))
        
    elif shape_style == "swoosh":
        # Swoosh style varies by palette
        if style_type in ("wave", "ocean_teal", "organic"):
            swoosh = _make_swoosh_curve(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 200), thickness=max(44, size // 24), curve_type="wave")
            img = Image.alpha_composite(img, swoosh)
        elif style_type == "racing":
            swoosh = _make_swoosh_curve(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 210), thickness=max(38, size // 30), curve_type="arc")
            img = Image.alpha_composite(img, swoosh)
        else:
            swoosh = _make_swoosh_curve(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 210), thickness=max(40, size // 26), curve_type="wave")
            # Thin bright highlight for “metallic” feel (avoid big grey band).
            swoosh2 = _make_swoosh_curve(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 150), thickness=max(12, size // 90), curve_type="arc", flip=True)
            img = Image.alpha_composite(img, swoosh)
            img = Image.alpha_composite(img, swoosh2)
        
    elif shape_style == "blocks":
        block_style = "angular" if style_type in ("sharp", "aggressive", "racing") else "split" if style_type == "horizontal" else "corner"
        blocks = _make_geometric_blocks(size, color1=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 180), color2=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 130), block_style=block_style)
        img = Image.alpha_composite(img, blocks)
        
    elif shape_style == "minimal":
        # Minimal: Just 1-2 clean lines, position varies by palette
        angle = 0 if style_type == "horizontal" else angles[0] if style_type != "geometric" else 45
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 220), angle_deg=angle, thickness=0.015, position=positions[0], feather=1))
    
    elif shape_style == "mixed":
        # Mixed: Combine slash + swoosh uniquely per palette
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 180), angle_deg=angles[0], thickness=0.05, position=positions[0], feather=4))
        img = Image.alpha_composite(img, _make_swoosh_curve(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 140), thickness=max(25, size // 40), curve_type="arc"))
    
    elif shape_style == "kintsugi":
        # Kintsugi: gold “cracks” on dark panels (luxury vibe). Non-repeating, seed-driven.
        cracks = _make_kintsugi_cracks_layer(
            size,
            accent_rgb=accent_rgb,
            highlight_rgb=highlight_rgb,
            rng=rng,
        )
        img = Image.alpha_composite(img, cracks)
        # Add elegant racing-style stripes (thin highlight pinlines + translucent gold cores).
        # Keep these subtle so kintsugi remains the hero.
        try:
            # Gold cores
            img = Image.alpha_composite(
                img,
                _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 120), angle_deg=90, thickness=0.055, position=0.46, feather=0),
            )
            img = Image.alpha_composite(
                img,
                _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 120), angle_deg=90, thickness=0.055, position=0.54, feather=0),
            )
            # Cyan/Violet pinlines
            img = Image.alpha_composite(
                img,
                _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 200), angle_deg=90, thickness=0.010, position=0.41, feather=0),
            )
            img = Image.alpha_composite(
                img,
                _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 200), angle_deg=90, thickness=0.010, position=0.59, feather=0),
            )
            # One faint diagonal cutline to break the symmetry
            img = Image.alpha_composite(
                img,
                _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 55), angle_deg=25.0, thickness=0.020, position=float(rng.uniform(0.30, 0.70)), feather=2),
            )
        except Exception:
            pass
        # Add a faint “marble structure” so it reads less flat on large panels.
        try:
            topo = _make_topographic_lines_layer(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255), opacity=16)
            img = Image.alpha_composite(img, topo)
        except Exception:
            pass
        # NOTE: Intentionally no halftone/dust dots for kintsugi variants (user preference).
    
    elif shape_style == "circuit":
        # Circuit: PCB traces pattern
        # "Handcrafted" Mode: Use UV masks if available to place circuits on Sidepods (Zones 5/6)
        # and keep Hood (Zone 2) cleaner or simpler.
        
        circuit_layer = _make_circuit_trace_layer(
            size,
            accent_rgb=accent_rgb,
            highlight_rgb=highlight_rgb,
            rng=rng,
        )

        island_masks = feature_rects.get("island_masks", {}) if feature_rects else {}
        
        # Check if we have the critical masks for Stadium car
        has_side_mask = (5 in island_masks and 6 in island_masks)
        
        if has_side_mask:
            # Create a composite: Base is clean(er), Sides have dense circuit
            
            # 1. Hood/Top Layer: Cleaner, maybe just a few sparse traces or simple gradient
            # We'll use the generated 'img' (base effect) as the clean layer.
            # But let's add a VERY subtle tech grid to the hood so it's not empty.
            hood_layer = img
            try:
                hood_grid = _make_halftone_layer(size, color=highlight_rgb, opacity=15, dot_size=2, spacing=14)
                hood_layer = Image.alpha_composite(hood_layer, hood_grid)
            except Exception:
                pass

            # 2. Side Layer: The dense circuit pattern
            # We need to mask the circuit layer to ONLY the sides (plus maybe nose/rear if desired).
            # Let's combine islands 5, 6 (sides) and maybe 12, 13 (fenders).
            side_mask_img = Image.new("L", (size, size), 0)
            for mask_id in [5, 6, 12, 13, 21]: # Sides + Fenders + Rear
                if mask_id in island_masks:
                    side_mask_img.paste(255, (0, 0), island_masks[mask_id])
            
            # Blur mask slightly for smooth transition
            side_mask_img = side_mask_img.filter(ImageFilter.GaussianBlur(2))
            
            # Composite: Paste circuit_layer onto hood_layer using side_mask_img
            img = Image.composite(circuit_layer, hood_layer, side_mask_img)
            
            # 3. Add a gold pinstripe border between the zones?
            # Edge detection on the mask to find the boundary
            try:
                edges = side_mask_img.filter(ImageFilter.FIND_EDGES)
                edges = edges.point(lambda p: 255 if p > 20 else 0)
                edge_layer = Image.new("RGBA", (size, size), (0,0,0,0))
                # Gold pinstripe
                edge_layer.paste((accent_rgb[0], accent_rgb[1], accent_rgb[2], 255), (0,0), edges)
                img = Image.alpha_composite(img, edge_layer)
            except Exception:
                pass
                
        else:
            # Fallback: Apply to whole car if no masks
            img = Image.alpha_composite(img, circuit_layer)

    elif shape_style == "fusion":
        # Fusion: Mix multiple motifs, but keep hierarchy and negative space.
        try:
            tb = _make_tape_block_layer(size, color=secondary_rgb, opacity=int(rng.uniform(70, 140)), rng=rng)
            img = Image.alpha_composite(img, tb)
            # Crisp edge to increase perceived sharpness
            ta = tb.getchannel("A")
            edges = ta.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 14 else 0).filter(ImageFilter.GaussianBlur(radius=0.8))
            edge_layer = Image.new("RGBA", (size, size), (highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 0))
            edge_layer.putalpha(edges.point(lambda p: int(p * 0.42)))
            img = Image.alpha_composite(img, edge_layer)
        except Exception:
            pass

        try:
            swo = _make_swoosh_curve(
                size,
                color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 210),
                thickness=max(42, size // 28),
                curve_type=rng.choice(["wave", "arc"]),
            )
            img = Image.alpha_composite(img, swo)
            sa = swo.getchannel("A")
            sedge = sa.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 18 else 0).filter(ImageFilter.GaussianBlur(radius=0.7))
            rim = Image.new("RGBA", (size, size), (highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 0))
            rim.putalpha(sedge.point(lambda p: int(p * 0.35)))
            img = Image.alpha_composite(img, rim)
        except Exception:
            pass

        # Signature motif: either kintsugi cracks or circuit (not both).
        if rng.random() < 0.55:
            try:
                cracks = _make_kintsugi_cracks_layer(size, accent_rgb=accent_rgb, highlight_rgb=highlight_rgb, rng=rng)
                # Lower alpha so it doesn't overtake the livery.
                ca = cracks.getchannel("A").point(lambda p: int(p * 0.60))
                cracks.putalpha(ca)
                img = Image.alpha_composite(img, cracks)
            except Exception:
                pass
        else:
            try:
                cir = _make_circuit_trace_layer(size, accent_rgb=accent_rgb, highlight_rgb=highlight_rgb, rng=rng)
                # Keep circuit sparse by reducing alpha
                ca = cir.getchannel("A").point(lambda p: int(p * 0.55))
                cir.putalpha(ca)
                img = Image.alpha_composite(img, cir)
            except Exception:
                pass

        try:
            topo = _make_topographic_lines_layer(size, color=(255, 255, 255, 255), opacity=int(rng.uniform(10, 22)))
            img = Image.alpha_composite(img, topo)
        except Exception:
            pass

        # Rimlight from used mask if available (adds premium lighting vibe)
        try:
            if isinstance(feature_rects, dict):
                um = feature_rects.get("used_mask")
                if isinstance(um, Image.Image):
                    rim = _make_rimlight_from_mask(mask_l=um.convert("L"), color=highlight_rgb, opacity=int(rng.uniform(38, 75)), blur=float(rng.uniform(3.5, 7.0)))
                    img = Image.alpha_composite(img, rim)
        except Exception:
            pass

        try:
            hatch = _make_micro_hatch_layer(size, color=highlight_rgb, opacity=14, rng=rng)
            img = Image.alpha_composite(img, hatch)
        except Exception:
            pass
    
    elif shape_style == "heritage":
        # Heritage stripes (JPS-inspired): clean twin center stripes + pinstripes.
        # Works well for black+gold: bold but not noisy.
        # Main twin stripes
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 225), angle_deg=90, thickness=0.085, position=0.44, feather=0))
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 225), angle_deg=90, thickness=0.085, position=0.56, feather=0))
        # Thin bright pinstripes on the outside
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 210), angle_deg=90, thickness=0.012, position=0.38, feather=0))
        img = Image.alpha_composite(img, _make_diagonal_slash(size, color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 210), angle_deg=90, thickness=0.012, position=0.62, feather=0))
        # A subtle diagonal gold sweep to avoid “too simple”
        sweep = _make_diagonal_band_layer(
            size,
            color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 42),
            highlight_color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 28),
            band_width=0.22,
            angle_deg=-18.0,
            offset_x_frac=float(rng.uniform(-0.06, 0.06)),
            offset_y_frac=float(rng.uniform(-0.08, 0.05)),
        )
        img = Image.alpha_composite(img, sweep)
    
    # Layer 3: Subtle texture overlay (varies by base effect)
    if base_effect not in ("galaxy",):
        img = _apply_texture_overlay(img, texture_type="grain", opacity=0.06)
    
    # Layer 4: Pinstripe highlight
    if shape_style != "minimal":
        pinstripe = _make_diagonal_slash(
            size,
            color=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 80),
            angle_deg=55.0,
            thickness=0.006,
            position=0.22,
            feather=0,
        )
        img = Image.alpha_composite(img, pinstripe)

    # Layer 4.5: Clearcoat sweeps (adds “premium paint” depth; inspired by our Welk/TJ pro packs)
    # Keep subtle so it doesn't become stripey.
    try:
        base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
        # For very dark bases, large white sweeps read like “grey paint”.
        a0, a1 = ((6, 14) if base_sum < 70 else (10, 22))
        bw0, bw1 = ((0.18, 0.28) if base_sum < 70 else (0.22, 0.34))
        sweep1 = _make_diagonal_band_layer(
            size,
            color=(255, 255, 255, int(rng.uniform(a0, a1))),
            highlight_color=(0, 0, 0, 0),
            band_width=float(rng.uniform(bw0, bw1)),
            angle_deg=float(rng.uniform(-32, 32)),
            offset_x_frac=float(rng.uniform(-0.10, 0.10)),
            offset_y_frac=float(rng.uniform(-0.12, 0.08)),
        )
        img = Image.alpha_composite(img, sweep1)
        if rng.random() < 0.55:
            sweep2 = _make_diagonal_band_layer(
                size,
                color=(255, 255, 255, int(rng.uniform(max(4, a0 - 2), max(8, a1 - 6)))),
                highlight_color=(0, 0, 0, 0),
                band_width=float(rng.uniform(0.12, 0.20) if base_sum < 70 else rng.uniform(0.14, 0.22)),
                angle_deg=float(rng.uniform(-55, 55)),
                offset_x_frac=float(rng.uniform(-0.12, 0.12)),
                offset_y_frac=float(rng.uniform(-0.12, 0.10)),
            )
            img = Image.alpha_composite(img, sweep2)
    except Exception:
        pass

    # Layer 5: Micro details (subtle tech finish) – makes pro_swoosh_fade feel more premium.
    # Keep this restrained so it doesn't turn into another "pattern theme".
    if base_effect == "fade" and shape_style in ("swoosh", "mixed"):
        # Thin white tech pinlines to separate black/grey panels.
        tech1 = _make_diagonal_slash(
            size,
            color=(245, 245, 248, 52),
            angle_deg=-35.0,
            thickness=0.0035,
            position=0.78,
            feather=0,
        )
        tech2 = _make_diagonal_slash(
            size,
            color=(245, 245, 248, 34),
            angle_deg=65.0,
            thickness=0.0028,
            position=0.14,
            feather=0,
        )
        img = Image.alpha_composite(img, tech1)
        img = Image.alpha_composite(img, tech2)

        # Edge crispness pass (very low) to add "handmade" separation without brightening the whole car.
        try:
            edges = img.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 26 else 0)
            edges = edges.filter(ImageFilter.GaussianBlur(radius=max(1, size // 1700)))
            ec = (min(255, highlight_rgb[0] + 35), min(255, highlight_rgb[1] + 35), min(255, highlight_rgb[2] + 35))
            edge_layer = Image.new("RGBA", (size, size), (ec[0], ec[1], ec[2], 0))
            edge_layer.putalpha(edges.point(lambda p: int(p * 0.10)))
            img = Image.alpha_composite(img, edge_layer)
        except Exception:
            pass
    
    # For very dark bases (black/gold packs), the combination of carbon + secondary panels + clearcoat
    # can make the car read as “grey” under lighting. We gently crush near-black tones while leaving
    # bright accents (gold/cyan/violet) untouched.
    try:
        base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
        if base_sum <= 24 and np is not None:
            arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
            rgb = arr[..., :3]
            lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            thr = 85.0
            t = np.clip(lum / thr, 0.0, 1.0)
            # Darken low luminance more aggressively; keep mids mostly unchanged.
            s = (t ** 1.35)[..., None]
            rgb2 = np.clip(rgb * s, 0.0, 255.0)
            arr[..., :3] = rgb2
            img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    except Exception:
        pass

    # Global contrast punch for vivid neon looks (RGB only).
    try:
        # Avoid overdoing for already-bright bases; focus on dark wrap aesthetics.
        base_sum = int(base_rgb[0]) + int(base_rgb[1]) + int(base_rgb[2])
        if base_sum <= 90:
            a = img.getchannel("A")
            c = 1.24 if grade_contrast is None else float(grade_contrast)
            col = 1.08 if grade_color is None else float(grade_color)
            g = 0.93 if grade_gamma is None else float(grade_gamma)
            # Keep within sensible bounds to avoid posterization.
            c = max(1.00, min(1.60, c))
            col = max(0.90, min(1.35, col))
            g = max(0.70, min(1.20, g))
            img2 = _contrast_punch_rgb(img, contrast=c, color=col, gamma=g)
            img2.putalpha(a)
            img = img2
    except Exception:
        pass

    return img


def _make_kintsugi_cracks_layer(
    size: int,
    *,
    accent_rgb: Tuple[int, int, int],
    highlight_rgb: Tuple[int, int, int],
    rng: "random.Random",
) -> Image.Image:
    """
    Gold crack lines inspired by kintsugi.
    Designed to read well after DXT compression: crisp core + soft glow.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # Thickness tuned for 1024–2048.
    base_w = max(2, size // 560)
    glow_w = max(4, base_w + 3)
    n = int(rng.randint(12, 22))

    def rand_edge_point() -> Tuple[int, int]:
        side = rng.choice(["t", "b", "l", "r"])
        if side == "t":
            return (rng.randrange(0, size), 0)
        if side == "b":
            return (rng.randrange(0, size), size - 1)
        if side == "l":
            return (0, rng.randrange(0, size))
        return (size - 1, rng.randrange(0, size))

    for _ in range(n):
        a = rand_edge_point()
        b = rand_edge_point()
        # Build a jittered polyline from a->b.
        steps = rng.randint(4, 8)
        pts = []
        for i in range(steps + 1):
            t = i / max(1, steps)
            x = int(a[0] * (1 - t) + b[0] * t)
            y = int(a[1] * (1 - t) + b[1] * t)
            j = int(size * 0.03)
            x = int(max(0, min(size - 1, x + rng.randint(-j, j))))
            y = int(max(0, min(size - 1, y + rng.randint(-j, j))))
            pts.append((x, y))

        # Add a couple branch cracks from random midpoints (thin).
        if len(pts) >= 5 and rng.random() < 0.85:
            for _b in range(rng.randint(1, 2)):
                mid = pts[rng.randint(1, len(pts) - 2)]
                # short branch to a nearby target with jitter
                tx = int(max(0, min(size - 1, mid[0] + rng.randint(-int(size * 0.18), int(size * 0.18)))))
                ty = int(max(0, min(size - 1, mid[1] + rng.randint(-int(size * 0.18), int(size * 0.18)))))
                bpts = []
                bsteps = rng.randint(2, 5)
                for i in range(bsteps + 1):
                    t = i / max(1, bsteps)
                    x = int(mid[0] * (1 - t) + tx * t)
                    y = int(mid[1] * (1 - t) + ty * t)
                    j = int(size * 0.02)
                    x = int(max(0, min(size - 1, x + rng.randint(-j, j))))
                    y = int(max(0, min(size - 1, y + rng.randint(-j, j))))
                    bpts.append((x, y))
                bw = max(1, base_w - 1)
                d.line(bpts, fill=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 160), width=max(3, bw + 2), joint="curve")
                d.line(bpts, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 220), width=bw, joint="curve")
                d.line(bpts, fill=(255, 248, 230, 140), width=max(1, bw - 1), joint="curve")

        # Glow underlay
        d.line(pts, fill=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 180), width=glow_w, joint="curve")
        # Gold core
        d.line(pts, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 238), width=base_w, joint="curve")
        # Specular highlight (warm white) to make it read as metallic gold
        if base_w >= 2:
            d.line(pts, fill=(255, 248, 230, 160), width=max(1, base_w - 1), joint="curve")
        # NOTE: Intentionally no “sparkle dots” on cracks for kintsugi variants (user preference).

    # Soft glow pass (edge-only blur feel)
    glow = layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, size / 1200)))
    # Keep glow subtle
    ga = glow.getchannel("A").point(lambda p: int(p * 0.75))
    glow.putalpha(ga)
    out = Image.alpha_composite(glow, layer)
    return out


def _make_circuit_trace_layer(
    size: int,
    *,
    accent_rgb: Tuple[int, int, int],
    highlight_rgb: Tuple[int, int, int],
    rng: "random.Random",
) -> Image.Image:
    """
    PCB/circuit-trace motif. High density, 'Handcrafted' look with micro-details.
    """
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 1. Grid setup - finer grid for more detail
    grid_size = size // rng.randint(40, 60)  # ~50px on 2048
    cols = size // grid_size
    rows = size // grid_size

    # 2. Generate Logic Gates / Chips (Rectangles)
    # Place these first so traces can route around or to them
    chips = []
    num_chips = rng.randint(5, 12)
    for _ in range(num_chips):
        cx = rng.randint(2, cols - 4)
        cy = rng.randint(2, rows - 4)
        cw = rng.randint(1, 3)
        ch = rng.randint(1, 3)
        # Draw Chip Background (Darker Accent)
        x1, y1 = cx * grid_size, cy * grid_size
        x2, y2 = (cx + cw) * grid_size, (cy + ch) * grid_size
        
        # Chip body
        d.rectangle([x1, y1, x2, y2], fill=(accent_rgb[0]//2, accent_rgb[1]//2, accent_rgb[2]//2, 200))
        # Chip border
        d.rectangle([x1, y1, x2, y2], outline=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255), width=2)
        chips.append((cx, cy, cw, ch))

    # 3. Dense Traces
    num_traces = rng.randint(60, 100)
    
    # Helper to check if point is inside a chip
    def in_chip(tx, ty):
        for (cx, cy, cw, ch) in chips:
            if cx <= tx < cx + cw and cy <= ty < cy + ch:
                return True
        return False

    for _ in range(num_traces):
        # Start point (snap to grid)
        sx, sy = rng.randint(0, cols), rng.randint(0, rows)
        if in_chip(sx, sy): continue
        
        points = [(sx * grid_size, sy * grid_size)]
        curr_x, curr_y = sx, sy
        
        length = rng.randint(5, 20)
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]) # Cardinal directions
        
        for _step in range(length):
            # 20% chance to turn 90 degrees
            if rng.random() < 0.2:
                if direction[0] == 0: # Moving vertically
                    direction = rng.choice([(1, 0), (-1, 0)])
                else: # Moving horizontally
                    direction = rng.choice([(0, 1), (0, -1)])
            
            curr_x += direction[0]
            curr_y += direction[1]
            
            # Boundary check
            curr_x = max(0, min(cols, curr_x))
            curr_y = max(0, min(rows, curr_y))
            
            points.append((curr_x * grid_size, curr_y * grid_size))
            
            # Stop if we hit a chip (connect to it)
            if in_chip(curr_x, curr_y):
                break

        # Draw Trace
        width = max(2, size // 600)
        # Outer Glow (Wide, Low Alpha)
        d.line(points, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 40), width=width + 6)
        # Main Trace (Accent)
        d.line(points, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 180), width=width + 2)
        # Inner Core (Highlight)
        d.line(points, fill=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255), width=max(1, width - 2))

        # 4. Vias / Terminals at ends
        for px, py in [points[0], points[-1]]:
            r = width + 2
            d.ellipse([px-r, py-r, px+r, py+r], fill=(highlight_rgb[0], highlight_rgb[1], highlight_rgb[2], 255))
            d.ellipse([px-r//2, py-r//2, px+r//2, py+r//2], fill=(0, 0, 0, 255)) # Hole

    return layer


def generate_team_skin(
    *,
    size: int,
    team_name: str,
    tag: str,
    logo: Optional[Image.Image],
    base_color: Tuple[int, int, int, int],
    accent_color: Tuple[int, int, int, int],
    stripe_color: Tuple[int, int, int, int],
    style: str = "neon",
    feature_rects: Optional[Dict[str, object]] = None,
    nose_logo: bool = False,
    wing_top_text: Optional[str] = None,
    wing_bottom_text: Optional[str] = None,
    wing_sticker: Optional[Image.Image] = None,
    wing_sticker_fit: str = "contain",
    wing_sticker_scale: float = 1.0,
    wing_sticker_opacity: float = 1.0,
    palette_name: Optional[str] = None,
    seed: Optional[int] = None,
    wheel_rgb: Optional[Tuple[int, int, int]] = None,
    inspire_zip: Optional[Path] = None,
    inspire_source: str = "auto",
    inspire_strength: float = 0.78,
    logo_layout: str = "default",
    logo_plate: str = "auto",
    logo_scale: float = 1.0,
    sidepod_branding: bool = False,
    sidepod_tag_text: Optional[str] = None,
    sidepod_team_text: Optional[str] = None,
    sidepod_branding_scale: float = 0.70,
    sidepod_branding_mirror: str = "auto",
    sticker_images: Optional[List[Image.Image]] = None,
    sticker_count: int = 0,
    sticker_min_scale: float = 0.03,
    sticker_max_scale: float = 0.08,
    sticker_rotate: bool = True,
    sticker_mode: str = "grid",
    sticker_scope: str = "used",  # used | hero
    sponsor_images: Optional[List[Image.Image]] = None,
    sponsor_scale: float = 1.0,
    sponsor_opacity: float = 1.0,
    sponsor_slots: Optional[List[Tuple[int, str, str, int]]] = None,
    wheel_sticker: Optional[Image.Image] = None,
    wheel_sticker_scale: float = 0.48,
    wheel_sticker_scope: str = "caps",
    flag: Optional[str] = None,
    flag_location: str = "nose",
    flag_scale: float = 0.18,
    plate_text: Optional[str] = None,
    plate_scale: float = 0.92,
    finish_alpha: str = "auto",  # "opaque" | "neutral" | "auto"
    finish_neutral: int = 0x8E,
    finish_invert: bool = False,
    finish_design: str = "off",  # off | edges | sweep
    finish_design_strength: float = 0.35,
    mudguards: bool = True,
    mudguards_color: Optional[Tuple[int, int, int]] = None,
    mudguards_mode: str = "darken",
    mudguards_strength: float = 0.85,
    mudguards_feather: int = 3,
    spatial_aware: bool = False,
    spatial_accent_parts: Optional[List[str]] = None,
    spatial_secondary_parts: Optional[List[str]] = None,
    car_geometry: Optional["CarGeometry"] = None,  # type: ignore
    grade_contrast: Optional[float] = None,
    grade_color: Optional[float] = None,
    grade_gamma: Optional[float] = None,
    vignette_strength: Optional[int] = None,
) -> Image.Image:
    """
    Procedural "team skin" generator.

    Note: Without a Stadium UV template this isn't perfectly UV-aware, so we deliberately place
    the logo/tag in multiple regions and use a global pattern so something looks good on-car.
    """
    # Per-skin deterministic variation: avoid "same angles/patterns every time".
    # If seed is not provided, derive a stable one from identifying inputs.
    if seed is None:
        seed_src = f"{team_name}|{tag}|{style}|{palette_name or ''}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(int(seed))

    # Build RGB fully opaque (alpha=255). We'll optionally synthesize a TMNF-style "finish alpha"
    # at the end (so we never accidentally treat alpha as transparency during compositing).
    bg = Image.new("RGBA", (size, size), (base_color[0], base_color[1], base_color[2], 255))

    # PRO STYLES: Use the new layered design system
    if style.startswith("pro_"):
        # Parse style: "pro_slashes", "pro_slashes_fade", "pro_swoosh_splatter", etc.
        parts = style.replace("pro_", "").split("_")
        shape_style = parts[0]  # "slashes", "swoosh", "blocks", "minimal", "mixed", "kintsugi", "circuit", "heritage"
        
        # Determine base effect from suffix
        if len(parts) > 1:
            base_effect = parts[1]  # "fade", "splatter", "galaxy", "fluid", "carbon"
        else:
            base_effect = "gradient"  # default
        
        if palette_name == "auto_vibrant" and sample_vibrant_palette is not None:
            # Sample a novel high-contrast palette based on seed and use it as a custom pro palette.
            pal = sample_vibrant_palette(int(seed))
            custom_pal = {
                "base": pal.base,
                "secondary": pal.secondary,
                "accent": pal.accent,
                "highlight": pal.highlight,
                "text": (255, 255, 255),
            }
            img = _generate_pro_skin_layers(
                size,
                shape_style=shape_style,
                base_effect=base_effect,
                custom_palette=custom_pal,
                rng=rng,
                feature_rects=feature_rects,
                grade_contrast=grade_contrast,
                grade_color=grade_color,
                grade_gamma=grade_gamma,
                vignette_strength=vignette_strength,
            )
        elif palette_name and palette_name in PRO_COLOR_PALETTES:
            # Use the named palette
            img = _generate_pro_skin_layers(
                size,
                palette_name=palette_name,
                shape_style=shape_style,
                base_effect=base_effect,
                rng=rng,
                feature_rects=feature_rects,
                grade_contrast=grade_contrast,
                grade_color=grade_color,
                grade_gamma=grade_gamma,
                vignette_strength=vignette_strength,
            )
        else:
            # Build a custom palette from the provided colors
            base_sum = int(base_color[0]) + int(base_color[1]) + int(base_color[2])
            # For very dark bases, adding +25 makes “secondary” a noticeable grey, which reads as “not black”
            # in-game. Keep the secondary lift smaller so blacks stay black.
            sec_boost = 12 if base_sum < 70 else 25
            custom_pal = {
                "base": (base_color[0], base_color[1], base_color[2]),
                "secondary": (
                    min(255, base_color[0] + sec_boost),
                    min(255, base_color[1] + sec_boost),
                    min(255, base_color[2] + sec_boost),
                ),
                "accent": (accent_color[0], accent_color[1], accent_color[2]),
                "highlight": (stripe_color[0], stripe_color[1], stripe_color[2]),
                "text": (255, 255, 255),
            }
            
            img = _generate_pro_skin_layers(
                size,
                shape_style=shape_style,
                base_effect=base_effect,
                custom_palette=custom_pal,
                rng=rng,
                feature_rects=feature_rects,
                grade_contrast=grade_contrast,
                grade_color=grade_color,
                grade_gamma=grade_gamma,
                vignette_strength=vignette_strength,
            )
    
    # DIVERSE THEMES - Fundamentally different visual approaches
    elif style.startswith("theme_"):
        theme_type = style.replace("theme_", "")
        base_rgb = (base_color[0], base_color[1], base_color[2])
        accent_rgb = (accent_color[0], accent_color[1], accent_color[2])
        stripe_rgb = (stripe_color[0], stripe_color[1], stripe_color[2])
        
        if theme_type == "clean":
            # CLEAN: Solid base + single thin horizontal accent line
            img = Image.new("RGBA", (size, size), (*base_rgb, 255))
            # Just one clean accent stripe at 1/3 height
            line = _make_diagonal_slash(size, color=(*accent_rgb, 200), angle_deg=0, thickness=0.012, position=0.33, feather=1)
            img = Image.alpha_composite(img, line)
            
        elif theme_type == "twotone":
            # TWO-TONE: Hard diagonal split - two solid colors
            img = Image.new("RGBA", (size, size), (*base_rgb, 255))
            d = ImageDraw.Draw(img)
            # Diagonal split from top-left to bottom-right
            pts = [(0, 0), (size, 0), (size, size * 0.4), (0, size * 0.6)]
            d.polygon(pts, fill=(*accent_rgb, 255))
            # Thin separator line
            line = _make_diagonal_slash(size, color=(*stripe_rgb, 255), angle_deg=55, thickness=0.015, position=0.48, feather=0)
            img = Image.alpha_composite(img, line)
            
        elif theme_type == "gradient":
            # GRADIENT: Pure smooth gradient from base to accent - NO shapes
            img = _make_multi_stop_gradient(
                size,
                stops=[(0.0, base_rgb), (0.5, accent_rgb), (1.0, stripe_rgb)],
                angle_deg=-30.0,
                noise_strength=0.05,  # Very subtle noise
                quant_steps=0,  # No quantization - smooth
            )
            
        elif theme_type == "splatter":
            # SPLATTER: Heavy splatter texture - no slashes at all
            img = Image.new("RGBA", (size, size), (*base_rgb, 255))
            # Multiple splatter layers
            spl1 = _make_splatter_layer(size, color=accent_rgb, opacity=180, rng=rng)
            spl2 = _make_splatter_layer(size, color=stripe_rgb, opacity=120, rng=rng)
            spl3 = _make_splatter_layer(size, color=base_rgb, opacity=60, rng=rng)  # Base color splatter for depth
            img = Image.alpha_composite(img, spl1)
            img = Image.alpha_composite(img, spl2)
            img = Image.alpha_composite(img, spl3)
            
        elif theme_type == "racing":
            # RACING: Classic twin racing stripes down the middle
            img = Image.new("RGBA", (size, size), (*base_rgb, 255))
            # Two parallel vertical stripes
            stripe1 = _make_diagonal_slash(size, color=(*accent_rgb, 255), angle_deg=90, thickness=0.08, position=0.42, feather=0)
            stripe2 = _make_diagonal_slash(size, color=(*accent_rgb, 255), angle_deg=90, thickness=0.08, position=0.58, feather=0)
            # Thin border stripes
            border1 = _make_diagonal_slash(size, color=(*stripe_rgb, 200), angle_deg=90, thickness=0.015, position=0.38, feather=0)
            border2 = _make_diagonal_slash(size, color=(*stripe_rgb, 200), angle_deg=90, thickness=0.015, position=0.62, feather=0)
            img = Image.alpha_composite(img, stripe1)
            img = Image.alpha_composite(img, stripe2)
            img = Image.alpha_composite(img, border1)
            img = Image.alpha_composite(img, border2)
            
        elif theme_type == "stealth":
            # STEALTH: Nearly solid dark with very subtle accent
            darker = (max(0, base_rgb[0] - 10), max(0, base_rgb[1] - 10), max(0, base_rgb[2] - 10))
            img = _make_panel_gradient(size, top_color=base_rgb, bottom_color=darker)
            # Very subtle accent - almost invisible
            subtle = _make_diagonal_slash(size, color=(*accent_rgb, 40), angle_deg=20, thickness=0.25, position=0.5, feather=30)
            img = Image.alpha_composite(img, subtle)
            # Tiny highlight pinstripe
            pin = _make_diagonal_slash(size, color=(*stripe_rgb, 80), angle_deg=20, thickness=0.008, position=0.35, feather=0)
            img = Image.alpha_composite(img, pin)

        elif theme_type == "suminagashi" and generate_suminagashi is not None:
            # SUMINAGASHI: Japanese floating-ink marbling with concentric rings warped by a vector field.
            sumi_colors = [base_rgb, accent_rgb, stripe_rgb]
            # Add lighter/darker variants for richer banding
            sumi_colors.append(tuple(min(255, c + 40) for c in accent_rgb))
            sumi_colors.append(tuple(max(0, c - 30) for c in base_rgb))
            img = generate_suminagashi(
                size,
                colors=sumi_colors,
                seed=int(rng.randint(0, 2**31)),
                num_drops=int(rng.randint(10, 16)),
                rings_per_drop=int(rng.randint(14, 22)),
                warp_strength=float(rng.uniform(0.8, 1.4)),
            )
            # Subtle vignette to ground the design
            try:
                img = Image.alpha_composite(img, _vignette_layer(size, strength=60, rng=rng))
            except Exception:
                pass

        elif theme_type == "moire" and generate_moire_interference is not None:
            # MOIRE: Overlapping radial grids create emergent optical interference waves.
            img = generate_moire_interference(
                size,
                line_color=accent_rgb,
                accent_color=stripe_rgb,
                bg_color=base_rgb,
                seed=int(rng.randint(0, 2**31)),
                num_grids=int(rng.choice([2, 3])),
                line_density=int(rng.randint(60, 100)),
            )
            # Light vignette
            try:
                img = Image.alpha_composite(img, _vignette_layer(size, strength=50, rng=rng))
            except Exception:
                pass

        elif theme_type == "palimpsest" and generate_palimpsest is not None:
            # PALIMPSEST: Layered urban abstraction (city grid + dot grid + spray arcs + veil).
            img = generate_palimpsest(
                size,
                base_color=base_rgb,
                spray_colors=[accent_rgb, stripe_rgb],
                grid_color=tuple(min(255, c + 60) for c in base_rgb),
                veil_warm=accent_rgb,
                veil_cool=stripe_rgb,
                seed=int(rng.randint(0, 2**31)),
            )
            # Soft vignette
            try:
                img = Image.alpha_composite(img, _vignette_layer(size, strength=55, rng=rng))
            except Exception:
                pass

        else:
            img = bg
            
    # Base paint
    elif style == "solid":
        img = bg
    elif style in ("fade", "fade_splatter"):
        # Use base -> accent -> stripe as 3-stop gradient (varied per skin).
        mid = float(rng.uniform(0.42, 0.68))
        ang = float(-18.0 + rng.uniform(-22.0, 22.0))
        noise_strength = float(rng.uniform(0.18, 0.28))
        quant_steps = int(rng.choice([12, 14, 16, 18, 20, 22]))
        img = _make_multi_stop_gradient(
            size,
            stops=[
                (0.0, (base_color[0], base_color[1], base_color[2])),
                (mid, (accent_color[0], accent_color[1], accent_color[2])),
                (1.0, (stripe_color[0], stripe_color[1], stripe_color[2])),
            ],
            angle_deg=ang,
            # Kacky-like: slightly noisy + quantized so it reads like a handcrafted/dithered fade, not a flat gradient.
            noise_strength=noise_strength,
            quant_steps=quant_steps,
        )
    else:
        # Directional base paint (varied per skin). This is a big contributor to "handmade" feel.
        base_rgb = (base_color[0], base_color[1], base_color[2])
        accent_rgb = (accent_color[0], accent_color[1], accent_color[2])
        stripe_rgb = (stripe_color[0], stripe_color[1], stripe_color[2])
        base_dark = (
            max(0, base_rgb[0] - int(rng.uniform(12, 38))),
            max(0, base_rgb[1] - int(rng.uniform(12, 38))),
            max(0, base_rgb[2] - int(rng.uniform(12, 38))),
        )
        ang = float(rng.uniform(-60.0, 60.0))
        p1 = float(rng.uniform(0.30, 0.55))
        p2 = float(rng.uniform(0.55, 0.82))
        if rng.random() < 0.55:
            # Base -> accent -> dark base (subtle sporty)
            stops = [(0.0, base_rgb), (p2, accent_rgb), (1.0, base_dark)]
        else:
            # Base -> accent -> stripe (higher contrast)
            stops = [(0.0, base_rgb), (p1, accent_rgb), (1.0, stripe_rgb)]
        img = _make_multi_stop_gradient(
            size,
            stops=stops,
            angle_deg=ang,
            noise_strength=0.0,
            quant_steps=0,
        )

    # Optional: inspired-by-example composition (handmade structure)
    # We blend it primarily onto “hero” islands (nose + sidepods) when island masks are available.
    # IMPORTANT: If we have island masks (standard Stadium), do NOT blend globally here.
    # Global blending can imprint large text/logos from the inspiration source onto small islands
    # (e.g. wheel caps), which then show up in-game. In that case we defer to the hero-only blend
    # later (nose + sidepods).
    island_masks_for_inspire: Dict[int, Image.Image] = {}
    if isinstance(feature_rects, dict):
        island_masks_for_inspire = feature_rects.get("island_masks", {}) if isinstance(feature_rects.get("island_masks", {}), dict) else {}
    has_island_masks = bool(island_masks_for_inspire) and (np is not None)

    if inspire_zip is not None and (not has_island_masks):
        try:
            inspired = _build_inspired_layer_from_zip(
                inspire_zip_path=inspire_zip,
                out_size=(size, size),
                source=inspire_source,
                c0=(max(0, base_color[0] - 25), max(0, base_color[1] - 25), max(0, base_color[2] - 25)),
                c1=(accent_color[0], accent_color[1], accent_color[2]),
                c2=(stripe_color[0], stripe_color[1], stripe_color[2]),
                rng=rng,
            )
            strength = float(max(0.0, min(1.0, inspire_strength)))

            # Default: blend everywhere; if we later have island masks, we’ll clip to hero areas below.
            if strength > 0.0:
                # Simple lerp in RGB space
                if np is not None:
                    a0 = np.asarray(img.convert("RGBA"), dtype=np.float32)
                    a1 = np.asarray(inspired.convert("RGBA"), dtype=np.float32)
                    mix = a0 * (1.0 - strength) + a1 * strength
                    img = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
                else:
                    img = Image.blend(img.convert("RGBA"), inspired.convert("RGBA"), strength)
        except Exception:
            # If inspire fails, proceed with procedural skin.
            pass

    # Texture styles (base material)
    if style == "carbon":
        carbon = _make_carbon_fiber_overlay(size, color=(255, 255, 255, 30))
        img = Image.alpha_composite(img, carbon)
    elif style == "stone":
        # Noisy "rock" texture: effect_noise is fast and good enough.
        noise = Image.effect_noise((size, size), 18).filter(ImageFilter.GaussianBlur(radius=1.2))
        rock = ImageOps.colorize(noise, black=(10, 10, 12), white=(70, 70, 78)).convert("RGBA")
        rock.putalpha(70)
        img = Image.alpha_composite(img, rock)
    elif style == "fluid":
        # Fluid base sheen
        sheen = _make_fluid_sheen_layer(
            size,
            a=(max(0, base_color[0] - 10), max(0, base_color[1] - 10), max(0, base_color[2] - 10)),
            b=(accent_color[0], accent_color[1], accent_color[2]),
            opacity=55,
        )
        img = Image.alpha_composite(img, sheen)
    elif style == "galaxy":
        # Galaxy base: nebula + stars.
        neb1 = _make_galaxy_nebula_layer(
            size,
            dark=(10, 8, 20),
            bright=(accent_color[0], accent_color[1], accent_color[2]),
            opacity=70,
        )
        neb2 = _make_galaxy_nebula_layer(
            size,
            dark=(5, 5, 10),
            bright=(stripe_color[0], stripe_color[1], stripe_color[2]),
            opacity=55,
        )
        stars = _make_starfield_layer(size, density=0.0014, color=(240, 240, 255), max_alpha=90)
        img = Image.alpha_composite(img, neb1)
        img = Image.alpha_composite(img, neb2)
        img = Image.alpha_composite(img, stars)
    elif style == "camo":
        # Camo wrap: 3-4 blob colors derived from your palette.
        cols = [
            (max(0, base_color[0] - 42), max(0, base_color[1] - 42), max(0, base_color[2] - 42)),
            (base_color[0], base_color[1], base_color[2]),
            (min(255, accent_color[0] + 10), min(255, accent_color[1] + 10), min(255, accent_color[2] + 10)),
            (min(255, stripe_color[0] + 12), min(255, stripe_color[1] + 12), min(255, stripe_color[2] + 12)),
        ]
        camo = _make_camo_layer(size, colors=cols, opacity=int(rng.uniform(185, 235)), rng=rng)
        img = Image.alpha_composite(img, camo)
        # Add one bold accent sweep so camo reads as a “designed wrap” (angle/pos varies per skin).
        sweep = _make_diagonal_band_layer(
            size,
            color=(accent_color[0], accent_color[1], accent_color[2], int(rng.uniform(22, 44))),
            highlight_color=(stripe_color[0], stripe_color[1], stripe_color[2], int(rng.uniform(28, 58))),
            band_width=float(rng.uniform(0.18, 0.32)),
            angle_deg=float(rng.uniform(-55, 55)),
            offset_x_frac=float(rng.uniform(-0.10, 0.10)),
            offset_y_frac=float(rng.uniform(-0.14, 0.08)),
        )
        img = Image.alpha_composite(img, sweep)
        img = _apply_texture_overlay(img, texture_type="grain", opacity=float(rng.uniform(0.05, 0.11)))
    elif style == "halftone":
        # Halftone overlay on top of base paint.
        dots = _make_halftone_layer(
            size,
            color=(stripe_color[0], stripe_color[1], stripe_color[2]),
            opacity=int(rng.uniform(70, 140)),
            rng=rng,
        )
        dots2 = _make_halftone_layer(
            size,
            color=(accent_color[0], accent_color[1], accent_color[2]),
            opacity=int(rng.uniform(45, 110)),
            rng=rng,
        )
        img = Image.alpha_composite(img, dots)
        img = Image.alpha_composite(img, dots2)
        img = _apply_texture_overlay(img, texture_type="grain", opacity=float(rng.uniform(0.04, 0.09)))
    elif style == "shards":
        # Shattered polygonal shards overlay.
        cols = [
            (accent_color[0], accent_color[1], accent_color[2]),
            (stripe_color[0], stripe_color[1], stripe_color[2]),
            (min(255, base_color[0] + 20), min(255, base_color[1] + 20), min(255, base_color[2] + 20)),
        ]
        shards = _make_shards_layer(size, colors=cols, opacity=int(rng.uniform(120, 200)), rng=rng)
        img = Image.alpha_composite(img, shards)
        # Add a subtle edge sheen
        sheen = _make_diagonal_band_layer(
            size,
            color=(255, 255, 255, int(rng.uniform(10, 22))),
            highlight_color=(0, 0, 0, 0),
            band_width=float(rng.uniform(0.16, 0.28)),
            angle_deg=float(rng.uniform(-55, 55)),
            offset_x_frac=float(rng.uniform(-0.10, 0.10)),
            offset_y_frac=float(rng.uniform(-0.10, 0.10)),
        )
        img = Image.alpha_composite(img, sheen)
        img = _apply_texture_overlay(img, texture_type="grain", opacity=float(rng.uniform(0.05, 0.10)))
    elif style == "brushed":
        # Brushed metal finish over base paint (subtle).
        brushed = _make_brushed_metal_layer(
            size,
            color=(min(255, base_color[0] + 40), min(255, base_color[1] + 40), min(255, base_color[2] + 40)),
            opacity=int(rng.uniform(130, 205)),
            rng=rng,
        )
        img = Image.alpha_composite(img, brushed)
        # Second pass for depth (different direction)
        brushed2 = _make_brushed_metal_layer(
            size,
            color=(min(255, base_color[0] + 28), min(255, base_color[1] + 28), min(255, base_color[2] + 28)),
            opacity=int(rng.uniform(55, 120)),
            rng=rng,
        )
        img = Image.alpha_composite(img, brushed2)
        # Add a clearer highlight sweep
        pin = _make_diagonal_band_layer(
            size,
            color=(stripe_color[0], stripe_color[1], stripe_color[2], int(rng.uniform(22, 46))),
            highlight_color=(255, 255, 255, int(rng.uniform(14, 30))),
            band_width=float(rng.uniform(0.12, 0.26)),
            angle_deg=float(rng.uniform(-40, 40)),
            offset_x_frac=float(rng.uniform(-0.10, 0.10)),
            offset_y_frac=float(rng.uniform(-0.12, 0.08)),
        )
        img = Image.alpha_composite(img, pin)
    elif style == "glitch":
        # Glitch accents (works well with neon palettes).
        gl = _make_glitch_layer(
            size,
            color=(stripe_color[0], stripe_color[1], stripe_color[2]),
            opacity=int(rng.uniform(120, 210)),
            rng=rng,
        )
        gl2 = _make_glitch_layer(
            size,
            color=(accent_color[0], accent_color[1], accent_color[2]),
            opacity=int(rng.uniform(90, 170)),
            rng=rng,
        )
        img = Image.alpha_composite(img, gl)
        img = Image.alpha_composite(img, gl2)
        # A small white glitch pass to make the effect pop on darker paints.
        gl3 = _make_glitch_layer(
            size,
            color=(255, 255, 255),
            opacity=int(rng.uniform(35, 85)),
            rng=rng,
        )
        img = Image.alpha_composite(img, gl3)
    elif style in ("splatter", "fade_splatter"):
        # High contrast splatter on top of base.
        primary = (
            (accent_color[0], accent_color[1], accent_color[2])
            if style == "splatter"
            else (stripe_color[0], stripe_color[1], stripe_color[2])
        )
        spl = _make_splatter_layer(size, color=primary, opacity=165, rng=rng)
        img = Image.alpha_composite(img, spl)
        if style == "fade_splatter":
            # Secondary darker splatter for depth (subtle).
            spl2 = _make_splatter_layer(size, color=(accent_color[0], accent_color[1], accent_color[2]), opacity=55, dots=False, rng=rng)
            img = Image.alpha_composite(img, spl2)

    # Big diagonal band (key: makes non-pro variants look distinct, not stripey).
    # Pro styles already have a curated layered system, so the extra band can wash out the design.
    if (not style.startswith("pro_")) and style not in ("galaxy", "solid", "splatter", "fade", "fade_splatter"):
        # Vary band placement/angle per skin so designs don't all share the same diagonal.
        base_angle = (-12.0 if style == "topo" else -18.0)
        angle = base_angle + rng.uniform(-16.0, 16.0)
        width = (0.24 if style == "carbon" else 0.30) * rng.uniform(0.78, 1.22)
        width = max(0.18, min(0.42, width))
        offx = rng.uniform(-0.12, 0.12)
        offy = rng.uniform(-0.18, 0.10)

        band = _make_diagonal_band_layer(
            size,
            color=(accent_color[0], accent_color[1], accent_color[2], 90),
            highlight_color=(stripe_color[0], stripe_color[1], stripe_color[2], 110),
            band_width=width,
            angle_deg=angle,
            offset_x_frac=offx,
            offset_y_frac=offy,
        )
        img = Image.alpha_composite(img, band)

        # Occasionally add a secondary faint band for a more "handmade layered" feel.
        if rng.random() < 0.33:
            band2 = _make_diagonal_band_layer(
                size,
                color=(stripe_color[0], stripe_color[1], stripe_color[2], 26),
                highlight_color=(255, 255, 255, 18),
                band_width=max(0.14, width * rng.uniform(0.45, 0.70)),
                angle_deg=angle + rng.uniform(-32.0, 32.0),
                offset_x_frac=offx + rng.uniform(-0.18, 0.18),
                offset_y_frac=offy + rng.uniform(-0.18, 0.18),
            )
            img = Image.alpha_composite(img, band2)
    elif style in ("fade", "fade_splatter"):
        # Very subtle highlight band on fade (still varied per skin).
        angle = -18.0 + rng.uniform(-12.0, 12.0)
        width = max(0.16, min(0.30, 0.22 * rng.uniform(0.85, 1.25)))
        band = _make_diagonal_band_layer(
            size,
            color=(255, 255, 255, 16),
            highlight_color=(0, 0, 0, 0),
            band_width=width,
            angle_deg=angle,
            offset_x_frac=rng.uniform(-0.10, 0.10),
            offset_y_frac=rng.uniform(-0.14, 0.08),
        )
        img = Image.alpha_composite(img, band)

    # Structure layers (topo + lava cracks)
    if style in ("neon", "topo", "stone", "galaxy"):
        topo_opacity = int(rng.uniform(18, 42)) if style != "topo" else int(rng.uniform(26, 58))
        topo = _make_topographic_lines_layer(size, color=(255, 255, 255, 255), opacity=topo_opacity)
        img = Image.alpha_composite(img, topo)

    if style in ("stone", "carbon", "fluid", "neon", "topo", "galaxy"):
        lava_opacity = int(rng.uniform(55, 95)) if style not in ("carbon",) else int(rng.uniform(40, 75))
        lava = _make_lava_cracks_layer(
            size,
            glow_color=(accent_color[0], accent_color[1], accent_color[2], 255),
            opacity=lava_opacity,
        )
        img = Image.alpha_composite(img, lava)

    # Accent lines (VERY subtle now; avoid "all stripes" look)
    stripe_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if style == "neon":
        stripe_width = max(3, size // 240)
        _draw_diagonal_stripes(
            stripe_layer,
            stripe_color=(stripe_color[0], stripe_color[1], stripe_color[2], min(28, stripe_color[3])),
            stripe_width=stripe_width,
        )
        img = Image.alpha_composite(img, stripe_layer)

    # If we have UV island masks, we can keep the “handmade” composition purposeful:
    # concentrate busy patterns on key panels (sidepods/nose), keep others cleaner.
    island_masks: Dict[int, Image.Image] = {}
    island_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    if isinstance(feature_rects, dict):
        island_masks = feature_rects.get("island_masks", {}) if isinstance(feature_rects.get("island_masks", {}), dict) else {}
        island_bboxes = feature_rects.get("island_bboxes", {}) if isinstance(feature_rects.get("island_bboxes", {}), dict) else {}

    # If we used an inspired layer, re-apply it (stronger) only to hero islands when masks exist.
    if inspire_zip is not None and island_masks and np is not None:
        try:
            inspired = _build_inspired_layer_from_zip(
                inspire_zip_path=inspire_zip,
                out_size=(size, size),
                source=inspire_source,
                c0=(max(0, base_color[0] - 25), max(0, base_color[1] - 25), max(0, base_color[2] - 25)),
                c1=(accent_color[0], accent_color[1], accent_color[2]),
                c2=(stripe_color[0], stripe_color[1], stripe_color[2]),
                rng=rng,
            )
            strength = float(max(0.0, min(1.0, inspire_strength)))
            if strength > 0.0:
                # Hero mask: nose + sidepods (2,5,6). Exclude wing (8) and wheel caps.
                hero = Image.new("L", (size, size), 0)
                for rid in [2, 5, 6]:
                    m = island_masks.get(rid)
                    if m is None:
                        continue
                    hero = ImageChops.lighter(hero, m.convert("L"))

                m_arr = (np.asarray(hero, dtype=np.float32) / 255.0)[..., None]
                a0 = np.asarray(img.convert("RGBA"), dtype=np.float32)
                a1 = np.asarray(inspired.convert("RGBA"), dtype=np.float32)
                mix = a0 * (1.0 - strength * m_arr) + a1 * (strength * m_arr)
                img = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
        except Exception:
            pass

    # NOTE: we intentionally do NOT spray the tag across the body texture. It tends to look like
    # unwanted “ghost text” on UV seams and reads as clutter in-game. Tag is reserved for wing plate.

    # Sidepod branding using UV island ids (from your UV debug screenshots):
    # - nose/top = 2
    # - sidepods = 5 and 6
    # - wheel covers = 10/12/13/21
    # We place logos ONLY on those islands so they won't overlap rear wing or helmet.
    if island_masks:
        # Decide where logos go.
        layout = (logo_layout or "default").strip().lower()
        plate_mode = (logo_plate or "auto").strip().lower()
        if plate_mode not in ("auto", "on", "off"):
            plate_mode = "auto"
        # Sidepods: readable by default (plate on). Nose/front: blend by default (plate off).
        plate_sidepods = plate_mode in ("auto", "on")
        plate_nose = plate_mode == "on"
        try:
            logo_scale_f = float(logo_scale)
        except Exception:
            logo_scale_f = 1.0
        logo_scale_f = max(0.15, min(3.0, logo_scale_f))
        if layout == "default":
            do_sidepods = True
            do_front = False
            do_cockpit = False
            do_nose_small = bool(nose_logo)
        else:
            do_sidepods = ("sidepods" in layout) or (layout == "all")
            do_front = ("front" in layout) or (layout == "all")
            do_cockpit = ("cockpit" in layout) or (layout == "all")
            do_nose_small = False

        if logo is not None and island_bboxes:
            # Make logo consistently readable: choose a contrasting plate + outline + glow.
            logo_lum = _mean_luminance_rgba(logo)
            logo_is_dark = logo_lum < 110.0

            # Plate behind logo
            if logo_is_dark:
                plate_rgb = (245, 245, 245)
                plate_alpha = 150
                outline_color = (0, 0, 0, 220)
            else:
                plate_rgb = (0, 0, 0)
                plate_alpha = 120
                outline_color = (255, 255, 255, 225)

            # Glow color: ensure it contrasts the plate (fallback to black/white when needed).
            sr, sg, sb = stripe_color[0], stripe_color[1], stripe_color[2]
            stripe_lum = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb
            plate_lum = 0.2126 * plate_rgb[0] + 0.7152 * plate_rgb[1] + 0.0722 * plate_rgb[2]
            if abs(stripe_lum - plate_lum) < 45:
                glow = (0, 0, 0, 160) if plate_lum > 140 else (255, 255, 255, 180)
            else:
                glow = (sr, sg, sb, 170)

            # Sidepods
            if do_sidepods:
                for island_id, island_scale in [(5, 0.48), (6, 0.48)]:
                    m = island_masks.get(island_id)
                    bb = island_bboxes.get(island_id)
                    if m is None or bb is None:
                        continue
                    x0, y0, x1, y1 = bb
                    rw = max(1, x1 - x0)
                    rh = max(1, y1 - y0)

                    # Optional plate behind logo (keeps logo readable)
                    if plate_sidepods:
                        panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        pd = ImageDraw.Draw(panel)
                        inset = max(10, min(rw, rh) // 14)
                        pd.rectangle(
                            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                            fill=(plate_rgb[0], plate_rgb[1], plate_rgb[2], plate_alpha),
                        )
                        # Border (very subtle, but helps decal feel)
                        border_w = max(2, min(rw, rh) // 140)
                        pd.rectangle(
                            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                            outline=(glow[0], glow[1], glow[2], 140),
                            width=border_w,
                        )
                        # Clip panel to island
                        pa = panel.getchannel("A")
                        pa = ImageChops.multiply(pa, m)
                        panel.putalpha(pa)
                        img = Image.alpha_composite(img, panel)

                    # Logo layer centered in bbox
                    logo_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    target_size = int(min(rw, rh) * island_scale * logo_scale_f)
                    _composite_logo_decal(
                        logo_layer,
                        logo=logo,
                        xy=(x0 + (rw - target_size) // 2, y0 + (rh - target_size) // 2),
                        size_px=target_size,
                        glow_color=glow,
                        glow_radius=max(4, size // 180),
                        outline_color=outline_color,
                        outline_px=max(2, size // 1024),
                    )
                    la = logo_layer.getchannel("A")
                    la = ImageChops.multiply(la, m)
                    logo_layer.putalpha(la)
                    img = Image.alpha_composite(img, logo_layer)

            # Nose/top island (rank 2): two placements
            m2 = island_masks.get(2)
            bb2 = island_bboxes.get(2)
            if m2 is not None and bb2 is not None and (do_front or do_cockpit):
                x0, y0, x1, y1 = bb2
                rw = max(1, x1 - x0)
                rh = max(1, y1 - y0)

                def _stamp_at(
                    cx: int,
                    cy: int,
                    target_size: int,
                    *,
                    rotate_deg: float = 0.0,
                    with_plate: bool = False,
                ) -> None:
                    # Clamp to bbox area to avoid wild placements
                    cx = max(x0 + 8, min(x1 - 8, cx))
                    cy = max(y0 + 8, min(y1 - 8, cy))
                    target_size = max(24, min(int(min(rw, rh) * 0.26), target_size))

                    img2 = img
                    if with_plate:
                        nx = cx - target_size // 2
                        ny = cy - target_size // 2
                        pad = max(3, target_size // 10)
                        nose_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        pd = ImageDraw.Draw(nose_layer)
                        pd.rectangle(
                            (nx - pad, ny - pad, nx + target_size + pad, ny + target_size + pad),
                            fill=(plate_rgb[0], plate_rgb[1], plate_rgb[2], int(plate_alpha * 0.72)),
                            outline=(glow[0], glow[1], glow[2], 150),
                            width=max(2, target_size // 64),
                        )
                        na = nose_layer.getchannel("A")
                        na = ImageChops.multiply(na, m2)
                        nose_layer.putalpha(na)
                        img2 = Image.alpha_composite(img, nose_layer)

                    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    _composite_logo_decal(
                        tmp,
                        logo=logo,
                        xy=(cx, cy),
                        size_px=target_size,
                        glow_color=glow,
                        glow_radius=max(3, size // 220),
                        outline_color=outline_color,
                        outline_px=max(1, size // 1400),
                        rotate_deg=rotate_deg,
                        anchor="center",
                        enable_glow=False,
                    )
                    ta = tmp.getchannel("A")
                    ta = ImageChops.multiply(ta, m2)
                    tmp.putalpha(ta)

                    img3 = Image.alpha_composite(img2, tmp)
                    img.paste(img3)

                if do_front:
                    cx, cy, ts, rot = _estimate_nose_logo_placement(m2, mode="front", rng=rng)
                    # Smaller + blends by default (no square plate).
                    _stamp_at(cx, cy, int(ts * 0.40 * logo_scale_f), rotate_deg=rot, with_plate=plate_nose)
                if do_cockpit:
                    cx, cy, ts, rot = _estimate_nose_logo_placement(m2, mode="cockpit", rng=rng)
                    _stamp_at(cx, cy, int(ts * 0.36 * logo_scale_f), rotate_deg=rot, with_plate=plate_nose)

            # Optional small logo on nose (legacy --nose-logo behavior)
            if logo is not None and island_bboxes and do_nose_small:
                m2s = island_masks.get(2)
                bb2s = island_bboxes.get(2)
                if m2s is not None and bb2s is not None:
                    x0, y0, x1, y1 = bb2s
                    rw = max(1, x1 - x0)
                    rh = max(1, y1 - y0)
                    target_size = int(min(rw, rh) * 0.22)
                    nx = x0 + int(rw * 0.68)
                    ny = y0 + int(rh * 0.16)
                    pad = max(3, target_size // 10)
                    # Rotate based on island direction so it isn't sideways.
                    _, _, _, rot = _estimate_nose_logo_placement(m2s, mode="front", rng=rng)

                    img2 = img
                    if plate_nose:
                        nose_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        pd = ImageDraw.Draw(nose_layer)
                        pd.rectangle(
                            (nx - target_size // 2 - pad, ny - target_size // 2 - pad, nx + target_size // 2 + pad, ny + target_size // 2 + pad),
                            fill=(plate_rgb[0], plate_rgb[1], plate_rgb[2], int(plate_alpha * 0.65)),
                            outline=(glow[0], glow[1], glow[2], 120),
                            width=max(2, target_size // 64),
                        )
                        pa = nose_layer.getchannel("A")
                        pa = ImageChops.multiply(pa, m2s)
                        nose_layer.putalpha(pa)
                        img2 = Image.alpha_composite(img, nose_layer)

                    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    _composite_logo_decal(
                        tmp,
                        logo=logo,
                        xy=(nx, ny),
                        size_px=target_size,
                        glow_color=glow,
                        glow_radius=max(3, size // 220),
                        outline_color=outline_color,
                        outline_px=max(2, size // 1400),
                        rotate_deg=rot,
                        anchor="center",
                        enable_glow=False,
                    )
                    ta = tmp.getchannel("A")
                    ta = ImageChops.multiply(ta, m2s)
                    tmp.putalpha(ta)
                    img = Image.alpha_composite(img2, tmp)

        # Optional flag decal (e.g. PL) – clipped to UV islands so it doesn't bleed onto other parts.
        if flag and island_bboxes:
            flag_code = str(flag).strip().lower()
            loc = str(flag_location or "nose").strip().lower()
            if loc not in ("nose", "sidepods"):
                loc = "nose"
            try:
                fs = float(flag_scale)
            except Exception:
                fs = 0.18
            fs = max(0.08, min(0.45, fs))

            def _stamp_flag_on_island(
                island_id: int,
                *,
                cx: Optional[int] = None,
                cy: Optional[int] = None,
                rotate_deg: float = 0.0,
                mirror_lr: bool = False,
            ) -> None:
                nonlocal img
                m = island_masks.get(island_id)
                bb = island_bboxes.get(island_id)
                if m is None or bb is None:
                    return
                x0, y0, x1, y1 = bb
                rw = max(1, x1 - x0)
                rh = max(1, y1 - y0)

                # Keep an ~8:5 ratio (common flag proportion) and ensure it stays inside bbox.
                fh = int(min(rw, rh) * fs)
                fw = int(fh * 1.6)
                fw = max(24, min(max(24, rw - 10), fw))
                fh = max(18, min(max(18, rh - 10), fh))

                if flag_code in ("pl", "poland", "polska", "pol"):
                    decal = _make_polish_flag_rgba(w=fw, h=fh)
                else:
                    return

                if mirror_lr:
                    decal = decal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if abs(float(rotate_deg)) > 0.01:
                    decal = decal.rotate(float(rotate_deg), expand=True, resample=Image.Resampling.BICUBIC)

                if cx is None or cy is None:
                    cx = x0 + int(rw * 0.76)
                    cy = y0 + int(rh * 0.34)
                px = int(cx - decal.size[0] // 2)
                py = int(cy - decal.size[1] // 2)

                layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer.alpha_composite(decal, (px, py))
                a = layer.getchannel("A")
                a = ImageChops.multiply(a, m.convert("L"))
                layer.putalpha(a)
                img = Image.alpha_composite(img, layer)

            if loc == "sidepods":
                _stamp_flag_on_island(5, mirror_lr=False)
                _stamp_flag_on_island(6, mirror_lr=True)
            else:
                m2f = island_masks.get(2)
                if m2f is not None:
                    # If logo already occupies the nose regions, bias flag away from it.
                    layout2 = (logo_layout or "default").strip().lower()
                    if "front" in layout2 and "cockpit" in layout2:
                        _stamp_flag_on_island(5, mirror_lr=False)
                        _stamp_flag_on_island(6, mirror_lr=True)
                    else:
                        flag_mode = "cockpit" if ("front" in layout2) else ("front" if ("cockpit" in layout2) else "cockpit")
                        cx, cy, _, rot = _estimate_nose_logo_placement(m2f, mode=flag_mode, rng=rng)
                        _stamp_flag_on_island(2, cx=cx, cy=cy, rotate_deg=rot, mirror_lr=False)
                else:
                    _stamp_flag_on_island(2, rotate_deg=0.0, mirror_lr=False)

        # Optional number plate text (auto-detected plate island(s) from base zip UV islands).
        if plate_text and island_bboxes:
            txt = str(plate_text).strip()
            if txt:
                try:
                    ps = float(plate_scale)
                except Exception:
                    ps = 0.92
                ps = max(0.50, min(1.25, ps))

                plate_ids: List[int] = []
                if isinstance(feature_rects, dict):
                    pids = feature_rects.get("plate_islands")
                    if isinstance(pids, list):
                        for v in pids:
                            if isinstance(v, int):
                                plate_ids.append(int(v))
                            elif isinstance(v, str) and v.strip().isdigit():
                                plate_ids.append(int(v.strip()))

                # Fallback: infer from available bboxes (best-effort).
                if not plate_ids:
                    cands: List[Tuple[float, int]] = []
                    for rid, bb in island_bboxes.items():
                        if not isinstance(rid, int) or not isinstance(bb, tuple) or len(bb) != 4:
                            continue
                        x0, y0, x1, y1 = bb
                        w = max(1, x1 - x0)
                        h = max(1, y1 - y0)
                        ar = w / float(h)
                        af = (w * h) / float(size * size)
                        if 1.6 <= ar <= 6.2 and 0.0006 <= af <= 0.03:
                            cands.append((abs(ar - 3.05) + abs(af - 0.0055) * 120.0, rid))
                    cands.sort()
                    plate_ids = [rid for _, rid in cands[:2]]

                # If we have two plates stacked vertically with similar X-span, auto-flip the lower one.
                bbs = [island_bboxes.get(pid) for pid in plate_ids if island_bboxes.get(pid) is not None]
                stacked_pair = False
                if len(bbs) == 2:
                    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = bbs  # type: ignore[misc]
                    aw, ah = (ax1 - ax0), (ay1 - ay0)
                    bw, bh = (bx1 - bx0), (by1 - by0)
                    if abs(ax0 - bx0) <= 24 and abs(ax1 - bx1) <= 24 and abs(aw - bw) <= 24 and abs(ah - bh) <= 24:
                        stacked_pair = True

                for pid in plate_ids:
                    m = island_masks.get(pid)
                    bb = island_bboxes.get(pid)
                    if m is None or bb is None:
                        continue
                    x0, y0, x1, y1 = bb
                    rw = max(1, x1 - x0)
                    rh = max(1, y1 - y0)

                    pad = int(max(3, min(rw, rh) * 0.10))
                    rect = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)

                    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    d = ImageDraw.Draw(layer)
                    rr = max(2, int(min(rw, rh) * 0.22))
                    # Darken the plate slightly + white border (typical number plate look).
                    d.rounded_rectangle(
                        rect,
                        radius=rr,
                        fill=(0, 0, 0, int(170 * min(1.0, ps))),
                        outline=(245, 245, 245, 235),
                        width=max(2, int(min(rw, rh) * 0.06)),
                    )
                    _draw_centered_text_in_rect(
                        layer,
                        rect=rect,
                        text=txt,
                        fill=(245, 245, 245, 245),
                        stroke_fill=(0, 0, 0, 235),
                        max_font_px=max(10, int((rect[3] - rect[1]) * 0.68 * ps)),
                        rotate_degrees=0,
                    )

                    if stacked_pair:
                        cy = (y0 + y1) * 0.5
                        if cy > (size * 0.5):
                            crop = layer.crop((x0, y0, x1, y1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                            layer.paste(crop, (x0, y0))

                    # IMPORTANT: Plate islands in some templates are a "frame" (ring) with a dark interior
                    # that may be close to the template background. Our UV island detector can treat that
                    # interior as background, producing a mask with a hole (text would get clipped away).
                    # So for plate stamping, clip to a rounded-rect fill of the bbox (covers interior too).
                    plate_clip = Image.new("L", img.size, 0)
                    pd = ImageDraw.Draw(plate_clip)
                    outer_pad = max(1, int(min(rw, rh) * 0.03))
                    outer_rect = (x0 + outer_pad, y0 + outer_pad, x1 - outer_pad, y1 - outer_pad)
                    outer_r = max(2, int((outer_rect[3] - outer_rect[1]) * 0.48))
                    try:
                        pd.rounded_rectangle(outer_rect, radius=outer_r, fill=255)
                    except Exception:
                        pd.rectangle(outer_rect, fill=255)

                    la = layer.getchannel("A")
                    la = ImageChops.multiply(la, plate_clip)
                    layer.putalpha(la)
                    img = Image.alpha_composite(img, layer)

        # Sidepod text branding (independent of logo placement)
        if island_bboxes and sidepod_branding:
            sp_tag = (sidepod_tag_text if sidepod_tag_text is not None else tag).strip()
            sp_team = (sidepod_team_text if sidepod_team_text is not None else team_name).strip()
            if sp_tag or sp_team:
                try:
                    sp_scale = float(sidepod_branding_scale)
                except Exception:
                    sp_scale = 0.70
                # Allow very small branding; we control readability via font fitting + stroke scaling.
                sp_scale = max(0.08, min(1.0, sp_scale))
                mirror_mode = (sidepod_branding_mirror or "auto").strip().lower()
                if mirror_mode not in ("auto", "none", "right", "left", "both"):
                    mirror_mode = "auto"
                for island_id in [5, 6]:
                    m = island_masks.get(island_id)
                    bb = island_bboxes.get(island_id)
                    if m is None or bb is None:
                        continue
                    x0, y0, x1, y1 = bb
                    rw = max(1, x1 - x0)
                    rh = max(1, y1 - y0)

                    # Build a text overlay and clip it to the island.
                    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    # Scale by changing the available branding area (padding), keeping text readable.
                    # Smaller scale => more padding => smaller overall branding footprint (but still legible).
                    # NOTE: Sidepods are visually large; use aggressive padding so branding stays subtle.
                    # Primary complaint is "too big" horizontally, so shrink width more than height.
                    # Compute a target branding rectangle size (more predictable than additive padding).
                    # Smaller scale => smaller rect => more padding.
                    rect_w = int(rw * (0.12 + 0.50 * sp_scale))
                    rect_h = int(rh * (0.16 + 0.55 * sp_scale))
                    rect_w = max(int(rw * 0.18), min(int(rw * 0.62), rect_w))
                    rect_h = max(int(rh * 0.18), min(int(rh * 0.70), rect_h))
                    pad_x = max(12, (rw - rect_w) // 2)
                    pad_y = max(8, (rh - rect_h) // 2)
                    rect = (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)
                    cx0, cy0, cx1, cy1 = rect
                    ch = max(1, cy1 - cy0)

                    has_top = bool(sp_tag)
                    has_bottom = bool(sp_team)
                    if has_top and has_bottom:
                        top_h = int(ch * 0.62)
                        top_rect = (cx0, cy0, cx1, cy0 + top_h)
                        bot_rect = (cx0, cy0 + top_h, cx1, cy1)
                        _draw_centered_text_in_rect(
                            layer,
                            rect=top_rect,
                            text=sp_tag,
                            fill=(245, 245, 245, 245),
                            stroke_fill=(0, 0, 0, 245),
                            # Slightly smaller tag on sidepods (user wants more subtle branding).
                            max_font_px=max(14, int((top_rect[3] - top_rect[1]) * 0.68)),
                            rotate_degrees=0,
                        )
                        _draw_centered_text_in_rect(
                            layer,
                            rect=bot_rect,
                            text=sp_team,
                            fill=(235, 235, 238, 235),
                            stroke_fill=(0, 0, 0, 235),
                            max_font_px=max(10, int((bot_rect[3] - bot_rect[1]) * 0.50)),
                            rotate_degrees=0,
                        )
                    elif has_top:
                        _draw_centered_text_in_rect(
                            layer,
                            rect=rect,
                            text=sp_tag,
                            fill=(245, 245, 245, 245),
                            stroke_fill=(0, 0, 0, 245),
                            max_font_px=max(14, int(ch * 0.58)),
                            rotate_degrees=0,
                        )
                    elif has_bottom:
                        _draw_centered_text_in_rect(
                            layer,
                            rect=rect,
                            text=sp_team,
                            fill=(235, 235, 238, 235),
                            stroke_fill=(0, 0, 0, 235),
                            max_font_px=max(10, int(ch * 0.46)),
                            rotate_degrees=0,
                        )

                    # Auto-mirror on the right-side island to counter typical UV mirroring.
                    # If the side is NOT mirrored in this pack, this will be obvious and we can flip the logic.
                    # Determine which side of the car this island corresponds to.
                    # NOTE: For standard Stadium UV, sidepods (5/6) are stacked vertically in the texture,
                    # so X-based heuristics fail (both live on the left half). Use Y to decide "right side".
                    cy = (y0 + y1) * 0.5
                    do_flip = False
                    is_right = (cy > (size * 0.5)) if island_id in (5, 6) else ((x0 + x1) * 0.5 > (size * 0.5))
                    if mirror_mode == "both":
                        do_flip = True
                    elif mirror_mode == "right":
                        do_flip = is_right
                    elif mirror_mode == "left":
                        do_flip = not is_right
                    elif mirror_mode == "auto":
                        # Default behavior: flip right side only (common for mirrored UVs).
                        do_flip = is_right
                    else:  # "none"
                        do_flip = False

                    if do_flip:
                        # Sidepod islands 5/6 are stacked vertically in the UV and are effectively a vertical flip
                        # of each other; flipping TOP/BOTTOM keeps the branding consistent between car sides.
                        if island_id in (5, 6):
                            crop = layer.crop((x0, y0, x1, y1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        else:
                            crop = layer.crop((x0, y0, x1, y1)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        layer.paste(crop, (x0, y0))

                    la = layer.getchannel("A")
                    la = ImageChops.multiply(la, m)
                    layer.putalpha(la)
                    img = Image.alpha_composite(img, layer)

        # Sticker bomb overlay (apples/bananas/etc) across the whole used UV area.
        if sticker_images and sticker_count > 0:
            used_mask = None
            if isinstance(feature_rects, dict):
                um = feature_rects.get("used_mask")
                if isinstance(um, Image.Image):
                    used_mask = um
            mask_l = used_mask
            scope = (sticker_scope or "used").strip().lower()
            if scope not in ("used", "hero"):
                scope = "used"
            if scope == "hero" and island_masks:
                # Concentrate decals where they are guaranteed visible in-game:
                # - nose/top = 2
                # - sidepods = 5/6
                hero = Image.new("L", (size, size), 0)
                for rid in [2, 5, 6]:
                    m = island_masks.get(rid)
                    if m is None:
                        continue
                    hero = ImageChops.lighter(hero, m.convert("L"))
                mask_l = hero
            exclude = []
            if isinstance(feature_rects, dict):
                wr = feature_rects.get("wing_rects")
                if isinstance(wr, list):
                    exclude = [tuple(r) for r in wr if isinstance(r, tuple) and len(r) == 4]  # type: ignore
            # If wheel caps are using a special sticker tiling, keep fruits off them.
            if wheel_sticker is not None and island_bboxes:
                for wid in [10, 12, 14, 15, 16, 21]:
                    bb = island_bboxes.get(wid)
                    if bb is not None and len(bb) == 4:
                        exclude.append(tuple(bb))  # type: ignore[arg-type]
            img = _sprinkle_stickers(
                img,
                stickers=sticker_images,
                mask_l=mask_l,
                exclude_rects=exclude,
                rng=rng,
                count=int(sticker_count),
                min_scale=float(sticker_min_scale),
                max_scale=float(sticker_max_scale),
                rotate=bool(sticker_rotate),
                mode=str(sticker_mode),
            )

        # Wheel covers / caps: if wheel_rgb is provided, force a SOLID uniform cap color across all wheels.
        # This avoids the “each wheel cap looks different” issue caused by global patterns.
        # Wheel-cap candidates (4 repeated rectangular islands in Kacky/standard Stadium UV layout).
        # These correspond well to the “thing on top of wheels” in-game.
        # Note: some packs rank wheel-cap UV islands differently; include a couple extra candidates
        # so we reliably cover all visible wheel caps.
        wheel_islands = [10, 11, 12, 14, 15, 16, 21]  # caps + faces (11 is a prominent wheel face)
        tyre_islands = [13, 17, 18, 19, 20, 22]  # sidewall strips + rims + brake details + small wheel parts
        if wheel_sticker is not None:
            scope = (wheel_sticker_scope or "caps").strip().lower()
            if scope not in ("caps", "caps+tyres"):
                scope = "caps"
            target_islands = list(wheel_islands)
            if scope == "caps+tyres":
                target_islands.extend(tyre_islands)
            for wid in target_islands:
                m = island_masks.get(wid)
                bb = island_bboxes.get(wid)
                if m is None:
                    continue
                if bb is None:
                    bb = m.getbbox() or (0, 0, size, size)
                img = _tile_sticker_on_island(
                    img,
                    sticker=wheel_sticker,
                    island_mask_l=m,
                    island_bbox=bb,
                    tile_scale=float(wheel_sticker_scale),
                    rng=rng,
                )
        else:
            wheel_fill = wheel_rgb or (accent_color[0], accent_color[1], accent_color[2])
            # Paint ALL wheel + tyre islands when wheel_rgb is set
            all_wheel = wheel_islands + tyre_islands
            for wid in all_wheel:
                m = island_masks.get(wid)
                if m is None:
                    continue

                # Solid fill for consistency across all wheel caps + tyres.
                layer = Image.new("RGBA", img.size, (wheel_fill[0], wheel_fill[1], wheel_fill[2], 0))
                layer.putalpha(m)
                img = Image.alpha_composite(img, layer)

        # Strategic sponsor decals (rally-style): fixed slots on sidepods + nose (no randomness).
        if sponsor_images and island_masks and island_bboxes:
            try:
                s_imgs = [s.convert("RGBA") for s in sponsor_images if s is not None]
                if s_imgs:
                    def pick(idx: int) -> Image.Image:
                        return s_imgs[int(idx) % len(s_imgs)]

                    def _anchor_xy(bb: Tuple[int, int, int, int], anchor: str) -> Tuple[int, int]:
                        x0, y0, x1, y1 = bb
                        rw = max(1, x1 - x0)
                        rh = max(1, y1 - y0)
                        a = (anchor or "center").strip().lower()
                        if a in ("top", "t"):
                            return (x0 + int(rw * 0.52), y0 + int(rh * 0.22))
                        if a in ("bottom", "b"):
                            return (x0 + int(rw * 0.52), y0 + int(rh * 0.78))
                        if a in ("left", "l", "side"):
                            return (x0 + int(rw * 0.22), y0 + int(rh * 0.52))
                        if a in ("right", "r"):
                            return (x0 + int(rw * 0.78), y0 + int(rh * 0.52))
                        if a in ("top_right", "tr"):
                            return (x0 + int(rw * 0.78), y0 + int(rh * 0.22))
                        if a in ("top_left", "tl"):
                            return (x0 + int(rw * 0.22), y0 + int(rh * 0.22))
                        if a in ("bottom_right", "br"):
                            return (x0 + int(rw * 0.78), y0 + int(rh * 0.78))
                        if a in ("bottom_left", "bl"):
                            return (x0 + int(rw * 0.22), y0 + int(rh * 0.78))
                        return (x0 + int(rw * 0.52), y0 + int(rh * 0.52))

                    def _anchor_xy_mask(m_full: Image.Image, bb: Tuple[int, int, int, int], anchor: str) -> Tuple[int, int]:
                        """
                        Pick an anchor point inside the island mask using quantiles.
                        This avoids anchors landing in empty bbox space and getting snapped to the wrong edge.
                        """
                        x0, y0, x1, y1 = bb
                        # Fallback if numpy not available
                        if np is None:
                            return _anchor_xy(bb, anchor)
                        try:
                            mc = m_full.convert("L").crop((x0, y0, x1, y1))
                            arr = np.asarray(mc, dtype=np.uint8)
                            ys, xs = np.where(arr >= 8)
                            if xs.size < 10:
                                return _anchor_xy(bb, anchor)

                            a = (anchor or "center").strip().lower()
                            qx, qy = (0.50, 0.50)
                            if a in ("top", "t"):
                                qx, qy = (0.50, 0.08)
                            elif a in ("bottom", "b"):
                                qx, qy = (0.50, 0.92)
                            elif a in ("left", "l", "side"):
                                qx, qy = (0.08, 0.50)
                            elif a in ("right", "r"):
                                qx, qy = (0.92, 0.50)
                            elif a in ("top_right", "tr"):
                                qx, qy = (0.92, 0.08)
                            elif a in ("top_left", "tl"):
                                qx, qy = (0.08, 0.08)
                            elif a in ("bottom_right", "br"):
                                qx, qy = (0.92, 0.92)
                            elif a in ("bottom_left", "bl"):
                                qx, qy = (0.08, 0.92)

                            ax = int(np.clip(np.quantile(xs.astype(np.float32), qx), 0, max(0, (x1 - x0) - 1)))
                            ay = int(np.clip(np.quantile(ys.astype(np.float32), qy), 0, max(0, (y1 - y0) - 1)))
                            return (x0 + ax, y0 + ay)
                        except Exception:
                            return _anchor_xy(bb, anchor)

                    def _max_size(bb: Tuple[int, int, int, int], size_name: str) -> Tuple[int, int]:
                        x0, y0, x1, y1 = bb
                        rw = max(1, x1 - x0)
                        rh = max(1, y1 - y0)
                        sname = (size_name or "big").strip().lower()
                        # Keep these modest; user can still adjust via sponsor_scale.
                        if sname in ("tiny", "micro", "xs"):
                            # Slightly bigger than before (user: 13/12/10/9 barely visible)
                            return (int(rw * 0.56 * float(sponsor_scale)), int(rh * 0.24 * float(sponsor_scale)))
                        if sname in ("small", "s"):
                            return (int(rw * 0.34 * float(sponsor_scale)), int(rh * 0.14 * float(sponsor_scale)))
                        # big/default
                        return (int(rw * 0.42 * float(sponsor_scale)), int(rh * 0.16 * float(sponsor_scale)))

                    # If user provided explicit sponsor slots, obey them exactly.
                    if sponsor_slots:
                        for (island_id, anchor, size_name, sponsor_idx) in sponsor_slots:
                            m = island_masks.get(int(island_id))
                            bb = island_bboxes.get(int(island_id))
                            if m is None or bb is None:
                                continue
                            # Auto-flip: keep this VERY conservative.
                            # We previously flipped some non-mirrored islands (e.g. 3) and it made text upside down.
                            # Only apply to the classic stacked sidepod pair where UV mirroring is common.
                            flip = None
                            try:
                                x0, y0, x1, y1 = bb
                                cy_mid = (y0 + y1) * 0.5
                                if int(island_id) in (5, 6) and (cy_mid > (size * 0.5)):
                                    flip = "tb"
                            except Exception:
                                flip = None
                            dec = pick(int(sponsor_idx))
                            img = _place_decal_on_island(
                                img,
                                decal=dec,
                                island_mask_l=m.convert("L"),
                                island_bbox=bb,
                                center_xy=_anchor_xy_mask(m, bb, str(anchor)),
                                max_size=_max_size(bb, str(size_name)),
                                rotate_deg=0.0,
                                flip=flip,
                                opacity=float(sponsor_opacity),
                            )
                    else:
                        # Backwards-compatible default: sidepods 5/6 + cockpit-area nose on 2.
                        for idx, sid in enumerate([5, 6]):
                            m = island_masks.get(sid)
                            bb = island_bboxes.get(sid)
                            if m is None or bb is None:
                                continue
                            x0, y0, x1, y1 = bb
                            rw = max(1, x1 - x0)
                            rh = max(1, y1 - y0)
                            cx = x0 + int(rw * 0.56)
                            cy = y0 + int(rh * 0.46)
                            cy_mid = (y0 + y1) * 0.5
                            is_right = cy_mid > (size * 0.5)
                            flip = "tb" if is_right else None
                            img = _place_decal_on_island(
                                img,
                                decal=pick(idx),
                                island_mask_l=m.convert("L"),
                                island_bbox=bb,
                                center_xy=(cx, cy),
                                max_size=(int(rw * 0.42 * float(sponsor_scale)), int(rh * 0.16 * float(sponsor_scale))),
                                rotate_deg=0.0,
                                flip=flip,
                                opacity=float(sponsor_opacity),
                            )
                        m2 = island_masks.get(2)
                        bb2 = island_bboxes.get(2)
                        if m2 is not None and bb2 is not None:
                            cx, cy, target_size, rot = _estimate_nose_logo_placement(m2.convert("L"), mode="cockpit", rng=rng)
                            img = _place_decal_on_island(
                                img,
                                decal=pick(0),
                                island_mask_l=m2.convert("L"),
                                island_bbox=bb2,
                                center_xy=(cx, cy),
                                max_size=(int(target_size * 1.25 * float(sponsor_scale)), int(target_size * 0.72 * float(sponsor_scale))),
                                rotate_deg=rot,
                                flip=None,
                                opacity=float(sponsor_opacity),
                            )
            except Exception:
                pass

    # Fallback logo placements (if we DON'T have island masks)
    elif logo is not None:
        glow = (stripe_color[0], stripe_color[1], stripe_color[2], 120)
        _composite_logo_with_glow(
            img,
            logo=logo,
            xy=(int(size * 0.10), int(size * 0.28)),
            size_px=int(size * 0.28),
            glow_color=glow,
            glow_radius=max(4, size // 140),
        )
        _composite_logo_with_glow(
            img,
            logo=logo,
            xy=(int(size * 0.72), int(size * 0.62)),
            size_px=int(size * 0.14),
            glow_color=glow,
            glow_radius=max(3, size // 200),
        )

    # Rear wing plate (if known). We render it LAST so nothing (logo/patterns) can overlap it.
    wing_rects = (feature_rects or {}).get("wing_rects", []) if isinstance(feature_rects, dict) else []
    if wing_rects:
        # Ensure strong contrast on bright body colors.
        base_brightness = base_color[0] + base_color[1] + base_color[2]
        wing_bg = (14, 14, 16) if base_brightness > 380 else (max(0, base_color[0] - 25), max(0, base_color[1] - 25), max(0, base_color[2] - 25))
        top = (wing_top_text if wing_top_text is not None else tag).strip()
        bottom = (wing_bottom_text if wing_bottom_text is not None else team_name).strip()
        _render_wing_plate(
            img,
            wing_rect=wing_rects[0],
            top_text=top,
            bottom_text=bottom,
            accent_rgb=(accent_color[0], accent_color[1], accent_color[2]),
            background_rgb=wing_bg,
            decal=wing_sticker,
            decal_fit=str(wing_sticker_fit),
            decal_scale=float(wing_sticker_scale),
            decal_opacity=float(wing_sticker_opacity),
        )
    else:
        # Fallback: modest bottom-left label (not huge).
        name_font = _try_load_font(max(20, size // 18), bold=True)
        tag_font = _try_load_font(max(22, size // 14), bold=True)
        _draw_text_block(
            img,
            xy=(int(size * 0.06), int(size * 0.70)),
            text=team_name.upper(),
            font=name_font,
            fill=(245, 245, 245, 230),
            stroke_width=max(2, size // 360),
            stroke_fill=(0, 0, 0, 220),
            shadow=True,
        )
        if tag.strip():
            _draw_text_block(
                img,
                xy=(int(size * 0.06), int(size * 0.78)),
                text=tag,
                font=tag_font,
                fill=(accent_color[0], accent_color[1], accent_color[2], 235),
                stroke_width=max(2, size // 320),
                stroke_fill=(0, 0, 0, 220),
                shadow=True,
            )

    # Spatial-aware per-part color overlay (applies accent/secondary to specific parts).
    if spatial_aware and car_geometry is not None and isinstance(feature_rects, dict):
        island_masks_sa = feature_rects.get("island_masks", {}) if isinstance(feature_rects.get("island_masks", {}), dict) else {}
        if island_masks_sa:
            # Determine which parts get accent vs secondary color
            accent_parts = set(spatial_accent_parts or ["FRONT_WING", "SIDE_SKIRT", "REAR_WING_ENDPLATE", "NOSE_DETAIL"])
            secondary_parts = set(spatial_secondary_parts or ["FENDER", "REAR_SECTION"])
            
            # Apply accent color to accent parts
            accent_rgb = (accent_color[0], accent_color[1], accent_color[2])
            for island_id, mask in island_masks_sa.items():
                part = car_geometry.get_part(island_id)
                if part and part.name in accent_parts:
                    # Create color overlay
                    overlay = Image.new("RGBA", img.size, accent_rgb + (255,))
                    # Blend with ~70% strength to preserve some texture
                    mask_l = mask.convert("L")
                    # Reduce mask intensity for blending
                    mask_blend = Image.eval(mask_l, lambda x: int(x * 0.7))
                    img = Image.composite(overlay, img, mask_blend)
            
            # Apply secondary color to secondary parts
            secondary_rgb = (stripe_color[0], stripe_color[1], stripe_color[2])
            for island_id, mask in island_masks_sa.items():
                part = car_geometry.get_part(island_id)
                if part and part.name in secondary_parts:
                    overlay = Image.new("RGBA", img.size, secondary_rgb + (255,))
                    mask_l = mask.convert("L")
                    mask_blend = Image.eval(mask_l, lambda x: int(x * 0.6))
                    img = Image.composite(overlay, img, mask_blend)

    # Mudguard/wheel-arch color harmonization (ensures cohesive appearance across all 4 mudguards).
    if mudguards and isinstance(feature_rects, dict):
        island_masks_mg = feature_rects.get("island_masks", {}) if isinstance(feature_rects.get("island_masks", {}), dict) else {}
        island_bboxes_mg = feature_rects.get("island_bboxes", {}) if isinstance(feature_rects.get("island_bboxes", {}), dict) else {}
        comps_ranked_mg = feature_rects.get("comps_ranked_full")  # For automatic mudguard detection
        
        # Determine mudguard color based on mode.
        mg_color = _pick_mudguard_color(
            mode=mudguards_mode,
            base_rgb=(base_color[0], base_color[1], base_color[2]),
            accent_rgb=(accent_color[0], accent_color[1], accent_color[2]),
            secondary_rgb=(stripe_color[0], stripe_color[1], stripe_color[2]),
            custom_rgb=mudguards_color,
        )
        
        # Apply harmonization using island masks (preferred) or fallback rects.
        # Automatic detection uses comps_ranked_full when available.
        fallback_rects = _standard_stadium_mudguard_rects(size, size) if not island_masks_mg else None
        img = _apply_mudguard_harmonization(
            img,
            island_masks=island_masks_mg,
            island_bboxes=island_bboxes_mg,
            mudguard_color=mg_color,
            strength=mudguards_strength,
            feather=mudguards_feather,
            fallback_rects=fallback_rects,
            comps_ranked_full=comps_ranked_mg,
        )

    # Finish/material alpha (TMNF/TMN-style):
    # - It's not transparency; it trades off brightness vs reflection.
    # - We keep composition opaque, then write an alpha map based on chosen mode.
    finish_mode = str(finish_alpha or "auto").lower()
    if finish_mode == "opaque":
        img.putalpha(255)
    elif finish_mode == "neutral":
        img.putalpha(Image.new("L", (size, size), int(finish_neutral) & 0xFF))
    else:
        # auto: synthesize a readable finish map from RGB, then make stripes slightly more matte.
        spec = _make_tmnf_finish_alpha_from_rgb(img, neutral=int(finish_neutral) & 0xFF)
        stripe_mask = stripe_layer.getchannel("A")
        matte_val = max(18, int((int(finish_neutral) & 0xFF) - 70))
        spec = Image.composite(Image.new("L", (size, size), matte_val), spec, stripe_mask)
        # Optional finish choreography layer (premium matte/gloss separation).
        spec = _apply_finish_design(spec, rgb_src=img, mode=str(finish_design or "off"), strength=float(finish_design_strength))
        if finish_invert:
            # Invert synthesized finish map for packs/tutorials that use the opposite convention.
            spec = spec.point(lambda p: 255 - int(p))
        img.putalpha(spec)

    return img


def generate_team_icon(
    *,
    size: int,
    team_name: str,
    tag: str,
    logo: Optional[Image.Image],
    base_color: Tuple[int, int, int, int],
    accent_color: Tuple[int, int, int, int],
    stripe_color: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Simple 128x128-ish icon for Stadium mod packs (Icon.dds).
    """
    bg = Image.new("RGBA", (size, size), (base_color[0], base_color[1], base_color[2], 255))

    # Radial-ish gradient
    g = Image.new("L", (size, size), 0)
    gd = ImageDraw.Draw(g)
    for r in range(size, 0, -6):
        a = int(180 * (1 - (r / size)))
        gd.ellipse(((size - r) // 2, (size - r) // 2, (size + r) // 2, (size + r) // 2), fill=a)
    tint = Image.new("RGBA", (size, size), (accent_color[0], accent_color[1], accent_color[2], 255))
    tint.putalpha(g)
    icon = Image.alpha_composite(bg, tint)

    # Ring
    d = ImageDraw.Draw(icon)
    ring_w = max(3, size // 24)
    pad = max(6, size // 12)
    d.ellipse((pad, pad, size - pad, size - pad), outline=(stripe_color[0], stripe_color[1], stripe_color[2], 220), width=ring_w)

    # Logo
    if logo is not None:
        logo_size = int(size * 0.66)
        _composite_logo_with_glow(
            icon,
            logo=logo,
            xy=((size - logo_size) // 2, (size - logo_size) // 2 - max(0, size // 24)),
            size_px=logo_size,
            glow_color=(stripe_color[0], stripe_color[1], stripe_color[2], 200),
            glow_radius=max(3, size // 40),
        )

    # Tag text
    if tag.strip():
        f = _try_load_font(max(12, size // 6), bold=True)
        bbox = ImageDraw.Draw(icon).textbbox((0, 0), tag, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        _draw_text_block(
            icon,
            xy=((size - tw) // 2, size - th - max(6, size // 10)),
            text=tag,
            font=f,
            fill=(255, 255, 255, 240),
            stroke_width=max(1, size // 64),
            stroke_fill=(0, 0, 0, 220),
            shadow=True,
        )

    icon.putalpha(255)
    return icon


def _zip_skin(
    *,
    zip_path: Path,
    skin_dir_name: str,
    files: Sequence[Tuple[Path, str]],
) -> None:
    """
    files: list of (source_path, archive_name)
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arcname in files:
            z.write(src, arcname=arcname)


def _build_reskinned_mod_zip(
    *,
    base_zip_path: Path,
    out_zip_path: Path,
    replacements: Dict[str, bytes],
    additions: Optional[Dict[str, bytes]] = None,
) -> None:
    """
    Create a new zip with the same entries as base_zip_path, but with certain filenames replaced.
    Filenames are matched exactly (case-sensitive).
    """
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    additions = additions or {}
    with zipfile.ZipFile(base_zip_path, "r") as zin, zipfile.ZipFile(
        out_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        wrote_names: set[str] = set()
        for info in zin.infolist():
            name = info.filename

            # Preserve directory entries.
            if hasattr(info, "is_dir") and info.is_dir():
                zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                zi.external_attr = info.external_attr
                zout.writestr(zi, b"")
                wrote_names.add(name)
                continue

            if name in replacements:
                zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = info.external_attr
                zout.writestr(zi, replacements[name])
                wrote_names.add(name)
                continue

            # Stream-copy file content to avoid large memory spikes.
            zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            with zin.open(info, "r") as src, zout.open(zi, "w") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            wrote_names.add(name)

        # Add new files that were not present in the base zip.
        for name, data in additions.items():
            if (name in wrote_names) or (name in replacements):
                continue
            zi = zipfile.ZipInfo(filename=name)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, data)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate a TMNF car skin texture DDS (RGBA8 + mipmaps).")
    p.add_argument("--name", required=True, help="Skin name (used for output folder name + demo text).")
    p.add_argument(
        "--base-zip",
        default=None,
        help=(
            "Optional path to a WORKING Stadium car mod zip (contains Diffuse.dds/Mainbody*.Gbx, etc). "
            "If provided, this tool will output a new zip where Diffuse.dds is replaced with a generated texture."
        ),
    )
    p.add_argument(
        "--base-profile",
        default=None,
        help="Optional JSON profile for the base zip (from tools/profile_base_zip.py). If omitted, auto-loads profiles/<sha256>.json when present.",
    )
    p.add_argument(
        "--no-base-profile-auto",
        action="store_true",
        help="Disable auto profile lookup in profiles/ for --base-zip.",
    )
    p.add_argument("--size", type=int, default=1024, help="Texture size (typically 1024 for TMNF templates).")
    p.add_argument("--out", default="out", help="Output directory.")
    p.add_argument("--texture-filename", default="CarSport.dds", help="DDS filename to write (e.g. CarSport.dds).")
    p.add_argument("--no-mipmaps", action="store_true", help="Disable mipmaps (not recommended).")
    p.add_argument(
        "--dds-format",
        choices=["auto", "rgba8", "dxt5"],
        default="auto",
        help="DDS pixel format to output. 'auto' uses DXT5 when reskinning a DXT5 mod zip; otherwise RGBA8.",
    )
    p.add_argument("--logo", default=None, help="Optional team logo PNG to place on the skin/icon.")
    p.add_argument("--logo-tint", type=_parse_hex_color, default=None, help="Optional logo tint color (RRGGBB or RRGGBBAA).")
    p.add_argument(
        "--logo-cutout",
        choices=["auto", "off"],
        default="auto",
        help="Auto-remove opaque flat background from logo images (fixes 'square logo' when PNG has no transparency).",
    )
    p.add_argument("--team-name", default=None, help="Optional team name text to render (defaults to --name).")
    p.add_argument("--tag", default=None, help="Optional short tag to render (e.g. 'CH').")
    p.add_argument(
        "--no-text",
        action="store_true",
        help="Disable all generated team/tag/wing text (useful when you only want graphics like wing stickers and patterns).",
    )
    p.add_argument("--wing-text", default=None, help="Custom text for the rear wing (overrides default tag+team name). Use for player names etc.")
    p.add_argument("--wing-sticker", default=None, help="Optional PNG decal to render on the rear wing plate (standard Stadium model only).")
    p.add_argument(
        "--wing-sticker-fit",
        choices=["contain", "cover", "stretch"],
        default="contain",
        help="How to fit the wing sticker inside the wing rect (contain=keep full decal, cover=fill+crop).",
    )
    p.add_argument("--wing-sticker-scale", type=float, default=1.0, help="Scale multiplier for wing sticker (0.05..10).")
    p.add_argument("--wing-sticker-opacity", type=float, default=1.0, help="Opacity for wing sticker (0..1).")
    p.add_argument("--wing-sticker-keep-text", action="store_true", help="Keep wing text plate text even when --wing-sticker is set.")
    p.add_argument("--nose-logo", action="store_true", help="Place a small logo on the nose (UV island 2 on standard model).")
    p.add_argument(
        "--logo-layout",
        choices=["default", "sidepods", "front", "cockpit", "sidepods+front", "sidepods+cockpit", "front+cockpit", "all"],
        default="default",
        help="Where to place the main logo. default = sidepods + optional --nose-logo. front = between front wheels (nose tip). cockpit = in front of driver.",
    )
    p.add_argument(
        "--logo-plate",
        choices=["auto", "on", "off"],
        default="auto",
        help="Background plate behind logo decals. auto=plate on sidepods, blend on nose/front/cockpit. on=always plate. off=never plate.",
    )
    p.add_argument("--logo-scale", type=float, default=1.0, help="Logo size multiplier (applies to all logo placements).")
    p.add_argument("--sidepod-branding", action="store_true", help="Add sidepod text branding: tag on top + small team name underneath (UV islands 5/6).")
    p.add_argument("--sidepod-tag-text", default=None, help="Override sidepod tag text (defaults to --tag).")
    p.add_argument("--sidepod-team-text", default=None, help="Override sidepod team text (defaults to --team-name).")
    p.add_argument("--sidepod-branding-scale", type=float, default=0.70, help="Scale for sidepod branding (0.25..1.0). Smaller = more subtle.")
    p.add_argument(
        "--sidepod-branding-mirror",
        choices=["auto", "none", "right", "left", "both"],
        default="auto",
        help="Mirror sidepod branding to compensate for UV mirroring. auto=flip right side, none=no flip.",
    )
    p.add_argument("--sticker", action="append", default=[], help="Sticker PNG path to sprinkle across the livery (repeat to add multiple).")
    p.add_argument("--sticker-mode", choices=["random", "grid"], default="grid", help="Sticker placement: random or grid (grid = even coverage).")
    p.add_argument("--sticker-count", type=int, default=0, help="How many stickers to place (0 disables).")
    p.add_argument("--sticker-min-scale", type=float, default=0.03, help="Min sticker size as fraction of texture size (e.g. 0.03).")
    p.add_argument("--sticker-max-scale", type=float, default=0.08, help="Max sticker size as fraction of texture size (e.g. 0.08).")
    p.add_argument("--sticker-no-rotate", action="store_true", help="Do not rotate stickers randomly.")
    p.add_argument(
        "--sticker-scope",
        choices=["used", "hero"],
        default="used",
        help="Where stickers may be placed. hero = only nose+sidepods (guaranteed visible).",
    )
    p.add_argument("--sponsor", action="append", default=[], help="Sponsor decal PNG to place strategically (sidepods + nose). Repeat to add multiple.")
    p.add_argument("--sponsor-scale", type=float, default=1.0, help="Scale multiplier for strategic sponsors (0.5..2.5).")
    p.add_argument("--sponsor-opacity", type=float, default=1.0, help="Opacity for strategic sponsors (0..1).")
    p.add_argument(
        "--sponsor-slot",
        action="append",
        default=[],
        help="Custom sponsor placement slot: ISLAND:ANCHOR:SIZE:IDX (e.g. 4:top:big:0). Repeat to add multiple.",
    )
    p.add_argument("--wheel-sticker", default=None, help="PNG to tile on wheel caps (smiley faces, etc).")
    p.add_argument("--wheel-sticker-scale", type=float, default=0.48, help="Wheel sticker tile size relative to wheel island (0.1..1.0).")
    p.add_argument("--wheel-sticker-scope", choices=["caps", "caps+tyres"], default="caps", help="Apply wheel sticker only to caps, or also to tyre sidewall strip (best-effort).")
    p.add_argument("--flag", default=None, help="Optional flag decal code (e.g. 'pl' for Poland).")
    p.add_argument("--flag-location", choices=["nose", "sidepods"], default="nose", help="Where to place the flag decal (requires UV island masks from --base-zip).")
    p.add_argument("--flag-scale", type=float, default=0.18, help="Flag size relative to target island (0.08..0.45).")
    p.add_argument("--plate-text", default=None, help="Optional number plate text (auto-detected plate island in standard Stadium packs).")
    p.add_argument("--plate-scale", type=float, default=0.92, help="Scale for plate text/background (0.5..1.25).")
    p.add_argument(
        "--plate-island",
        action="append",
        type=int,
        default=[],
        help="Override auto plate detection: use these UV island rank ids (use --uv-debug to see numbers). Repeat to add multiple.",
    )
    p.add_argument("--wheel-color", type=_parse_hex_color, default=None, help="Optional wheel/rim accent color to match (recolors base pack accents).")
    # Mudguard color harmonization (ensure cohesive wheel-arch appearance)
    p.add_argument(
        "--mudguards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mudguard/wheel-arch color harmonization (default: on). Ensures all 4 mudguards have a cohesive intentional color.",
    )
    p.add_argument(
        "--mudguards-color",
        type=_parse_hex_color,
        default=None,
        help="Override mudguard color (hex like 1a1a1a). If not set, uses --mudguards-mode to pick automatically.",
    )
    p.add_argument(
        "--mudguards-mode",
        choices=["match_base", "match_accent", "match_secondary", "darken", "custom"],
        default="darken",
        help="How to pick mudguard color: match_base=use base color, match_accent=use accent, match_secondary=use secondary, darken=darken base by 40%%, custom=use --mudguards-color.",
    )
    p.add_argument(
        "--mudguards-strength",
        type=float,
        default=0.85,
        help="Blend strength for mudguard tint (0.0=none, 1.0=full replacement). Default 0.85 preserves some texture detail.",
    )
    p.add_argument(
        "--mudguards-feather",
        type=int,
        default=3,
        help="Feather radius for mudguard mask edges (pixels). Higher = softer blend at boundaries.",
    )
    p.add_argument(
        "--sanitize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sanitize output: reduce visible banding on gradients and remove common base-pack watermarks/credits (best-effort).",
    )
    p.add_argument("--bw-only", action="store_true", help="Force RGB to black/white only (no colors).")
    p.add_argument("--proj-logo", action="store_true", help="Also generate a custom projection texture (ProjShad.dds) using the team logo.")
    p.add_argument("--proj-image", default=None, help="Optional PNG to use as ProjShad.dds (custom car shadow/projection). Overrides --proj-logo.")
    p.add_argument(
        "--proj-wings",
        action="store_true",
        help="Generate a raven-wings-style ProjShad.dds procedurally (white background, dark wing tiremarks). Overrides --proj-logo.",
    )
    p.add_argument(
        "--illum-image",
        default=None,
        help=(
            "Optional PNG/TGA to use as Illum.dds override (baked glow map for Details UVs). "
            "Alpha (if present) will be treated as a mask; output prefers DXT1 when possible."
        ),
    )
    p.add_argument("--inspire-zip", default=None, help="Optional example skin zip to use as a handmade composition guide (palette-mapped into your colors).")
    p.add_argument("--inspire-source", choices=["auto", "diffuse", "details"], default="auto", help="Which texture to sample from inspire zip (auto picks Diffuse unless it's flat).")
    p.add_argument("--inspire-strength", type=float, default=0.78, help="Blend strength for inspired composition (0..1).")
    p.add_argument(
        "--style",
        choices=[
            # Classic styles
            "solid", "fade", "fade_splatter", "splatter", "neon", "carbon", "stone", "fluid", "topo", "galaxy",
            # New handmade themes (non-repeating, heavily varied by seed)
            "camo", "halftone", "shards", "brushed", "glitch",
            # Pro styles (gradient base)
            "pro_slashes", "pro_swoosh", "pro_blocks", "pro_minimal", "pro_mixed", "pro_mixmatch",
            "pro_kintsugi", "pro_circuit",
            "pro_heritage",
            # Pro + Fade combinations
            "pro_slashes_fade", "pro_swoosh_fade", "pro_blocks_fade", "pro_mixmatch_fade",
            "pro_kintsugi_fade", "pro_circuit_fade",
            "pro_heritage_fade",
            # Pro + Splatter combinations
            "pro_slashes_splatter", "pro_swoosh_splatter", "pro_blocks_splatter", "pro_mixmatch_splatter",
            # Pro + Galaxy combinations
            "pro_slashes_galaxy", "pro_swoosh_galaxy", "pro_mixmatch_galaxy",
            # Pro + Fluid combinations
            "pro_slashes_fluid", "pro_swoosh_fluid", "pro_mixmatch_fluid",
            # Pro + Carbon combinations
            "pro_slashes_carbon", "pro_blocks_carbon", "pro_mixmatch_carbon",
            "pro_kintsugi_carbon", "pro_circuit_carbon",
            "pro_heritage_carbon",
            # Pro + Razzle (fade + shards + splatter + halftone + glitch)
            "pro_slashes_razzle", "pro_swoosh_razzle", "pro_blocks_razzle", "pro_mixmatch_razzle",
            # Pro + Holo (iridescent neon gradient + prism lines)
            "pro_slashes_holo", "pro_swoosh_holo", "pro_blocks_holo", "pro_mixmatch_holo",
            # Pro + Crafted (handmade/tape/brush + controlled splatter)
            "pro_slashes_crafted", "pro_swoosh_crafted", "pro_blocks_crafted", "pro_mixmatch_crafted",
            # Pro + Fusion (intentional multi-motif handcrafted)
            "pro_fusion", "pro_fusion_fade", "pro_fusion_aurora", "pro_fusion_inkblot",
            # DIVERSE THEMES - fundamentally different looks
            "theme_clean",       # Solid base, single thin accent line only
            "theme_twotone",     # Hard 50/50 split
            "theme_gradient",    # Pure smooth gradient, no shapes
            "theme_splatter",    # Heavy splatter texture only, no slashes
            "theme_racing",      # Classic dual racing stripes
            "theme_stealth",     # Nearly solid dark with subtle accent
            # Novel 2026 styles - unique algorithmic art
            "theme_suminagashi", # Japanese floating-ink marbling (concentric rings warped by vector field)
            "theme_moire",       # Optical interference from overlapping radial grids
            "theme_palimpsest",  # Layered urban abstraction (city grid + dot grid + spray arcs + color veil)
        ],
        default="neon",
        help="Visual style for team skins. Use theme_* for fundamentally different looks.",
    )
    p.add_argument(
        "--palette",
        choices=list(PRO_COLOR_PALETTES.keys()) + ["auto_vibrant"],
        default=None,
        help="Pro color palette (or 'auto_vibrant' for a seed-driven vibrant palette).",
    )

    p.add_argument(
        "--finish-alpha",
        choices=["opaque", "neutral", "auto"],
        default="auto",
        help="How to generate Diffuse alpha/material channel: opaque=255, neutral=0x8E everywhere, auto=TMNF-style finish map (bright matte accents + glossy dark base).",
    )
    p.add_argument(
        "--finish-neutral",
        type=int,
        default=0x8E,
        help="Neutral alpha (0..255) used by --finish-alpha neutral/auto as midpoint (default 0x8E from common TMNF tutorials).",
    )
    p.add_argument(
        "--finish-invert",
        action="store_true",
        help=(
            "Invert the synthesized finish alpha map. Useful because some community tutorials describe opposite alpha "
            "conventions and perceived brightness can flip depending on reflections/environment. Applies to --finish-alpha auto."
        ),
    )
    p.add_argument(
        "--finish-design",
        choices=["off", "edges", "sweep"],
        default="off",
        help="Optional finish/material design layer to improve 'pro paint' feel. edges=glossier cutlines, sweep=subtle clearcoat gloss band. Applies to --finish-alpha auto.",
    )
    p.add_argument(
        "--finish-design-strength",
        type=float,
        default=0.35,
        help="Strength for --finish-design (0..1).",
    )
    p.add_argument("--grade-contrast", type=float, default=None, help="Optional final contrast grade for pro styles (recommended 1.0..1.6). Default keeps current behavior.")
    p.add_argument("--grade-color", type=float, default=None, help="Optional final color saturation grade for pro styles (recommended 0.9..1.35). Default keeps current behavior.")
    p.add_argument("--grade-gamma", type=float, default=None, help="Optional final gamma grade for pro styles (recommended 0.7..1.2). Lower = darker mids. Default keeps current behavior.")
    p.add_argument("--vignette-strength", type=int, default=None, help="Optional vignette strength override for some pro effects (0..180). Default keeps per-style behavior.")
    p.add_argument(
        "--dxt-sharpen",
        choices=["auto", "on", "off"],
        default="auto",
        help="DXT-friendly edge sharpening before DDS export. auto=on for DXT5 output, off for RGBA8. Use off if you see halos/banding.",
    )
    p.add_argument("--dxt-sharpen-strength", type=float, default=0.35, help="Sharpen blend strength (0..1).")
    p.add_argument("--dxt-sharpen-radius", type=float, default=1.2, help="Unsharp radius in pixels (0.6..2.5).")
    p.add_argument("--dxt-sharpen-percent", type=int, default=140, help="Unsharp percent (50..250).")
    p.add_argument("--dxt-sharpen-threshold", type=int, default=6, help="Unsharp threshold (0..20).")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to control procedural variation. Defaults to a stable hash of --name/style/colors.",
    )
    p.add_argument("--no-icon", action="store_true", help="In --base-zip mode, do not replace Icon.dds.")
    p.add_argument(
        "--no-recolor-glow",
        action="store_true",
        help="In --base-zip team mode, do not recolor the base pack's glow textures (Illum/Details) to match your palette.",
    )
    p.add_argument(
        "--uv-debug",
        action="store_true",
        help="Generate a UV-debug zip (numbers/colors per UV island) to map which texture regions correspond to which car parts.",
    )
    p.add_argument(
        "--uv-debug-all",
        action="store_true",
        help="In --uv-debug mode, also replace Details/Dirty textures with big labeled debug patterns (helps find plates/tyres).",
    )

    # Spatial awareness mode
    p.add_argument(
        "--spatial-aware",
        action="store_true",
        help="Enable spatial-aware mode: auto-apply colors by part role (accent on wings/skirts, darken mudguards) and per-part finish zones.",
    )
    p.add_argument(
        "--spatial-accent-parts",
        nargs="*",
        default=["FRONT_WING", "SIDE_SKIRT", "REAR_WING_ENDPLATE", "NOSE_DETAIL"],
        help="Parts to receive accent color in --spatial-aware mode (default: FRONT_WING SIDE_SKIRT REAR_WING_ENDPLATE NOSE_DETAIL).",
    )
    p.add_argument(
        "--spatial-secondary-parts",
        nargs="*",
        default=["FENDER", "REAR_SECTION"],
        help="Parts to receive secondary/stripe color in --spatial-aware mode (default: FENDER REAR_SECTION).",
    )

    p.add_argument("--base-color", type=_parse_hex_color, default="#1a1a1aff", help="Base color (RRGGBB or RRGGBBAA).")
    p.add_argument("--accent-color", type=_parse_hex_color, default="#ff7a00a0", help="Accent color (RRGGBB or RRGGBBAA).")
    p.add_argument("--stripe-color", type=_parse_hex_color, default="#00b4ffa0", help="Stripe color (RRGGBB or RRGGBBAA).")

    p.add_argument("--template", default=None, help="Optional template PNG to start from (resized to --size).")
    p.add_argument("--prelight", default=None, help="Optional prelight PNG to multiply over the result.")
    p.add_argument("--prelight-strength", type=float, default=1.0, help="0..1 multiply strength for prelight.")

    p.add_argument("--preview-png", action="store_true", help="Also write a PNG preview next to the DDS.")
    p.add_argument("--zip", dest="zip_skin", action="store_true", help="Also create a zip package.")
    p.add_argument(
        "--zip-layout",
        choices=["folder", "root"],
        default="folder",
        help="If --zip: put DDS under <name>/ (folder) or at zip root (root).",
    )

    args = p.parse_args(argv)

    if args.size <= 0:
        print("ERROR: --size must be > 0", file=sys.stderr)
        return 2

    if not _is_power_of_two(args.size):
        print(f"WARNING: --size {args.size} is not a power of two. DDS will still be written, but TMNF may not like it.")

    out_dir = Path(args.out).expanduser().resolve()

    # Mode A: reskin a working Stadium mod zip by replacing Diffuse.dds
    if args.base_zip:
        base_zip_path = Path(args.base_zip).expanduser().resolve()
        if not base_zip_path.exists():
            print(f"ERROR: --base-zip not found: {base_zip_path}", file=sys.stderr)
            return 2

        # Optional: apply pack-specific defaults from a saved base-zip profile.
        # This is non-breaking: we only apply values when the user didn't explicitly opt out.
        try:
            prof = _try_load_base_zip_profile(
                base_zip_path,
                profile_path=str(getattr(args, "base_profile", None) or "") or None,
                allow_auto=not bool(getattr(args, "no_base_profile_auto", False)),
            )
            if isinstance(prof, dict):
                rec = prof.get("recommended")
                if isinstance(rec, dict):
                    if bool(rec.get("finish_invert", False)) and (not bool(getattr(args, "finish_invert", False))):
                        args.finish_invert = True
                    try:
                        fn = rec.get("finish_neutral", None)
                        if isinstance(fn, int) and int(getattr(args, "finish_neutral", 0x8E)) == 0x8E:
                            args.finish_neutral = int(fn)
                    except Exception:
                        pass
        except Exception:
            pass

        # Find Diffuse.dds in the base zip and read its dimensions.
        base_zip_names: set[str] = set()
        try:
            with zipfile.ZipFile(base_zip_path, "r") as zin:
                base_zip_names = set(zin.namelist())
                with zin.open("Diffuse.dds", "r") as f:
                    diffuse_hdr = f.read(128)
        except KeyError:
            print("ERROR: base zip does not contain 'Diffuse.dds' at zip root.", file=sys.stderr)
            print(
                "       Please provide a Stadium mod zip like your working one (it has Diffuse.dds, Details.dds, Mainbody*.Gbx...).",
                file=sys.stderr,
            )
            return 2

        try:
            diffuse_w, diffuse_h = _read_dds_dimensions_from_bytes(diffuse_hdr)
        except Exception as e:
            print(f"ERROR: Could not parse Diffuse.dds header in base zip: {e}", file=sys.stderr)
            return 2

        # Preserve base Diffuse alpha if it's constant. Many TMNF packs use alpha as a material channel.
        base_alpha_const: Optional[int] = None
        try:
            with zipfile.ZipFile(base_zip_path, "r") as zin:
                base_diffuse_bytes_for_alpha = zin.read("Diffuse.dds")
            base_diffuse_img_for_alpha = Image.open(io.BytesIO(base_diffuse_bytes_for_alpha)).convert("RGBA")
            mn, mx = base_diffuse_img_for_alpha.getchannel("A").getextrema()
            if mn == mx:
                base_alpha_const = int(mn)
        except Exception:
            base_alpha_const = None

        # Detect whether the base zip uses the standard Stadium model, so we can place rear-wing text reliably.
        is_standard_stadium = False
        try:
            with zipfile.ZipFile(base_zip_path, "r") as zin:
                if "MainBody.Solid.Gbx" in zin.namelist():
                    gbx = zin.read("MainBody.Solid.Gbx")
                    is_standard_stadium = hashlib.sha256(gbx).hexdigest() == STANDARD_STADIUM_MAINBODY_SHA256
        except Exception:
            is_standard_stadium = False

        base_fourcc = _read_dds_fourcc_from_bytes(diffuse_hdr)
        out_dds_format = args.dds_format
        if out_dds_format == "auto":
            out_dds_format = "dxt5" if base_fourcc == "DXT5" else "rgba8"

        # Generate at the same size as the working mod to maximize compatibility.
        if diffuse_w != diffuse_h:
            print(f"WARNING: Diffuse.dds is not square ({diffuse_w}x{diffuse_h}); generating at exact dims.", file=sys.stderr)

        if args.size != diffuse_w or args.size != diffuse_h:
            print(f"NOTE: Ignoring --size {args.size}; using base zip Diffuse.dds size {diffuse_w}x{diffuse_h}.")

        team_name = (args.team_name or args.name).strip()
        tag = (args.tag or "").strip()
        if bool(getattr(args, "no_text", False)):
            team_name = ""
            tag = ""
        logo = Image.open(args.logo).convert("RGBA") if args.logo else None
        if logo is not None and getattr(args, "logo_cutout", "auto") == "auto":
            logo = _auto_cutout_logo_background(logo, tolerance=28)
        if logo is not None and args.logo_tint is not None:
            logo = _tint_logo(logo, rgb=(args.logo_tint[0], args.logo_tint[1], args.logo_tint[2]))
        # Optional sticker assets (apples/bananas/etc).
        sticker_images: List[Image.Image] = []
        if getattr(args, "sticker", None):
            for sp in (args.sticker or []):
                try:
                    si = Image.open(sp).convert("RGBA")
                    si = _prepare_sticker_rgba(si, tolerance=32)
                    sticker_images.append(si)
                except Exception:
                    continue
        sponsor_images: List[Image.Image] = []
        if getattr(args, "sponsor", None):
            for sp in (args.sponsor or []):
                try:
                    si = Image.open(sp).convert("RGBA")
                    si = _prepare_sticker_rgba(si, tolerance=32)
                    sponsor_images.append(si)
                except Exception:
                    continue
        sponsor_slots: List[Tuple[int, str, str, int]] = []
        if getattr(args, "sponsor_slot", None):
            for spec in (args.sponsor_slot or []):
                try:
                    parts = str(spec).strip().split(":")
                    if len(parts) < 3:
                        continue
                    island_id = int(parts[0])
                    anchor = parts[1].strip()
                    size_name = parts[2].strip()
                    idx = int(parts[3]) if len(parts) >= 4 and parts[3].strip() else 0
                    if island_id <= 0:
                        continue
                    sponsor_slots.append((island_id, anchor, size_name, idx))
                except Exception:
                    continue
        wheel_sticker_img: Optional[Image.Image] = None
        if getattr(args, "wheel_sticker", None):
            try:
                ws = Image.open(args.wheel_sticker).convert("RGBA")  # type: ignore[arg-type]
                wheel_sticker_img = _prepare_sticker_rgba(ws, tolerance=32)
            except Exception:
                wheel_sticker_img = None
        wing_sticker_img: Optional[Image.Image] = None
        if getattr(args, "wing_sticker", None):
            try:
                wi = Image.open(args.wing_sticker).convert("RGBA")  # type: ignore[arg-type]
                wing_sticker_img = _prepare_sticker_rgba(wi, tolerance=32)
            except Exception:
                wing_sticker_img = None
        feature_rects: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None

        # NOTE: In many Stadium packs, the actual license plate surface is driven by Details.dds, not Diffuse.dds.
        # We'll stamp plate text onto Details.dds later if it exists; avoid stamping on Diffuse by default
        # UNLESS the user explicitly provided --plate-island (manual UV island on Diffuse).
        base_has_details = "Details.dds" in base_zip_names
        force_diffuse_plate = bool(getattr(args, "plate_island", None)) and bool(args.plate_island)

        # For the standard Stadium model, we KNOW the rear-wing UV island rect (verified via Kacky skins).
        # This avoids the helmet-mapping issue entirely.
        if is_standard_stadium:
            feature_rects = {"wing_rects": _standard_stadium_rear_wing_rects(diffuse_w, diffuse_h)}
        else:
            # Best-effort fallback (may vary by mod pack).
            feature_rects = None

        # If standard model, also compute island masks/bboxes so we can place logos on sidepods/nose precisely.
        if is_standard_stadium and not args.uv_debug:
            try:
                with zipfile.ZipFile(base_zip_path, "r") as zin:
                    base_diffuse_bytes = zin.read("Diffuse.dds")
                base_diffuse_img = Image.open(io.BytesIO(base_diffuse_bytes)).convert("RGBA").resize((diffuse_w, diffuse_h), Image.Resampling.NEAREST)

                # IMPORTANT: Use a known template for UV island detection.
                # Some packs paint Diffuse.dds everywhere (no near-black unused background), which makes
                # connected-component UV detection collapse into one giant island. The template zip has
                # a proper near-black unused background, so we get stable island ids (2/5/6/etc).
                tmpl = _load_standard_stadium_uv_template_diffuse(size=(diffuse_w, diffuse_h))
                uv_source = tmpl if tmpl is not None else base_diffuse_img

                ranked_map_small, comps_ranked_full, scale = _compute_ranked_uv_islands(uv_source, downscale_to=512)

                def mask_for(rank_id: int) -> Image.Image:
                    ms = (ranked_map_small == rank_id).astype(np.uint8) * 255  # type: ignore
                    m_img = Image.fromarray(ms).convert("L")
                    return m_img.resize((diffuse_w, diffuse_h), Image.Resampling.NEAREST)
                
                def mask_for_used() -> Image.Image:
                    ms = (ranked_map_small > 0).astype(np.uint8) * 255  # type: ignore
                    m_img = Image.fromarray(ms).convert("L")
                    return m_img.resize((diffuse_w, diffuse_h), Image.Resampling.NEAREST)

                # We care most about: nose=2, sidepods=5/6 (from your UV debug), wing is handled separately.
                # Extra ids available for fine-tuning (small repeated islands often map to wheel caps / fenders).
                # We include 12/14/15/16 which are similarly-sized repeated islands (good wheel-cap candidates).
                # Allow manual override via CLI (useful when auto detection isn't correct for a pack).
                if getattr(args, "plate_island", None):
                    plate_ids = [int(x) for x in (args.plate_island or []) if isinstance(x, int) and int(x) > 0]
                else:
                    plate_ids = _guess_number_plate_island_ids(comps_ranked_full, tex_w=diffuse_w, tex_h=diffuse_h, max_count=2)
                island_ids = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 21] + plate_ids
                # Deduplicate while preserving order.
                seen: set[int] = set()
                uniq: List[int] = []
                for rid in island_ids:
                    if rid in seen:
                        continue
                    seen.add(rid)
                    uniq.append(rid)
                island_ids = uniq
                island_masks = {rid: mask_for(rid) for rid in island_ids}
                island_bboxes = {rid: comps_ranked_full[rid - 1][:4] for rid in island_ids if 0 < rid <= len(comps_ranked_full)}
                feature_rects = dict(feature_rects or {})
                feature_rects["island_masks"] = island_masks
                feature_rects["island_bboxes"] = island_bboxes
                used_mask = mask_for_used().convert("L")
                # IMPORTANT: Some templates have "plate" regions that are used by the UVs but are
                # near-identical to the background in Diffuse.dds, so our automatic used-mask can miss them.
                # Ensure the full plate rectangle interior is considered "used" so plate text won't be cut out.
                if plate_ids:
                    ud = ImageDraw.Draw(used_mask)
                    for pid in plate_ids:
                        try:
                            x0, y0, x1, y1 = comps_ranked_full[int(pid) - 1][:4]
                        except Exception:
                            continue
                        rw = max(1, int(x1 - x0))
                        rh = max(1, int(y1 - y0))
                        outer_pad = max(1, int(min(rw, rh) * 0.03))
                        rect = (int(x0 + outer_pad), int(y0 + outer_pad), int(x1 - outer_pad), int(y1 - outer_pad))
                        r = max(2, int((rect[3] - rect[1]) * 0.48))
                        try:
                            ud.rounded_rectangle(rect, radius=r, fill=255)
                        except Exception:
                            ud.rectangle(rect, fill=255)
                feature_rects["used_mask"] = used_mask
                feature_rects["plate_islands"] = plate_ids
                feature_rects["comps_ranked_full"] = comps_ranked_full  # For automatic mudguard detection
            except Exception:
                # If anything fails, we still have wing placement.
                pass

        # UV debug mode: replace Diffuse.dds with a numbered debug texture. Optionally also replace
        # other textures so we can identify which channel drives plates/tyres.
        if args.uv_debug:
            with zipfile.ZipFile(base_zip_path, "r") as zin:
                base_diffuse_bytes = zin.read("Diffuse.dds")
            base_diffuse_img = Image.open(io.BytesIO(base_diffuse_bytes)).convert("RGBA")
            # IMPORTANT: Some packs paint Diffuse everywhere (no near-black unused background),
            # which breaks UV island detection in debug mode (collapses into 1 big island).
            # For standard Stadium, use the known-good UV template diffuse so island ids are stable.
            dbg_src = base_diffuse_img
            if is_standard_stadium:
                try:
                    tmpl = _load_standard_stadium_uv_template_diffuse(size=(diffuse_w, diffuse_h))
                    if tmpl is not None:
                        dbg_src = tmpl
                except Exception:
                    dbg_src = base_diffuse_img
            debug_img = _build_uv_debug_image_from_base(dbg_src)
            debug_dds_bytes = build_dds_dxt5_bytes(debug_img, mipmaps=not args.no_mipmaps)
            out_zip_path = out_dir / f"{args.name}.zip"
            replacements: Dict[str, bytes] = {"Diffuse.dds": debug_dds_bytes}

            if getattr(args, "uv_debug_all", False):
                # Add loud labels for other textures. These are not UV-island numbered; the goal is to
                # see *which texture* affects the plate/tyre surfaces.
                tex_specs = [
                    ("Details.dds", (0, 90, 180)),
                    ("DiffuseDirty.dds", (0, 140, 60)),
                    ("DetailsDirty.dds", (180, 120, 0)),
                    ("ProjShad.dds", (150, 0, 150)),
                ]
                with zipfile.ZipFile(base_zip_path, "r") as zin:
                    names = set(zin.namelist())
                    for tex_name, bg in tex_specs:
                        if tex_name not in names:
                            continue
                        try:
                            tex_bytes = zin.read(tex_name)
                            hdr = tex_bytes[:128]
                            tw, th = _read_dds_dimensions_from_bytes(hdr)
                            fourcc = _read_dds_fourcc_from_bytes(hdr) or ""
                        except Exception:
                            continue

                        dbg = _make_big_label_debug_image(size=(tw, th), label=tex_name, bg_rgb=bg)
                        if fourcc == "DXT1":
                            replacements[tex_name] = build_dds_dxt1_bytes(dbg, mipmaps=not args.no_mipmaps)
                        elif fourcc == "DXT3":
                            replacements[tex_name] = build_dds_dxt3_bytes(dbg, mipmaps=not args.no_mipmaps)
                        else:
                            # DXT3/DXT5: output as DXT5 (game accepts it).
                            replacements[tex_name] = build_dds_dxt5_bytes(dbg, mipmaps=not args.no_mipmaps)

            _build_reskinned_mod_zip(
                base_zip_path=base_zip_path,
                out_zip_path=out_zip_path,
                replacements=replacements,
            )
            print(f"Wrote: {out_zip_path}")
            return 0

        # Build base image (optional template underlay)
        if args.template:
            base = Image.open(args.template).convert("RGBA").resize((diffuse_w, diffuse_h), Image.Resampling.LANCZOS)
        else:
            base = Image.new("RGBA", (diffuse_w, diffuse_h), (0, 0, 0, 0))

        # Livery layer (generated)
        wants_team = bool(args.logo or args.team_name or args.tag)
        wants_team = wants_team or bool(getattr(args, "sticker", None)) or bool(getattr(args, "wheel_sticker", None)) or bool(getattr(args, "flag", None)) or bool(getattr(args, "plate_text", None))
        # Wing text/stickers/sponsors should force team mode (otherwise we'd render the demo skin).
        wants_team = wants_team or bool(getattr(args, "wing_sticker", None))
        wants_team = wants_team or bool(getattr(args, "wing_text", None))
        wants_team = wants_team or bool(getattr(args, "sponsor", None))
        # Parse wing text: if --wing-text is given, use it for wing_top_text (and leave bottom empty).
        if args.wing_text:
            wing_top = args.wing_text
            wing_bottom = ""  # explicitly empty so it doesn't fall back to team_name
        else:
            wing_top = None  # default: use tag fallback
            wing_bottom = None  # default: use team_name fallback

        # If text is disabled and the user didn't explicitly request wing text, suppress wing text entirely.
        if bool(getattr(args, "no_text", False)) and (not args.wing_text):
            wing_top = ""
            wing_bottom = ""

        # If a wing sticker is requested, suppress wing text unless explicitly kept.
        if wing_sticker_img is not None and (not bool(getattr(args, "wing_sticker_keep_text", False))):
            wing_top = ""
            wing_bottom = ""

        # Per-skin variation seed (stable by default; override with --seed).
        if args.seed is not None:
            skin_seed = int(args.seed)
        else:
            seed_src = (
                f"{args.name}|{args.style}|{args.palette or ''}|"
                f"{args.base_color}|{args.accent_color}|{args.stripe_color}|"
                f"{wing_top or ''}|{wing_bottom or ''}"
            )
            skin_seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)

        wheel_rgb = (args.wheel_color[0], args.wheel_color[1], args.wheel_color[2]) if args.wheel_color else None

        # Load CarGeometry for spatial-aware mode
        car_geo = None
        if getattr(args, "spatial_aware", False) and HAS_CAR_GEOMETRY:
            try:
                atlas_path = Path(__file__).parent / "out" / "uv_atlas" / "standard_stadium_islands_2048.json"
                if atlas_path.exists():
                    car_geo = CarGeometry.from_json_file(str(atlas_path))
                    print(f"NOTE: Spatial-aware mode enabled, loaded {len(car_geo.islands)} UV islands.")
                else:
                    print(f"WARNING: --spatial-aware requested but atlas not found at {atlas_path}")
            except Exception as e:
                print(f"WARNING: Failed to load CarGeometry: {e}")

        if wants_team:
            if diffuse_w == diffuse_h:
                demo = generate_team_skin(
                    size=diffuse_w,
                    team_name=team_name,
                    tag=tag,
                    logo=logo,
                    base_color=args.base_color,
                    accent_color=args.accent_color,
                    stripe_color=args.stripe_color,
                    style=args.style,
                    feature_rects=feature_rects,
                    nose_logo=args.nose_logo,
                    wing_top_text=wing_top,
                    wing_bottom_text=wing_bottom,
                    wing_sticker=wing_sticker_img,
                    wing_sticker_fit=str(getattr(args, "wing_sticker_fit", "contain")),
                    wing_sticker_scale=float(getattr(args, "wing_sticker_scale", 1.0)),
                    wing_sticker_opacity=float(getattr(args, "wing_sticker_opacity", 1.0)),
                    palette_name=args.palette,
                    finish_alpha=str(getattr(args, "finish_alpha", "auto")),
                    finish_neutral=int(getattr(args, "finish_neutral", 0x8E)),
                    finish_invert=bool(getattr(args, "finish_invert", False)),
                    finish_design=str(getattr(args, "finish_design", "off")),
                    finish_design_strength=float(getattr(args, "finish_design_strength", 0.35)),
                    mudguards=bool(getattr(args, "mudguards", True)),
                    mudguards_color=getattr(args, "mudguards_color", None),
                    mudguards_mode=str(getattr(args, "mudguards_mode", "darken")),
                    mudguards_strength=float(getattr(args, "mudguards_strength", 0.85)),
                    mudguards_feather=int(getattr(args, "mudguards_feather", 3)),
                    spatial_aware=bool(getattr(args, "spatial_aware", False)),
                    spatial_accent_parts=getattr(args, "spatial_accent_parts", None),
                    spatial_secondary_parts=getattr(args, "spatial_secondary_parts", None),
                    car_geometry=car_geo,
                    seed=skin_seed,
                    wheel_rgb=wheel_rgb,
                    inspire_zip=Path(args.inspire_zip) if args.inspire_zip else None,
                    inspire_source=args.inspire_source,
                    inspire_strength=args.inspire_strength,
                    logo_layout=args.logo_layout,
                    logo_plate=args.logo_plate,
                    logo_scale=args.logo_scale,
                    sidepod_branding=bool(args.sidepod_branding),
                    sidepod_tag_text=args.sidepod_tag_text,
                    sidepod_team_text=args.sidepod_team_text,
                    sidepod_branding_scale=float(args.sidepod_branding_scale),
                    sidepod_branding_mirror=str(args.sidepod_branding_mirror),
                    sticker_images=sticker_images or None,
                    sticker_count=int(args.sticker_count) if int(args.sticker_count) > 0 else (int(220 * (diffuse_w / 2048.0) * (diffuse_h / 2048.0)) if sticker_images else 0),
                    sticker_min_scale=float(args.sticker_min_scale),
                    sticker_max_scale=float(args.sticker_max_scale),
                    sticker_rotate=not bool(args.sticker_no_rotate),
                    sticker_mode=str(args.sticker_mode),
                    sticker_scope=str(getattr(args, "sticker_scope", "used")),
                    sponsor_images=sponsor_images or None,
                    sponsor_scale=float(getattr(args, "sponsor_scale", 1.0)),
                    sponsor_opacity=float(getattr(args, "sponsor_opacity", 1.0)),
                    sponsor_slots=sponsor_slots or None,
                    wheel_sticker=wheel_sticker_img,
                    wheel_sticker_scale=float(args.wheel_sticker_scale),
                    wheel_sticker_scope=str(args.wheel_sticker_scope),
                    flag=args.flag,
                    flag_location=str(args.flag_location),
                    flag_scale=float(args.flag_scale),
                    plate_text=args.plate_text if (force_diffuse_plate or (not base_has_details)) else None,
                    plate_scale=float(args.plate_scale),
                    grade_contrast=args.grade_contrast,
                    grade_color=args.grade_color,
                    grade_gamma=args.grade_gamma,
                    vignette_strength=args.vignette_strength,
                )
            else:
                demo = generate_team_skin(
                    size=max(diffuse_w, diffuse_h),
                    team_name=team_name,
                    tag=tag,
                    logo=logo,
                    base_color=args.base_color,
                    accent_color=args.accent_color,
                    stripe_color=args.stripe_color,
                    style=args.style,
                    feature_rects=feature_rects,
                    nose_logo=args.nose_logo,
                    palette_name=args.palette,
                    finish_alpha=str(getattr(args, "finish_alpha", "auto")),
                    finish_neutral=int(getattr(args, "finish_neutral", 0x8E)),
                    finish_invert=bool(getattr(args, "finish_invert", False)),
                    finish_design=str(getattr(args, "finish_design", "off")),
                    finish_design_strength=float(getattr(args, "finish_design_strength", 0.35)),
                    mudguards=bool(getattr(args, "mudguards", True)),
                    mudguards_color=getattr(args, "mudguards_color", None),
                    mudguards_mode=str(getattr(args, "mudguards_mode", "darken")),
                    mudguards_strength=float(getattr(args, "mudguards_strength", 0.85)),
                    mudguards_feather=int(getattr(args, "mudguards_feather", 3)),
                    spatial_aware=bool(getattr(args, "spatial_aware", False)),
                    spatial_accent_parts=getattr(args, "spatial_accent_parts", None),
                    spatial_secondary_parts=getattr(args, "spatial_secondary_parts", None),
                    car_geometry=car_geo,
                    wing_top_text=wing_top,
                    wing_bottom_text=wing_bottom,
                    wing_sticker=wing_sticker_img,
                    wing_sticker_fit=str(getattr(args, "wing_sticker_fit", "contain")),
                    wing_sticker_scale=float(getattr(args, "wing_sticker_scale", 1.0)),
                    wing_sticker_opacity=float(getattr(args, "wing_sticker_opacity", 1.0)),
                    seed=skin_seed,
                    wheel_rgb=wheel_rgb,
                    inspire_zip=Path(args.inspire_zip) if args.inspire_zip else None,
                    inspire_source=args.inspire_source,
                    inspire_strength=args.inspire_strength,
                    logo_layout=args.logo_layout,
                    logo_plate=args.logo_plate,
                    logo_scale=args.logo_scale,
                    sidepod_branding=bool(args.sidepod_branding),
                    sidepod_tag_text=args.sidepod_tag_text,
                    sidepod_team_text=args.sidepod_team_text,
                    sidepod_branding_scale=float(args.sidepod_branding_scale),
                    sidepod_branding_mirror=str(args.sidepod_branding_mirror),
                    sticker_images=sticker_images or None,
                    sticker_count=int(args.sticker_count) if int(args.sticker_count) > 0 else (int(220 * (diffuse_w / 2048.0) * (diffuse_h / 2048.0)) if sticker_images else 0),
                    sticker_min_scale=float(args.sticker_min_scale),
                    sticker_max_scale=float(args.sticker_max_scale),
                    sticker_rotate=not bool(args.sticker_no_rotate),
                    sticker_mode=str(args.sticker_mode),
                    sticker_scope=str(getattr(args, "sticker_scope", "used")),
                    sponsor_images=sponsor_images or None,
                    sponsor_scale=float(getattr(args, "sponsor_scale", 1.0)),
                    sponsor_opacity=float(getattr(args, "sponsor_opacity", 1.0)),
                    sponsor_slots=sponsor_slots or None,
                    wheel_sticker=wheel_sticker_img,
                    wheel_sticker_scale=float(args.wheel_sticker_scale),
                    wheel_sticker_scope=str(args.wheel_sticker_scope),
                    flag=args.flag,
                    flag_location=str(args.flag_location),
                    flag_scale=float(args.flag_scale),
                    plate_text=args.plate_text if (force_diffuse_plate or (not base_has_details)) else None,
                    plate_scale=float(args.plate_scale),
                    grade_contrast=args.grade_contrast,
                    grade_color=args.grade_color,
                    grade_gamma=args.grade_gamma,
                    vignette_strength=args.vignette_strength,
                ).resize((diffuse_w, diffuse_h), Image.Resampling.LANCZOS)
        else:
            if diffuse_w == diffuse_h:
                demo = generate_demo_skin(
                    size=diffuse_w,
                    name=args.name,
                    base_color=args.base_color,
                    accent_color=args.accent_color,
                    stripe_color=args.stripe_color,
                )
            else:
                demo = generate_demo_skin(
                    size=max(diffuse_w, diffuse_h),
                    name=args.name,
                    base_color=args.base_color,
                    accent_color=args.accent_color,
                    stripe_color=args.stripe_color,
                ).resize((diffuse_w, diffuse_h), Image.Resampling.LANCZOS)

        # IMPORTANT: In TMNF car mod packs, Diffuse alpha is commonly a material/spec channel,
        # not transparency. Using Image.alpha_composite() here would incorrectly treat that
        # channel as transparency and "leak" the base skin colors (often gray) into the output.
        #
        # So: composite using an OPAQUE version of the generated RGB, then restore the desired alpha.
        demo_alpha = demo.getchannel("A")
        demo_rgb = demo.copy()
        demo_rgb.putalpha(255)
        used_mask = None
        if isinstance(feature_rects, dict):
            um = feature_rects.get("used_mask")
            if isinstance(um, Image.Image):
                used_mask = um.convert("L")
        if used_mask is not None:
            img = Image.composite(demo_rgb, base, used_mask)
        else:
            # Without a mask, fully replace base diffuse RGB.
            img = demo_rgb
        img.putalpha(demo_alpha)

        if args.prelight:
            pre = Image.open(args.prelight)
            img = _apply_prelight(img, pre, strength=args.prelight_strength)

        if args.bw_only:
            # Convert RGB to strict black/white (preserve alpha/spec).
            a = img.getchannel("A")
            l = img.convert("L")
            bw = l.point(lambda p: 255 if p >= 128 else 0)
            img = Image.merge("RGBA", (bw, bw, bw, a))

        if base_alpha_const is not None:
            img.putalpha(base_alpha_const)

        # Sanitize output to reduce visible banding on dark gradients (outer body).
        if getattr(args, "sanitize", True):
            try:
                # Keep alpha/spec exactly as-is.
                a = img.getchannel("A")
                img2 = _deband_dither_rgba(img, seed=int(skin_seed), amp=2, lum_lo=8.0, lum_hi=155.0, grad_hi=10.0)
                img2.putalpha(a)
                img = img2
            except Exception:
                pass

        # DXT-friendly edge sharpening (helps thin cutlines survive DXT5 + mipmaps).
        try:
            mode = str(getattr(args, "dxt_sharpen", "auto") or "auto").lower()
            want = (mode == "on") or (mode == "auto" and out_dds_format == "dxt5")
            if want:
                img = _dxt_edge_sharpen_rgba(
                    img,
                    strength=float(getattr(args, "dxt_sharpen_strength", 0.35)),
                    radius=float(getattr(args, "dxt_sharpen_radius", 1.2)),
                    percent=int(getattr(args, "dxt_sharpen_percent", 140)),
                    threshold=int(getattr(args, "dxt_sharpen_threshold", 6)),
                )
        except Exception:
            pass

        if out_dds_format == "dxt5":
            print("NOTE: Writing Diffuse.dds as DXT5 (matches most TMNF mod packs).")
            diffuse_dds_bytes = build_dds_dxt5_bytes(img, mipmaps=not args.no_mipmaps)
        else:
            print("NOTE: Writing Diffuse.dds as RGBA8 (uncompressed).")
            diffuse_dds_bytes = build_dds_rgba8_bytes(img, mipmaps=not args.no_mipmaps)

        out_zip_path = out_dir / f"{args.name}.zip"
        replacements: Dict[str, bytes] = {"Diffuse.dds": diffuse_dds_bytes}
        additions: Dict[str, bytes] = {}

        base_names = _read_zip_names(base_zip_path)
        _warn_if_base_zip_looks_suspicious(base_zip_path, base_names)
        _ensure_stadium_aux_textures(base_names=base_names, additions=additions, mipmaps=not args.no_mipmaps)

        # Optional: user-provided Illum override (baked glow maps for complex Details UVs).
        if getattr(args, "illum_image", None):
            try:
                mode, payload = _build_illum_override_dds(
                    base_zip_path=base_zip_path,
                    illum_image_path=str(args.illum_image),
                    mipmaps=not args.no_mipmaps,
                )
                if mode == "replace":
                    replacements["Illum.dds"] = payload
                else:
                    additions["Illum.dds"] = payload
            except Exception:
                pass

        # Recolor the base pack glow (wheels/neon) to match your skin palette.
        if wants_team and (not args.no_recolor_glow):
            base_rgb = (args.base_color[0], args.base_color[1], args.base_color[2])
            accent_rgb = (args.accent_color[0], args.accent_color[1], args.accent_color[2])
            stripe_rgb = (args.stripe_color[0], args.stripe_color[1], args.stripe_color[2])
            wheel_rgb = (args.wheel_color[0], args.wheel_color[1], args.wheel_color[2]) if args.wheel_color else None
            # Default wheel color = stripe if set, otherwise accent.
            target_rgb = wheel_rgb or (stripe_rgb if args.stripe_color[3] > 0 else accent_rgb)
            for tex_name in ["Illum.dds", "Details.dds", "DetailsDirty.dds", "ProjShad.dds", "DiffuseDirty.dds"]:
                try:
                    with zipfile.ZipFile(base_zip_path, "r") as zin:
                        tex_bytes = zin.read(tex_name)
                except KeyError:
                    continue
                try:
                    tex_img = Image.open(io.BytesIO(tex_bytes)).convert("RGBA")
                except Exception:
                    continue
                recolored = _recolor_teal_glow_like_tron(tex_img, target_rgb=target_rgb)
                # Also recolor any non-gray accent pixels (useful for Deep Galaxy wheel rings etc.)
                recolored = _recolor_non_gray_accents(recolored, target_rgb=target_rgb, strength=0.9)

                # Keep original compression type if possible.
                hdr = tex_bytes[:128]
                fourcc = _read_dds_fourcc_from_bytes(hdr) or ""
                if fourcc == "DXT1":
                    replacements[tex_name] = build_dds_dxt1_bytes(recolored, mipmaps=not args.no_mipmaps)
                elif fourcc == "DXT3":
                    # Preserve DXT3 (common for Dirty textures in TMNF packs).
                    replacements[tex_name] = build_dds_dxt3_bytes(recolored, mipmaps=not args.no_mipmaps)
                elif fourcc == "DXT5":
                    replacements[tex_name] = build_dds_dxt5_bytes(recolored, mipmaps=not args.no_mipmaps)
                else:
                    # Fallback: also use DXT5 to avoid uncompressed textures (TMNF dislikes them).
                    replacements[tex_name] = build_dds_dxt5_bytes(recolored, mipmaps=not args.no_mipmaps)

        # Optional: custom projection texture (ProjShad.dds)
        if getattr(args, "proj_image", None):
            try:
                with zipfile.ZipFile(base_zip_path, "r") as zin:
                    if "ProjShad.dds" in zin.namelist():
                        hdr = zin.open("ProjShad.dds").read(128)
                        pw, ph = _read_dds_dimensions_from_bytes(hdr)
                        fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT1"
                    else:
                        pw, ph, fourcc = (512, 512, "DXT1")
            except Exception:
                pw, ph, fourcc = (512, 512, "DXT1")

            try:
                pi = Image.open(args.proj_image).convert("RGBA").resize((pw, ph), Image.Resampling.LANCZOS)  # type: ignore[arg-type]
                # ProjShad is a projection map; tutorials generally use white background + dark marks,
                # no alpha, and a horizontal flip (projection reads mirrored).
                bg = Image.new("RGB", (pw, ph), (255, 255, 255))
                try:
                    bg.paste(pi.convert("RGB"), (0, 0), pi.getchannel("A"))
                except Exception:
                    bg = pi.convert("RGB")
                proj_rgb = _finalize_proj_shad_rgb(bg)
                if fourcc == "DXT5":
                    proj_rgba = proj_rgb.convert("RGBA")
                    proj_rgba.putalpha(255)
                    replacements["ProjShad.dds"] = build_dds_dxt5_bytes(proj_rgba, mipmaps=not args.no_mipmaps)
                else:
                    replacements["ProjShad.dds"] = build_dds_dxt1_bytes(proj_rgb, mipmaps=not args.no_mipmaps)
            except Exception:
                pass

        elif bool(getattr(args, "proj_wings", False)):
            try:
                try:
                    with zipfile.ZipFile(base_zip_path, "r") as zin:
                        if "ProjShad.dds" in zin.namelist():
                            hdr = zin.open("ProjShad.dds").read(128)
                            pw, ph = _read_dds_dimensions_from_bytes(hdr)
                            fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT1"
                        else:
                            pw, ph, fourcc = (512, 512, "DXT1")
                except Exception:
                    pw, ph, fourcc = (512, 512, "DXT1")

                # Generate wings and apply tutorial hygiene (mirror + white border).
                proj_rgb = _finalize_proj_shad_rgb(_make_proj_shad_wings(size=pw, darkness=18))
                if fourcc == "DXT5":
                    proj_rgba = proj_rgb.convert("RGBA")
                    proj_rgba.putalpha(255)
                    replacements["ProjShad.dds"] = build_dds_dxt5_bytes(proj_rgba, mipmaps=not args.no_mipmaps)
                else:
                    replacements["ProjShad.dds"] = build_dds_dxt1_bytes(proj_rgb, mipmaps=not args.no_mipmaps)
            except Exception:
                pass

        elif args.proj_logo and wants_team and logo is not None:
            try:
                with zipfile.ZipFile(base_zip_path, "r") as zin:
                    if "ProjShad.dds" in zin.namelist():
                        hdr = zin.open("ProjShad.dds").read(128)
                        pw, ph = _read_dds_dimensions_from_bytes(hdr)
                        fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT1"
                    else:
                        pw, ph, fourcc = (512, 512, "DXT1")
            except Exception:
                pw, ph, fourcc = (512, 512, "DXT1")

            proj_img = _make_proj_shad_from_logo(
                size=pw,
                logo=logo,
                rgb=(args.accent_color[0], args.accent_color[1], args.accent_color[2]),
                invert=False,
            )
            proj_rgb = _finalize_proj_shad_rgb(proj_img)
            if fourcc == "DXT5":
                proj_rgba = proj_rgb.convert("RGBA")
                proj_rgba.putalpha(255)
                replacements["ProjShad.dds"] = build_dds_dxt5_bytes(proj_rgba, mipmaps=not args.no_mipmaps)
            else:
                replacements["ProjShad.dds"] = build_dds_dxt1_bytes(proj_rgb, mipmaps=not args.no_mipmaps)

        # License plate text: in many packs the actual plate surface is on Details.dds (not Diffuse.dds).
        # If Details exists, stamp there (and we avoid Diffuse stamping above).
        # If the user explicitly selected a Diffuse plate island (--plate-island), don't also stamp on Details.
        if wants_team and args.plate_text and base_has_details and (not (getattr(args, "plate_island", None) and args.plate_island)):
            try:
                # Use the recolored version if present; otherwise load from base zip.
                if "Details.dds" in replacements:
                    tex_bytes = replacements["Details.dds"]
                else:
                    with zipfile.ZipFile(base_zip_path, "r") as zin:
                        tex_bytes = zin.read("Details.dds")

                hdr = tex_bytes[:128]
                det_fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT5"
                det_img = Image.open(io.BytesIO(tex_bytes)).convert("RGBA")
                det_alpha = det_img.getchannel("A")

                rects = _find_details_plate_rects(det_img, downscale_to=1024, luma_threshold=160, max_count=2)
                if rects:
                    # Draw with opaque alpha to avoid premultiplied artifacts, then restore alpha.
                    work = det_img.copy()
                    work.putalpha(255)
                    txt = str(args.plate_text).strip()
                    try:
                        ps = float(getattr(args, "plate_scale", 0.92))
                    except Exception:
                        ps = 0.92
                    ps = max(0.50, min(1.25, ps))

                    for (x0, y0, x1, y1) in rects:
                        rw = max(1, x1 - x0)
                        rh = max(1, y1 - y0)
                        # Slight inset; keep border visible.
                        pad = max(2, int(rh * 0.12))
                        rect = (x0 + pad, y0 + pad, x1 - pad, y1 - pad)

                        layer = Image.new("RGBA", work.size, (0, 0, 0, 0))
                        d = ImageDraw.Draw(layer)
                        rr = max(2, int((rect[3] - rect[1]) * 0.48))
                        # Plate background + border (high contrast).
                        d.rounded_rectangle(
                            rect,
                            radius=rr,
                            fill=(0, 0, 0, int(185 * min(1.0, ps))),
                            outline=(245, 245, 245, 245),
                            width=max(2, int((rect[3] - rect[1]) * 0.16)),
                        )
                        _draw_centered_text_in_rect(
                            layer,
                            rect=rect,
                            text=txt,
                            fill=(245, 245, 245, 245),
                            stroke_fill=(0, 0, 0, 235),
                            max_font_px=max(10, int((rect[3] - rect[1]) * 0.80 * ps)),
                            rotate_degrees=0,
                        )
                        work = Image.alpha_composite(work, layer)

                    work.putalpha(det_alpha)
                    if det_fourcc == "DXT1":
                        replacements["Details.dds"] = build_dds_dxt1_bytes(work, mipmaps=not args.no_mipmaps)
                    elif det_fourcc == "DXT3":
                        replacements["Details.dds"] = build_dds_dxt3_bytes(work, mipmaps=not args.no_mipmaps)
                    else:
                        replacements["Details.dds"] = build_dds_dxt5_bytes(work, mipmaps=not args.no_mipmaps)
            except Exception:
                # If anything fails, keep base details.
                pass

        # Icon replacement (if present in base zip)
        if (not args.no_icon) and wants_team:
            try:
                with zipfile.ZipFile(base_zip_path, "r") as zin, zin.open("Icon.dds", "r") as f:
                    icon_hdr = f.read(128)
                icon_w, icon_h = _read_dds_dimensions_from_bytes(icon_hdr)
                icon_fourcc = _read_dds_fourcc_from_bytes(icon_hdr)
                icon_img = generate_team_icon(
                    size=icon_w,
                    team_name=team_name,
                    tag=tag,
                    logo=logo,
                    base_color=args.base_color,
                    accent_color=args.accent_color,
                    stripe_color=args.stripe_color,
                )
                if args.bw_only:
                    a = icon_img.getchannel("A")
                    l = icon_img.convert("L")
                    bw = l.point(lambda p: 255 if p >= 128 else 0)
                    icon_img = Image.merge("RGBA", (bw, bw, bw, a))

                if icon_fourcc == "DXT1":
                    icon_dds_bytes = build_dds_dxt1_bytes(icon_img, mipmaps=not args.no_mipmaps)
                elif icon_fourcc == "DXT5":
                    icon_dds_bytes = build_dds_dxt5_bytes(icon_img, mipmaps=not args.no_mipmaps)
                else:
                    icon_dds_bytes = build_dds_rgba8_bytes(icon_img, mipmaps=not args.no_mipmaps)

                replacements["Icon.dds"] = icon_dds_bytes

                if args.preview_png:
                    icon_preview_path = out_dir / f"{args.name}_Icon.png"
                    icon_img.save(icon_preview_path)
                    print(f"Wrote: {icon_preview_path}")
            except KeyError:
                # No Icon.dds in base zip; ignore.
                pass

        # Sanitize base-pack textures (best-effort): remove common watermarks from Details* and
        # reduce banding in DiffuseDirty if present.
        if getattr(args, "sanitize", True):
            try:
                with zipfile.ZipFile(base_zip_path, "r") as zin:
                    base_names = set(zin.namelist())

                    # Details watermarks
                    for tex_name in ("Details.dds", "DetailsDirty.dds"):
                        if (tex_name not in base_names) and (tex_name not in replacements):
                            continue
                        try:
                            src_bytes = replacements.get(tex_name, zin.read(tex_name))
                            hdr = src_bytes[:128]
                            fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT5"
                            tex = Image.open(io.BytesIO(src_bytes)).convert("RGBA")
                            a = tex.getchannel("A")
                            tex2 = _sanitize_common_details_watermarks(tex)
                            tex2.putalpha(a)
                            if fourcc == "DXT1":
                                replacements[tex_name] = build_dds_dxt1_bytes(tex2, mipmaps=not args.no_mipmaps)
                            elif fourcc == "DXT3":
                                replacements[tex_name] = build_dds_dxt3_bytes(tex2, mipmaps=not args.no_mipmaps)
                            else:
                                replacements[tex_name] = build_dds_dxt5_bytes(tex2, mipmaps=not args.no_mipmaps)
                        except Exception:
                            continue

                    # DiffuseDirty deband (outer-body banding shows up strongly on fade packs)
                    for tex_name in ("DiffuseDirty.dds",):
                        if (tex_name not in base_names) and (tex_name not in replacements):
                            continue
                        try:
                            src_bytes = replacements.get(tex_name, zin.read(tex_name))
                            hdr = src_bytes[:128]
                            fourcc = _read_dds_fourcc_from_bytes(hdr) or "DXT5"
                            tex = Image.open(io.BytesIO(src_bytes)).convert("RGBA")
                            a = tex.getchannel("A")
                            tex2 = _deband_dither_rgba(
                                tex,
                                seed=int(skin_seed) ^ 0xD1FF11,
                                amp=2,
                                lum_lo=8.0,
                                lum_hi=170.0,
                                grad_hi=10.0,
                            )
                            tex2.putalpha(a)
                            if fourcc == "DXT1":
                                replacements[tex_name] = build_dds_dxt1_bytes(tex2, mipmaps=not args.no_mipmaps)
                            elif fourcc == "DXT3":
                                replacements[tex_name] = build_dds_dxt3_bytes(tex2, mipmaps=not args.no_mipmaps)
                            else:
                                replacements[tex_name] = build_dds_dxt5_bytes(tex2, mipmaps=not args.no_mipmaps)
                        except Exception:
                            continue
            except Exception:
                pass

        # ── Paint wheel / tyre colour on Details.dds ────────────────────────
        # In TMNF/TMUF the 3D model's wheel meshes (d-prefixed parts) read
        # their base colour from Details.dds, NOT Diffuse.dds.  Our pipeline
        # already paints Diffuse.dds for the body; this block ensures the
        # wheel/tyre UV islands in Details.dds also get the requested colour.
        # Standard Stadium Car UV island bounding boxes (at 2048 px reference):
        _WHEEL_ISLAND_BBOXES_2048: Dict[int, Tuple[int, int, int, int]] = {
            10: (1940, 1084, 2024, 1412),
            11: ( 780, 1724,  996, 1840),
            12: ( 780,  320,  996,  436),
            13: (1520, 1152, 1604, 1440),
            14: (1104,  608, 1356,  728),
            15: (1104, 1432, 1356, 1552),
            16: ( 264, 1252,  396, 1400),
            17: ( 264,  760,  396,  908),
            18: (1152, 1188, 1412, 1272),
            19: (1152,  888, 1412,  972),
            20: ( 788,  200,  988,  304),
            21: ( 812,   36,  988,  180),
            22: (1428, 1832, 1600, 2032),
        }
        if wants_team and base_has_details and np is not None:
            _det_wheel_rgb = None
            _det_wheel_alpha = 50  # moderate shininess (similar to GolfMaster reference)
            if args.wheel_color:
                _det_wheel_rgb = (args.wheel_color[0], args.wheel_color[1], args.wheel_color[2])
            elif wants_team:
                # Default: use accent colour for wheels
                _det_wheel_rgb = (args.accent_color[0], args.accent_color[1], args.accent_color[2])
            if _det_wheel_rgb is not None:
                try:
                    if "Details.dds" in replacements:
                        _det_bytes = replacements["Details.dds"]
                    else:
                        with zipfile.ZipFile(base_zip_path, "r") as _zin:
                            _det_bytes = _zin.read("Details.dds")
                    _det_hdr = _det_bytes[:128]
                    _det_fourcc = _read_dds_fourcc_from_bytes(_det_hdr) or "DXT5"
                    _det_img = Image.open(io.BytesIO(_det_bytes)).convert("RGBA")
                    _dw, _dh = _det_img.size
                    _det_arr = np.array(_det_img)  # (H, W, 4)
                    _scale = _dw / 2048.0

                    for _iid, _bbox in _WHEEL_ISLAND_BBOXES_2048.items():
                        x0 = int(_bbox[0] * _scale)
                        y0 = int(_bbox[1] * _scale)
                        x1 = int(_bbox[2] * _scale)
                        y1 = int(_bbox[3] * _scale)
                        x1 = min(x1, _dw)
                        y1 = min(y1, _dh)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        crop = _det_arr[y0:y1, x0:x1]
                        # Mask: paint only where existing pixels are non-black
                        # (respects the UV geometry – black = unused UV space)
                        brightness = crop[:, :, :3].max(axis=2)
                        mask = brightness > 3  # threshold: anything brighter than near-black
                        if not mask.any():
                            continue
                        crop[mask, 0] = _det_wheel_rgb[0]
                        crop[mask, 1] = _det_wheel_rgb[1]
                        crop[mask, 2] = _det_wheel_rgb[2]
                        crop[mask, 3] = _det_wheel_alpha
                        _det_arr[y0:y1, x0:x1] = crop

                    _det_out = Image.fromarray(_det_arr, "RGBA")
                    if _det_fourcc == "DXT1":
                        replacements["Details.dds"] = build_dds_dxt1_bytes(_det_out, mipmaps=not args.no_mipmaps)
                    elif _det_fourcc == "DXT3":
                        replacements["Details.dds"] = build_dds_dxt3_bytes(_det_out, mipmaps=not args.no_mipmaps)
                    else:
                        replacements["Details.dds"] = build_dds_dxt5_bytes(_det_out, mipmaps=not args.no_mipmaps)
                except Exception:
                    pass

        _build_reskinned_mod_zip(
            base_zip_path=base_zip_path,
            out_zip_path=out_zip_path,
            replacements=replacements,
            additions=additions,
        )

        if args.preview_png:
            preview_path = out_dir / f"{args.name}_Diffuse.png"
            img.save(preview_path)
            print(f"Wrote: {preview_path}")

        print(f"Wrote: {out_zip_path}")
        return 0

    out_dds_format = args.dds_format if args.dds_format != "auto" else "rgba8"
    skin_dir = out_dir / args.name
    dds_path = skin_dir / args.texture_filename

    # Build base image
    if args.template:
        base = Image.open(args.template).convert("RGBA").resize((args.size, args.size), Image.Resampling.LANCZOS)
    else:
        base = Image.new("RGBA", (args.size, args.size), (0, 0, 0, 0))

    demo = generate_demo_skin(
        size=args.size,
        name=args.name,
        base_color=args.base_color,
        accent_color=args.accent_color,
        stripe_color=args.stripe_color,
    )
    img = Image.alpha_composite(base, demo)

    if args.prelight:
        pre = Image.open(args.prelight)
        img = _apply_prelight(img, pre, strength=args.prelight_strength)

    # Write DDS
    if out_dds_format == "dxt5":
        save_dds_dxt5(dds_path, img, mipmaps=not args.no_mipmaps)
    else:
        save_dds_rgba8(dds_path, img, mipmaps=not args.no_mipmaps)

    if args.preview_png:
        png_path = dds_path.with_suffix(".png")
        img.save(png_path)

    if args.zip_skin:
        zip_path = out_dir / f"{args.name}.zip"
        if args.zip_layout == "folder":
            arc_dds = f"{args.name}/{args.texture_filename}"
            files = [(dds_path, arc_dds)]
            if args.preview_png:
                files.append((dds_path.with_suffix(".png"), f"{args.name}/{args.texture_filename[:-4]}.png"))
        else:
            files = [(dds_path, args.texture_filename)]
            if args.preview_png:
                files.append((dds_path.with_suffix(".png"), dds_path.with_suffix(".png").name))
        _zip_skin(zip_path=zip_path, skin_dir_name=args.name, files=files)

    print(f"Wrote: {dds_path}")
    if args.zip_skin:
        print(f"Wrote: {out_dir / (args.name + '.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


