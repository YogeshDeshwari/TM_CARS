"""
Shared helpers for the gaming-aesthetic skin series
(Tacnet, Ghost Protocol, Tenno Sigil, Shardform, Ember Void).

Provides:
- save_with_tires: extends SkinCanvas.save() with tire_customizer integration
- load_mono_font / load_stencil_font: resolved TTF paths
- island_union_mask: union of all painted (non-neutral-cockpit) islands
- island_edge_points: sample boundary points of an island
- sdf_from_mask: signed-distance field via scipy
- hsv_to_rgb: numpy-vectorized color conversion
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skin_canvas import SkinCanvas  # noqa: E402


# --- Font loading ----------------------------------------------------------

_MONO_CANDIDATES = [
    Path.home() / "Library/Fonts/DejaVuSansMono-Bold.ttf",
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/System/Library/Fonts/Supplemental/Courier New Bold.ttf"),
]

_STENCIL_CANDIDATES = [
    Path.home() / "Library/Fonts/DejaVuSansCondensed-Bold.ttf",
    Path.home() / "Library/Fonts/DejaVuSans-Bold.ttf",
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
]


def load_mono_font(px: int) -> ImageFont.FreeTypeFont:
    for p in _MONO_CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), px)
    return ImageFont.load_default()


def load_stencil_font(px: int) -> ImageFont.FreeTypeFont:
    for p in _STENCIL_CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), px)
    return ImageFont.load_default()


# --- Robust island mask extraction ----------------------------------------
# The library's _build_island_masks (in both SkinCanvas and ProSkinEngine)
# samples the center pixel of the JSON bbox to pick each island's color.
# For several islands (7, 9, 10, ...) that center falls on a white label
# frame, so the mask ends up matching white everywhere in the diagnostic
# image and is unusable.  This helper rebuilds masks by sampling the most
# common *colored* (non-white, non-background) pixel inside the bbox.

_DIAG_PATH = Path(__file__).resolve().parent.parent / \
    "assets/uv_atlas/diagnostics_2048.png"
_ATLAS_JSON = Path(__file__).resolve().parent.parent / \
    "assets/uv_atlas/standard_stadium_islands_2048.json"


def _robust_island_color(diag: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Pick the most-common colored pixel inside bbox, ignoring BG and frame."""
    x0, y0, x1, y1 = bbox
    region = diag[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    # Exclude near-BG (dark) and near-white (frames/labels).
    lum = region.sum(axis=1)
    chroma = region.max(axis=1) - region.min(axis=1)
    keep = (lum > 40) & ((lum < 720) | (chroma > 40))
    if not keep.any():
        return None
    pix = region[keep]
    # Quantize to 8-bit bins to find mode.
    keys = (pix[:, 0] << 16) | (pix[:, 1] << 8) | pix[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode_key = vals[np.argmax(counts)]
    return np.array([(mode_key >> 16) & 0xFF,
                     (mode_key >> 8) & 0xFF,
                     mode_key & 0xFF], dtype=int)


def build_robust_masks(size: int = 2048) -> Dict[int, Image.Image]:
    """Build per-island pixel masks that correctly handle label-frame centers."""
    import json
    diag = np.array(Image.open(_DIAG_PATH).convert("RGB"))
    atlas = json.loads(_ATLAS_JSON.read_text())

    masks: Dict[int, Image.Image] = {}
    claimed = np.zeros(diag.shape[:2], dtype=bool)
    # Sort by bbox area descending so large islands claim territory first.
    isls = sorted(
        atlas["islands"],
        key=lambda i: -((i["bbox"][2] - i["bbox"][0])
                        * (i["bbox"][3] - i["bbox"][1])),
    )
    for isl in isls:
        iid = isl["id"]
        bbox = isl["bbox"]
        color = _robust_island_color(diag, bbox)
        if color is None:
            continue
        diff = np.abs(diag.astype(int) - color).sum(axis=2)
        global_match = diff < 30
        # Restrict to this island's bbox (with small padding) so that
        # two islands sharing the same color (rare but possible) don't
        # claim each other's pixels.
        x0, y0, x1, y1 = bbox
        pad = 4
        local = np.zeros_like(global_match)
        local[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
        m = global_match & local & ~claimed
        claimed |= m
        arr = (m.astype(np.uint8) * 255)
        img = Image.fromarray(arr, "L")
        if size != 2048:
            img = img.resize((size, size), Image.Resampling.NEAREST)
        masks[iid] = img
    return masks


def install_robust_masks(sc: SkinCanvas) -> Dict[int, Image.Image]:
    """Replace sc._island_masks / role_masks with correctly-built masks."""
    sc._ensure_geo()
    masks = build_robust_masks(sc.size)
    sc._island_masks = masks
    # Rebuild role masks
    from car_geometry import ColorRole
    sc._role_masks = {}
    for role in ColorRole:
        ids = sc._geo.get_islands_by_role(role)
        combined = Image.new("L", (sc.size, sc.size), 0)
        for iid in ids:
            if iid in masks:
                combined = ImageChops.lighter(combined, masks[iid])
        sc._role_masks[role.value] = combined
    return masks


# --- Real-mask geometry helpers -------------------------------------------

def mask_bbox(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox (x0, y0, x1, y1) of non-zero mask pixels, or None if empty."""
    arr = np.array(mask)
    ys, xs = np.where(arr > 128)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_centroid(mask: Image.Image) -> Optional[Tuple[int, int]]:
    """Centroid of non-zero mask pixels, or None if empty."""
    arr = np.array(mask)
    ys, xs = np.where(arr > 128)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def mask_row_span(mask: Image.Image, y: int) -> Optional[Tuple[int, int]]:
    """Return (x_lo, x_hi) inclusive of the mask at row y, or None if empty."""
    arr = np.array(mask)
    H, W = arr.shape
    if not (0 <= y < H):
        return None
    row = arr[y, :] > 128
    xs = np.where(row)[0]
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max())


def mask_contains(mask: Image.Image, x: float, y: float) -> bool:
    arr = np.array(mask)
    H, W = arr.shape
    xi, yi = int(x), int(y)
    if not (0 <= xi < W and 0 <= yi < H):
        return False
    return bool(arr[yi, xi] > 128)


# --- Island helpers --------------------------------------------------------

def island_union_mask(sc: SkinCanvas,
                      include_neutral: bool = False) -> Image.Image:
    """Union mask of all islands whose role is paintable (default: skip NEUTRAL)."""
    sc._ensure_geo()
    combined = Image.new("L", (sc.size, sc.size), 0)
    for iid, mask in sc._island_masks.items():
        part = sc._geo.islands[iid].part
        if part is None:
            continue
        if not include_neutral and part.role.value == "neutral":
            continue
        combined = ImageChops.lighter(combined, mask)
    return combined


def island_edge_points(mask: Image.Image,
                       n_points: int,
                       rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Sample n_points random points along the boundary of a mask."""
    arr = np.array(mask)
    edges_img = mask.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges_img)
    ys, xs = np.where(edge_arr > 64)
    if len(xs) == 0:
        return []
    k = min(n_points, len(xs))
    idx = rng.choice(len(xs), size=k, replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def island_interior_points(mask: Image.Image,
                           n_points: int,
                           rng: np.random.Generator,
                           erosion_px: int = 4) -> List[Tuple[int, int]]:
    """Sample n_points inside the mask (optionally eroded to stay away from edges)."""
    if erosion_px > 0:
        m = mask.filter(ImageFilter.MinFilter(erosion_px * 2 + 1))
    else:
        m = mask
    arr = np.array(m)
    ys, xs = np.where(arr > 128)
    if len(xs) == 0:
        return []
    k = min(n_points, len(xs))
    idx = rng.choice(len(xs), size=k, replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def sdf_from_mask(mask_np: np.ndarray) -> np.ndarray:
    """Signed distance field: positive outside feature, 0 on feature."""
    # mask_np True = feature pixel. Distance from each pixel to nearest feature.
    return distance_transform_edt(~mask_np)


# --- Colour helpers --------------------------------------------------------

def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV -> RGB, all inputs in [0,1]. Returns float array [..,3]."""
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p,
        np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v,
        np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t,
        np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([r, g, b], axis=-1)


# --- Save-with-tires path --------------------------------------------------

def save_with_tires(sc: SkinCanvas, name: str, tire_config: dict,
                    *, enhance: bool = True,
                    adaptive_alpha: bool = True,
                    apply_prelight: bool = False,
                    prelight_strength: float = 0.55) -> Path:
    """Save a SkinCanvas as a zip, with custom Details.dds tire/wheel styling.

    Mirrors SkinCanvas.save() but injects tire_customizer.customize_details
    into ProSkinEngine._build_details_texture before the engine writes.
    """
    from pro_skin_engine import ProSkinEngine
    from tire_customizer import customize_details

    if apply_prelight:
        sc.apply_prelight(strength=prelight_strength)
    if adaptive_alpha:
        sc.compute_adaptive_alpha()

    engine = ProSkinEngine(team_name=name, full_skin=True)
    engine.load_uv_geometry()

    # Override engine's broken masks with our robust ones so per-island
    # finish alphas, dirt, and projection shadows all sample the right area.
    robust = build_robust_masks(engine.size)
    engine._island_masks = robust
    from car_geometry import ColorRole as _CR
    engine._role_masks = {}
    for role in _CR:
        ids = engine._geo.get_islands_by_role(role)
        combined = Image.new("L", (engine.size, engine.size), 0)
        for iid in ids:
            if iid in robust:
                combined = ImageChops.lighter(combined, robust[iid])
        engine._role_masks[role.value] = combined

    diffuse = sc.get_diffuse()
    if enhance:
        rgb = diffuse.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(1.08)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.22)
        rgb = ImageEnhance.Color(rgb).enhance(1.18)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2.0, percent=50, threshold=2))
        alpha = diffuse.getchannel("A")
        diffuse = rgb.convert("RGBA")
        diffuse.putalpha(alpha)
    engine.diffuse = diffuse

    engine._island_finish_alphas = {}
    finish_arr = np.array(sc._finish_map)
    for iid in engine._island_masks:
        mask_arr = np.array(engine._island_masks[iid])
        vals = finish_arr[mask_arr > 128]
        engine._island_finish_alphas[iid] = (
            int(np.median(vals)) if len(vals) else 0x8E
        )

    _orig = engine._build_details_texture

    def _with_tires(base_sizes):
        base = _orig(base_sizes)
        return customize_details(base, **tire_config)

    engine._build_details_texture = _with_tires

    engine.save()
    out = Path(f"out/{name}.zip")
    print(f"Skin saved: {out}")
    return out


# --- Drawing primitives ----------------------------------------------------

def draw_l_bracket(draw: ImageDraw.ImageDraw,
                   corner: Tuple[int, int],
                   dx: int, dy: int,
                   length: int, thickness: int,
                   color: Tuple[int, int, int, int]):
    """Draw an L-shaped bracket.  dx,dy in {-1,+1} indicate which corner."""
    x, y = corner
    x2, y2 = x + dx * length, y
    x3, y3 = x, y + dy * length
    draw.line([(x, y), (x2, y2)], fill=color, width=thickness)
    draw.line([(x, y), (x3, y3)], fill=color, width=thickness)


def draw_ticks_along(draw: ImageDraw.ImageDraw,
                     start: Tuple[float, float],
                     end: Tuple[float, float],
                     count: int, short: int, long: int,
                     thickness: int,
                     color: Tuple[int, int, int, int],
                     major_every: int = 5):
    """Ruler ticks along a line; every major_every tick is long."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    nx, ny = dx / length, dy / length
    px, py = -ny, nx
    for i in range(count):
        t = i / max(1, count - 1)
        cx, cy = sx + dx * t, sy + dy * t
        tlen = long if (i % major_every == 0) else short
        x2 = cx + px * tlen
        y2 = cy + py * tlen
        draw.line([(cx, cy), (x2, y2)], fill=color, width=thickness)
