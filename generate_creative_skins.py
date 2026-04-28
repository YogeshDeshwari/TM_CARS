#!/usr/bin/env python3
"""
Generate 8 creative car skin designs using procedural math/geometry.

1. Liquid Chrome    -- fake environment reflection mapping
2. Wireframe Blueprint -- edge-detected UV outlines on dark blue
3. Camouflage       -- multi-scale thresholded noise (woodland, digital, arctic)
4. Weathered Patina -- layered decay (copper + verdigris)
5. Interference Film -- thin-film optical iridescence
6. Negative Space Typography -- dense text with knockout shapes
7. Generative Architecture -- parametric bezier band system
8. Hot Rod Pinstripe -- candy paint + Lissajous ornamental curves
"""

import math
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops, ImageOps
from pro_skin_engine import ProSkinEngine

# ---------------------------------------------------------------------------
# Noise primitives (PIL-based, no external deps)
# ---------------------------------------------------------------------------

def _perlin_like(size, scale, seed=42):
    """Smooth noise field via PIL effect_noise + bicubic upscale."""
    rng = np.random.RandomState(seed)
    res = max(4, size // scale)
    sigma = rng.randint(8, 16)
    raw = Image.effect_noise((res, res), sigma)
    up = raw.resize((size, size), Image.Resampling.BICUBIC)
    return np.array(up, dtype=np.float64) / 255.0


def _multi_octave_noise(size, scales, weights, seed=42):
    """Sum multiple noise octaves."""
    result = np.zeros((size, size), dtype=np.float64)
    for i, (sc, w) in enumerate(zip(scales, weights)):
        result += _perlin_like(size, sc, seed=seed + i * 17) * w
    mn, mx = result.min(), result.max()
    if mx - mn > 1e-9:
        result = (result - mn) / (mx - mn)
    return result


def _voronoi_dist(size, n_pts, seed=42):
    """Returns distance-to-nearest-point field and cell-id field."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(n_pts, 2) * size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    min_dist = np.full((size, size), 1e9)
    cell_id = np.zeros((size, size), dtype=int)
    for i, (px, py) in enumerate(pts):
        d = np.sqrt((xx - px)**2 + (yy - py)**2)
        closer = d < min_dist
        min_dist[closer] = d[closer]
        cell_id[closer] = i
    return min_dist, cell_id


# ---------------------------------------------------------------------------
# Palette mapping
# ---------------------------------------------------------------------------

def _palette_map(field, palette, size):
    """Map a 0-1 field to an RGB array via linear palette interpolation."""
    n = len(palette)
    pal = np.array(palette, dtype=np.float64)
    t = np.clip(field, 0, 1) * (n - 1)
    idx = np.clip(t.astype(int), 0, n - 2)
    frac = t - idx
    out = np.zeros((size, size, 3), dtype=np.uint8)
    for ch in range(3):
        lo = pal[idx, ch]
        hi = pal[np.minimum(idx + 1, n - 1), ch]
        out[:, :, ch] = (lo + frac * (hi - lo)).astype(np.uint8)
    return out


def _make_rgba(rgb_arr):
    """Convert HxWx3 uint8 to RGBA PIL Image."""
    h, w = rgb_arr.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb_arr
    out[:, :, 3] = 255
    return Image.fromarray(out, "RGBA")


# ---------------------------------------------------------------------------
# 1. LIQUID CHROME
# ---------------------------------------------------------------------------

def generate_liquid_chrome(size=2048, seed=42):
    """Fake environment reflection mapping with warped sky gradient."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2, size / 2

    dx = (xx - cx) / (size / 2)
    dy = (yy - cy) / (size / 2)

    noise1 = _perlin_like(size, 6, seed) * 0.4
    noise2 = _perlin_like(size, 10, seed + 7) * 0.25
    dx += noise1 - 0.2
    dy += noise2 - 0.125

    r2 = dx * dx + dy * dy
    nz = np.sqrt(np.clip(1.0 - np.minimum(r2, 0.95), 0.05, 1.0))

    dot_nv = nz
    ry = 2 * dot_nv * dy - 0.0
    ry = np.clip(ry, -1, 1)

    t = (ry + 1) / 2.0

    sky_palette = [
        (15, 12, 10),
        (60, 45, 35),
        (220, 200, 180),
        (255, 245, 235),
        (160, 195, 230),
        (70, 100, 160),
        (25, 35, 65),
    ]
    rgb = _palette_map(t, sky_palette, size)

    fresnel = np.power(np.clip(1.0 - nz, 0, 1), 3)
    for ch in range(3):
        boosted = rgb[:, :, ch].astype(np.float64) + fresnel * 80
        rgb[:, :, ch] = np.clip(boosted, 0, 255).astype(np.uint8)

    return _make_rgba(rgb)


# ---------------------------------------------------------------------------
# 2. WIREFRAME BLUEPRINT
# ---------------------------------------------------------------------------

def generate_wireframe_blueprint(size=2048, seed=42):
    """High-contrast technical blueprint with dense wireframes and annotations."""
    from skin_utils import build_robust_island_masks
    from scipy.ndimage import binary_dilation, binary_erosion
    import json

    BG      = (6, 12, 35)
    GRID_DIM = (15, 30, 70)
    GRID_MID = (25, 55, 110)
    WIRE     = (100, 210, 255)
    WIRE_BRT = (180, 240, 255)
    GLOW     = (40, 120, 200)
    ACCENT   = (255, 255, 255)

    img = Image.new("RGBA", (size, size), BG + (255,))
    draw = ImageDraw.Draw(img)

    # --- Dense grid: fine lines every 32px, medium every 128px, major every 512px ---
    fine = size // 64
    med = size // 16
    major = size // 4
    for y in range(0, size, fine):
        w = 1
        col = GRID_DIM
        if y % major == 0:
            w, col = 2, GRID_MID
        elif y % med == 0:
            w, col = 1, GRID_MID
        draw.line([(0, y), (size, y)], fill=col + (255,), width=w)
    for x in range(0, size, fine):
        w = 1
        col = GRID_DIM
        if x % major == 0:
            w, col = 2, GRID_MID
        elif x % med == 0:
            w, col = 1, GRID_MID
        draw.line([(x, 0), (x, size)], fill=col + (255,), width=w)

    # --- Island edge wireframes: thick + glow ---
    masks = build_robust_island_masks(size)
    with open("assets/uv_atlas/standard_stadium_islands_2048.json") as f:
        atlas = json.load(f)

    edge_layer = Image.new("L", (size, size), 0)
    inner_layer = Image.new("L", (size, size), 0)

    for isl in atlas["islands"]:
        iid = isl["id"]
        if iid not in masks:
            continue
        binary = np.array(masks[iid]) > 128

        outer = binary_dilation(binary, iterations=4)
        edge_outer = outer & ~binary
        edge_arr = np.array(edge_layer)
        edge_arr[edge_outer] = 255
        edge_layer = Image.fromarray(edge_arr)

        if binary.sum() > 500:
            eroded = binary_erosion(binary, iterations=2)
            inner_edge = binary & ~eroded
            ia = np.array(inner_layer)
            ia[inner_edge] = 180
            inner_layer = Image.fromarray(ia)

    glow_layer = edge_layer.filter(ImageFilter.GaussianBlur(6))
    glow_rgba = ImageOps.colorize(glow_layer, (0, 0, 0), GLOW).convert("RGBA")
    glow_rgba.putalpha(glow_layer)
    img = Image.alpha_composite(img, glow_rgba)

    wire_rgba = ImageOps.colorize(edge_layer, (0, 0, 0), WIRE_BRT).convert("RGBA")
    wire_rgba.putalpha(edge_layer)
    img = Image.alpha_composite(img, wire_rgba)

    inner_rgba = ImageOps.colorize(inner_layer, (0, 0, 0), WIRE).convert("RGBA")
    inner_rgba.putalpha(inner_layer)
    img = Image.alpha_composite(img, inner_rgba)

    # --- Internal subdivision lines within each island ---
    draw = ImageDraw.Draw(img)
    subdiv = size // 48
    for isl in atlas["islands"]:
        iid = isl["id"]
        if iid not in masks:
            continue
        x0, y0, x1, y1 = isl["bbox"]
        bw, bh = x1 - x0, y1 - y0
        if bw < 30 or bh < 30:
            continue
        mask_arr = np.array(masks[iid])
        for gy in range(y0, y1, subdiv):
            row = mask_arr[gy, x0:x1]
            runs = np.where(row > 128)[0]
            if len(runs) > 1:
                lx, rx = x0 + runs[0], x0 + runs[-1]
                draw.line([(lx, gy), (rx, gy)], fill=GRID_MID + (120,), width=1)
        for gx in range(x0, x1, subdiv):
            col_data = mask_arr[y0:y1, gx]
            runs = np.where(col_data > 128)[0]
            if len(runs) > 1:
                ty, by_ = y0 + runs[0], y0 + runs[-1]
                draw.line([(gx, ty), (gx, by_)], fill=GRID_MID + (120,), width=1)

    # --- Annotations: center crosshairs, dimension labels, ID tags ---
    try:
        afont = ImageFont.truetype(
            str(Path.home() / "Library/Fonts/DejaVuSansMono.ttf"), max(10, size // 180)
        )
    except OSError:
        afont = ImageFont.load_default()

    for isl in atlas["islands"]:
        iid = isl["id"]
        cx, cy = isl["center"]
        x0, y0, x1, y1 = isl["bbox"]
        bw, bh = x1 - x0, y1 - y0

        cr = max(6, size // 180)
        draw.line([(cx - cr, cy), (cx + cr, cy)], fill=ACCENT + (200,), width=1)
        draw.line([(cx, cy - cr), (cx, cy + cr)], fill=ACCENT + (200,), width=1)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=WIRE_BRT + (255,))

        if bw > 60 and bh > 60:
            label = f"I{iid}"
            draw.text((cx + cr + 2, cy - 6), label, fill=WIRE + (200,), font=afont)

            dim_txt = f"{bw}x{bh}"
            draw.text((x0 + 2, y1 + 2), dim_txt, fill=GRID_MID + (180,), font=afont)

    return img


# ---------------------------------------------------------------------------
# 3. CAMOUFLAGE GENERATOR
# ---------------------------------------------------------------------------

def generate_camouflage(size=2048, variant="woodland", seed=42):
    """Multi-scale thresholded noise camo. variant: woodland/digital/arctic."""
    palettes = {
        "woodland": [(75, 83, 32), (34, 49, 28), (120, 107, 70), (15, 15, 12)],
        "digital":  [(90, 95, 60), (50, 55, 35), (140, 135, 100), (25, 30, 20)],
        "arctic":   [(235, 240, 245), (180, 190, 200), (140, 150, 165), (100, 110, 130)],
    }
    pal = palettes.get(variant, palettes["woodland"])

    n1 = _perlin_like(size, 4, seed)
    n2 = _perlin_like(size, 8, seed + 11)
    n3 = _perlin_like(size, 16, seed + 23)

    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[:, :] = pal[0]

    mask2 = n1 > 0.45
    mask3 = n2 > 0.50
    mask4 = n3 > 0.55

    for ch in range(3):
        result[:, :, ch] = np.where(mask2, pal[1][ch], result[:, :, ch])
        result[:, :, ch] = np.where(mask3, pal[2][ch], result[:, :, ch])
        result[:, :, ch] = np.where(mask4, pal[3][ch], result[:, :, ch])

    if variant == "digital":
        pix = size // 8
        small = Image.fromarray(result).resize((pix, pix), Image.Resampling.NEAREST)
        result = np.array(small.resize((size, size), Image.Resampling.NEAREST))

    return _make_rgba(result)


# ---------------------------------------------------------------------------
# 4. WEATHERED PATINA
# ---------------------------------------------------------------------------

def generate_weathered_patina(size=2048, seed=42):
    """Copper with verdigris oxidation, paint chips, edge wear.

    Decay layers:
    - Oxidation: low-freq noise -> patina blending (smoothstep threshold)
    - Paint chips: high-freq noise threshold -> exposed grey metal
    - Rust: AND of chip regions + separate noise -> orange-brown
    - Edge wear: exponential falloff from texture edges -> more patina at boundaries
    """
    copper = np.array([184, 115, 51], dtype=np.float64)
    patina = np.array([64, 145, 108], dtype=np.float64)
    rust = np.array([120, 50, 20], dtype=np.float64)
    exposed_metal = np.array([90, 85, 80], dtype=np.float64)

    base = np.tile(copper, (size, size, 1)).astype(np.float64)

    oxidation = _perlin_like(size, 5, seed)
    t_ox = np.clip((oxidation - 0.3) / 0.4, 0, 1)

    for ch in range(3):
        base[:, :, ch] = base[:, :, ch] * (1 - t_ox * 0.7) + patina[ch] * t_ox * 0.7

    chip_noise = _perlin_like(size, 14, seed + 5)
    chip_mask = chip_noise > 0.72
    for ch in range(3):
        base[:, :, ch] = np.where(chip_mask, exposed_metal[ch], base[:, :, ch])

    rust_noise = _perlin_like(size, 20, seed + 9)
    rust_mask = (rust_noise > 0.7) & chip_mask
    for ch in range(3):
        base[:, :, ch] = np.where(rust_mask, rust[ch], base[:, :, ch])

    scratch_noise = _perlin_like(size, 30, seed + 15)
    scratch_mask = scratch_noise > 0.85
    scratch_col = np.array([140, 130, 115], dtype=np.float64)
    for ch in range(3):
        base[:, :, ch] = np.where(scratch_mask, scratch_col[ch], base[:, :, ch])

    from scipy.ndimage import distance_transform_edt
    paint_region = np.ones((size, size), dtype=bool)
    border = size // 80
    paint_region[:border, :] = False
    paint_region[-border:, :] = False
    paint_region[:, :border] = False
    paint_region[:, -border:] = False
    edge_dist = distance_transform_edt(paint_region)
    edge_wear = np.exp(-edge_dist / 22.0)

    for ch in range(3):
        base[:, :, ch] = base[:, :, ch] * (1 - edge_wear * 0.5) + patina[ch] * edge_wear * 0.5

    result = np.clip(base, 0, 255).astype(np.uint8)
    return _make_rgba(result)


# ---------------------------------------------------------------------------
# 5. INTERFERENCE FILM
# ---------------------------------------------------------------------------

def generate_interference_film(size=2048, seed=42):
    """Thin-film optical iridescence -- oil-on-water rainbow effect."""
    thickness = _multi_octave_noise(size, [5, 10, 20], [0.5, 0.3, 0.2], seed)
    wetness = _perlin_like(size, 6, seed + 33)

    hue = (thickness * 3.5) % 1.0
    sat = 0.75 + wetness * 0.2
    val = 0.3 + wetness * 0.6

    h6 = hue * 6.0
    sector = h6.astype(int) % 6
    frac = h6 - h6.astype(int)

    p = val * (1 - sat)
    q = val * (1 - sat * frac)
    t = val * (1 - sat * (1 - frac))

    rgb = np.zeros((size, size, 3), dtype=np.float64)
    for s_val in range(6):
        mask = sector == s_val
        if s_val == 0:
            rgb[mask, 0] = val[mask]; rgb[mask, 1] = t[mask]; rgb[mask, 2] = p[mask]
        elif s_val == 1:
            rgb[mask, 0] = q[mask]; rgb[mask, 1] = val[mask]; rgb[mask, 2] = p[mask]
        elif s_val == 2:
            rgb[mask, 0] = p[mask]; rgb[mask, 1] = val[mask]; rgb[mask, 2] = t[mask]
        elif s_val == 3:
            rgb[mask, 0] = p[mask]; rgb[mask, 1] = q[mask]; rgb[mask, 2] = val[mask]
        elif s_val == 4:
            rgb[mask, 0] = t[mask]; rgb[mask, 1] = p[mask]; rgb[mask, 2] = val[mask]
        else:
            rgb[mask, 0] = val[mask]; rgb[mask, 1] = p[mask]; rgb[mask, 2] = q[mask]

    dark_base = np.array([18, 18, 22], dtype=np.float64)
    film_strength = np.clip(wetness * 1.3, 0.2, 1.0)[:, :, None]
    final = dark_base * (1 - film_strength) + rgb * 255 * film_strength

    return _make_rgba(np.clip(final, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 6. NEGATIVE SPACE TYPOGRAPHY
# ---------------------------------------------------------------------------

def generate_negative_space_typography(size=2048, seed=42):
    """Dense text grid with knockout shapes revealed in bright white."""
    bg = (12, 14, 18)
    dim_text = (32, 35, 42)
    bright_text = (245, 248, 255)

    img = Image.new("RGBA", (size, size), bg + (255,))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            str(Path.home() / "Library/Fonts/DejaVuSansMono-Bold.ttf"), 11
        )
    except OSError:
        font = ImageFont.load_default()

    words = "TRACKMANIA UNITED FOREVER NADEO ESWC NATIONS RACING SPEED "
    text_block = (words * 300)
    char_w, line_h = 7, 13
    ci = 0
    for y in range(0, size, line_h):
        chunk = text_block[ci:ci + size // char_w]
        draw.text((0, y), chunk, fill=dim_text + (255,), font=font)
        ci = (ci + len(chunk)) % len(text_block)

    knockout = Image.new("L", (size, size), 0)
    ko_draw = ImageDraw.Draw(knockout)

    ko_draw.rectangle([size//4, size//3, 3*size//4, 2*size//3], fill=180)

    for i in range(5):
        y = size // 6 + i * size // 6
        thickness = 30 + (i % 3) * 15
        ko_draw.rectangle([0, y - thickness, size, y + thickness], fill=140)

    try:
        tm_icon = Image.open("examples/images/tm_logo.png").convert("L")
        tm_icon = tm_icon.resize((size // 3, size // 5), Image.Resampling.LANCZOS)
        ko_draw.bitmap(
            (size // 3, size * 2 // 5),
            tm_icon.point(lambda p: 255 if p < 100 else 0),
            fill=220
        )
    except Exception:
        pass

    knockout = knockout.filter(ImageFilter.GaussianBlur(3))
    ko_arr = np.array(knockout, dtype=np.float64) / 255.0

    img_arr = np.array(img)
    for ch in range(3):
        base = img_arr[:, :, ch].astype(np.float64)
        target = np.where(ko_arr > 0.3, bright_text[ch], dim_text[ch]).astype(np.float64)
        img_arr[:, :, ch] = np.clip(
            base * (1 - ko_arr) + target * ko_arr, 0, 255
        ).astype(np.uint8)

    return Image.fromarray(img_arr, "RGBA")


# ---------------------------------------------------------------------------
# 7. GENERATIVE ARCHITECTURE
# ---------------------------------------------------------------------------

def generate_generative_architecture(size=2048, seed=42):
    """Parametric bezier band system -- flowing Zaha Hadid curves.

    Each curve: cubic Bezier B(t) = sum_i C(3,i)(1-t)^(3-i) t^i P_i
    Band intensity: Gaussian falloff exp(-d^2 / 2*sigma^2) from curve center.
    Band width breathes along curve via sinusoidal modulation.
    """
    rng = np.random.RandomState(seed)
    bg = np.array([8, 8, 12], dtype=np.float64)
    band_colors = [
        (220, 220, 230), (180, 60, 40), (40, 150, 200),
        (255, 180, 50), (120, 200, 120), (200, 80, 180),
        (240, 120, 60), (60, 80, 200),
    ]

    canvas = np.tile(bg, (size, size, 1)).astype(np.float64)

    stripe_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(stripe_img)

    n_curves = 8
    curve_data = []
    for ci in range(n_curves):
        ctrl = rng.rand(4, 2) * size
        ts = np.linspace(0, 1, 600)
        pts = np.zeros((len(ts), 2))
        for i in range(4):
            bern = math.comb(3, i) * (ts ** i) * ((1 - ts) ** (3 - i))
            pts[:, 0] += bern * ctrl[i, 0]
            pts[:, 1] += bern * ctrl[i, 1]

        band_w = rng.uniform(20, 65)
        col = np.array(band_colors[ci % len(band_colors)], dtype=np.float64)

        for w in [band_w, band_w * 0.6, band_w * 0.3]:
            coords = [(float(p[0]), float(p[1])) for p in pts]
            draw.line(coords, fill=min(255, int(220 / (1 + (band_w - w) / 30))), width=max(1, int(w)))

        curve_data.append((col, band_w))

    stripe_arr = np.array(stripe_img, dtype=np.float64) / 255.0

    glow = stripe_img.filter(ImageFilter.GaussianBlur(8))
    glow_arr = np.array(glow, dtype=np.float64) / 255.0 * 0.4
    combined = np.clip(stripe_arr + glow_arr, 0, 1)

    avg_col = np.mean([cd[0] for cd in curve_data], axis=0)
    for ch in range(3):
        canvas[:, :, ch] = canvas[:, :, ch] * (1 - combined) + avg_col[ch] * combined

    for ci, (col, bw) in enumerate(curve_data):
        layer = Image.new("L", (size, size), 0)
        ld = ImageDraw.Draw(layer)
        ctrl = rng.rand(4, 2) * size
        ts = np.linspace(0, 1, 400)
        pts = np.zeros((len(ts), 2))
        for i in range(4):
            bern = math.comb(3, i) * (ts ** i) * ((1 - ts) ** (3 - i))
            pts[:, 0] += bern * ctrl[i, 0]
            pts[:, 1] += bern * ctrl[i, 1]
        coords = [(float(p[0]), float(p[1])) for p in pts]
        ld.line(coords, fill=255, width=max(1, int(bw * 0.5)))
        la = np.array(layer.filter(ImageFilter.GaussianBlur(max(1, int(bw * 0.3)))), dtype=np.float64) / 255.0
        for ch in range(3):
            canvas[:, :, ch] = canvas[:, :, ch] * (1 - la * 0.7) + col[ch] * la * 0.7

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 8. HOT ROD PINSTRIPE
# ---------------------------------------------------------------------------

def generate_hot_rod_pinstripe(size=2048, seed=42):
    """Candy paint base + Lissajous pinstripe ornaments."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = size / 2.0, size / 2.0

    dx = (xx - cx) / (size / 2)
    dy = (yy - cy) / (size / 2)
    r2 = np.clip(dx * dx + dy * dy, 0, 2)
    nz = np.sqrt(np.clip(1.0 - r2 * 0.4, 0.1, 1.0))
    fresnel = np.power(np.clip(1.0 - nz, 0, 1), 3)

    base_dark = np.array([40, 0, 5], dtype=np.float64)
    candy = np.array([200, 15, 30], dtype=np.float64)
    canvas = np.zeros((size, size, 3), dtype=np.float64)
    t_candy = 0.25 + 0.75 * fresnel
    for ch in range(3):
        canvas[:, :, ch] = base_dark[ch] * (1 - t_candy) + candy[ch] * t_candy

    stripe_img = Image.new("L", (size, size), 0)
    stripe_draw = ImageDraw.Draw(stripe_img)

    rng = np.random.RandomState(seed)
    n_stripes = 6
    for si in range(n_stripes):
        a_freq = rng.randint(1, 5)
        b_freq = rng.randint(1, 5)
        delta = rng.uniform(0, math.pi)
        amp_x = rng.uniform(size * 0.15, size * 0.4)
        amp_y = rng.uniform(size * 0.15, size * 0.4)
        off_x = rng.uniform(size * 0.2, size * 0.8)
        off_y = rng.uniform(size * 0.2, size * 0.8)

        pts = []
        for t in np.linspace(0, 2 * math.pi, 800):
            px = off_x + amp_x * math.sin(a_freq * t + delta)
            py = off_y + amp_y * math.sin(b_freq * t)
            pts.append((px, py))
        stripe_draw.line(pts, fill=255, width=3)

        mirrored = [(px, size - py) for px, py in pts]
        stripe_draw.line(mirrored, fill=200, width=2)

    stripe_img = stripe_img.filter(ImageFilter.GaussianBlur(1.5))
    stripe_arr = np.array(stripe_img, dtype=np.float64) / 255.0

    gold = np.array([255, 210, 80], dtype=np.float64)
    for ch in range(3):
        canvas[:, :, ch] = canvas[:, :, ch] * (1 - stripe_arr) + gold[ch] * stripe_arr

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 9. ZELLIGE MOSAIC (v2 -- deeper Moroccan palette, denser tiles, sharp stars)
# ---------------------------------------------------------------------------

def generate_zellige_mosaic(size=2048, seed=42):
    """Moroccan Islamic star-and-cross tessellation with gold grout.

    v2 improvements: denser grid (size//24), saturated authentic palette,
    tighter star geometry, thicker grout, per-tile glaze variation.
    """
    rng = np.random.RandomState(seed)

    # authentic Moroccan palette -- heavy on blues, controlled accents
    palette = [
        (15, 60, 150),   # deep cobalt
        (20, 65, 155),   # cobalt variant
        (0, 115, 110),   # Fez teal
        (5, 130, 125),   # turquoise
        (235, 235, 225), # bone white
        (190, 155, 40),  # aged saffron
        (130, 40, 22),   # dark sienna
        (25, 80, 165),   # royal blue
    ]

    cell = max(16, size // 24)
    half = cell // 2
    grout_width = max(2, cell // 16)

    canvas = np.zeros((size, size, 3), dtype=np.float64)

    yy, xx = np.mgrid[0:size, 0:size]
    cx = xx % cell - half
    cy = yy % cell - half

    grid_ix = xx // cell
    grid_iy = yy // cell

    # 8-pointed star formed by intersection of axis-aligned and 45-deg-rotated
    # squares. Tighter shrink = sharper star points.
    shrink = 0.78
    r_aligned = np.maximum(np.abs(cx), np.abs(cy))
    r_rotated = np.abs(cx) + np.abs(cy)

    aligned_limit = half * shrink
    rotated_limit = half * shrink * 1.42

    in_aligned = r_aligned < (aligned_limit - grout_width)
    in_rotated = r_rotated < (rotated_limit - grout_width)

    star = in_aligned & in_rotated
    cross = in_aligned & ~in_rotated
    kite = ~in_aligned & in_rotated

    # grout = everything not in a tile shape
    in_any = star | cross | kite

    cell_hash = (grid_iy * 1031 + grid_ix * 769 + seed) % 2147483647

    # per-tile color from palette, with stars biased toward blues
    n_pal = len(palette)
    star_pal_idx = (cell_hash + 0) % 4          # first 4 entries are blues/teals
    cross_pal_idx = (cell_hash * 3 + 7) % n_pal
    kite_pal_idx = (cell_hash * 5 + 13) % n_pal

    # per-tile brightness jitter for handmade feel
    jitter = ((cell_hash % 51).astype(np.float64) - 25.0) / 25.0 * 14.0

    for ch in range(3):
        pal_vals = np.array([palette[i][ch] for i in range(n_pal)], dtype=np.float64)
        canvas[:, :, ch] = np.where(star, pal_vals[star_pal_idx] + jitter, canvas[:, :, ch])
        canvas[:, :, ch] = np.where(cross, pal_vals[cross_pal_idx] + jitter * 0.7, canvas[:, :, ch])
        canvas[:, :, ch] = np.where(kite, pal_vals[kite_pal_idx] + jitter * 0.5, canvas[:, :, ch])

    # grout: warm gold
    grout_col = np.array([175, 145, 70], dtype=np.float64)
    grout_mask = ~in_any
    for ch in range(3):
        canvas[:, :, ch] = np.where(grout_mask, grout_col[ch], canvas[:, :, ch])

    # reinforce grout at all tile-type transitions
    edge_h = (in_any[:-1, :] != in_any[1:, :])
    edge_v = (in_any[:, :-1] != in_any[:, 1:])
    type_h = (star[:-1, :].astype(np.int8) * 2 + cross[:-1, :].astype(np.int8)) != \
             (star[1:, :].astype(np.int8) * 2 + cross[1:, :].astype(np.int8))
    type_v = (star[:, :-1].astype(np.int8) * 2 + cross[:, :-1].astype(np.int8)) != \
             (star[:, 1:].astype(np.int8) * 2 + cross[:, 1:].astype(np.int8))
    edge_map = np.zeros((size, size), dtype=np.float64)
    edge_map[:-1, :] = np.where(edge_h | type_h, 1.0, edge_map[:-1, :])
    edge_map[:, :-1] = np.maximum(edge_map[:, :-1], np.where(edge_v | type_v, 1.0, 0.0))

    # thicken grout lines via MaxFilter + blur
    edge_img = Image.fromarray((edge_map * 255).astype(np.uint8), "L")
    edge_img = edge_img.filter(ImageFilter.MaxFilter(max(3, grout_width)))
    edge_img = edge_img.filter(ImageFilter.GaussianBlur(1.0))
    edge_arr = np.array(edge_img, dtype=np.float64) / 255.0

    for ch in range(3):
        canvas[:, :, ch] = canvas[:, :, ch] * (1.0 - edge_arr * 0.85) + grout_col[ch] * edge_arr * 0.85

    # per-tile glaze sheen: slight specular variation within each tile
    # (lighter at centre of tile, darker near grout)
    dist_to_centre = np.sqrt(cx.astype(np.float64)**2 + cy.astype(np.float64)**2) / (half + 1e-9)
    glaze_shade = np.clip(1.15 - dist_to_centre * 0.30, 0.85, 1.15)
    for ch in range(3):
        canvas[:, :, ch] = np.where(in_any, canvas[:, :, ch] * glaze_shade, canvas[:, :, ch])

    # surface texture: fine noise for hand-glazed irregularity
    glaze_noise = _perlin_like(size, 22, seed + 99)
    for ch in range(3):
        canvas[:, :, ch] += (glaze_noise - 0.5) * 8

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 10. STAINED GLASS (v2 -- many more cells, real distance-to-edge shading,
#     much thicker lead came, dominant blue/red palette)
# ---------------------------------------------------------------------------

def generate_stained_glass(size=2048, seed=42):
    """Gothic cathedral stained glass with lead came and jewel tones.

    v3: fewer/bigger cells (100-130), heavily blue-dominant palette with
    weighted random assignment, deeper shade range, thicker came.
    """
    rng = np.random.RandomState(seed)
    n_cells = 100 + rng.randint(0, 30)

    pts_x = rng.randint(0, size, n_cells).astype(np.float64)
    pts_y = rng.randint(0, size, n_cells).astype(np.float64)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cell_map = np.zeros((size, size), dtype=np.int32)
    min_dist = np.full((size, size), 1e18, dtype=np.float64)
    sec_dist = np.full((size, size), 1e18, dtype=np.float64)

    for i in range(n_cells):
        d = np.sqrt((xx - pts_x[i])**2 + (yy - pts_y[i])**2)
        farther = d >= min_dist
        closer = ~farther
        sec_dist = np.where(closer, np.minimum(sec_dist, min_dist), sec_dist)
        sec_dist = np.where(farther, np.minimum(sec_dist, d), sec_dist)
        cell_map[closer] = i
        min_dist[closer] = d[closer]

    edge_dist = (sec_dist - min_dist) / 2.0
    edge_dist_max = np.percentile(edge_dist, 95)
    edge_norm = np.clip(edge_dist / (edge_dist_max + 1e-9), 0, 1)

    # weighted jewel palette: 60% deep blues, 25% rubies, 15% rare accents
    # assignment via weighted random choice, not uniform
    jewels = [
        (15, 25, 120),   # deep sapphire
        (20, 35, 140),   # sapphire
        (25, 40, 155),   # bright sapphire
        (10, 20, 100),   # midnight blue
        (130, 15, 20),   # dark ruby
        (155, 20, 28),   # ruby
        (170, 25, 35),   # bright ruby
        (15, 85, 40),    # emerald (rare)
        (175, 130, 10),  # amber (rare)
        (180, 170, 150), # clear/warm white (rare)
    ]
    weights = np.array([0.15, 0.18, 0.15, 0.12, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05])
    weights /= weights.sum()
    color_assign = rng.choice(len(jewels), n_cells, p=weights)
    bright_jitter = rng.uniform(-8, 8, n_cells)

    # deeper shade range: dark at edges, luminous at centre
    shade = 0.15 + 0.75 * edge_norm

    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for i in range(n_cells):
        mask = cell_map == i
        col = jewels[color_assign[i]]
        for ch in range(3):
            canvas[:, :, ch] = np.where(
                mask,
                (col[ch] + bright_jitter[i]) * shade,
                canvas[:, :, ch],
            )

    # lead came: thicker for game readability
    edge_h = (cell_map[:-1, :] != cell_map[1:, :]).astype(np.uint8) * 255
    edge_v = (cell_map[:, :-1] != cell_map[:, 1:]).astype(np.uint8) * 255
    edge_raw = np.zeros((size, size), dtype=np.uint8)
    edge_raw[:-1, :] = np.maximum(edge_raw[:-1, :], edge_h)
    edge_raw[:, :-1] = np.maximum(edge_raw[:, :-1], edge_v)

    came_thick = max(9, size // 200) | 1  # must be odd
    came_img = Image.fromarray(edge_raw, "L")
    came_img = came_img.filter(ImageFilter.MaxFilter(came_thick))
    came_img = came_img.filter(ImageFilter.GaussianBlur(1.8))
    came_arr = np.array(came_img, dtype=np.float64) / 255.0

    lead_color = np.array([8, 8, 12], dtype=np.float64)
    for ch in range(3):
        canvas[:, :, ch] = canvas[:, :, ch] * (1.0 - came_arr) + lead_color[ch] * came_arr

    # metallic highlight on one edge of the came
    hl = np.zeros((size, size), dtype=np.float64)
    hl[1:, 1:] = np.maximum(0, came_arr[1:, 1:] - came_arr[:-1, :-1])
    for ch in range(3):
        canvas[:, :, ch] += hl * 35

    # glass texture
    tex1 = _perlin_like(size, 10, seed + 55)
    tex2 = _perlin_like(size, 25, seed + 66)
    for ch in range(3):
        canvas[:, :, ch] += ((tex1 - 0.5) * 8 + (tex2 - 0.5) * 4) * shade

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 11. CYMATICS / CHLADNI PLATE
# ---------------------------------------------------------------------------

def generate_cymatics(size=2048, seed=42):
    """Chladni plate vibration pattern -- sand settling on nodal lines of
    superimposed standing wave modes on a square plate."""
    rng = np.random.RandomState(seed)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x_norm = xx / size
    y_norm = yy / size

    # superimpose several plate modes: W = sin(m*pi*x) * sin(n*pi*y)
    modes = [
        (3, 5, 1.0),
        (5, 3, 0.9),
        (7, 4, 0.6),
        (4, 7, 0.55),
        (6, 6, 0.35),
    ]

    W = np.zeros((size, size), dtype=np.float64)
    for m, n, amp in modes:
        phase_x = rng.uniform(-0.1, 0.1)
        phase_y = rng.uniform(-0.1, 0.1)
        W += amp * np.sin((m * math.pi * x_norm) + phase_x) * \
                   np.sin((n * math.pi * y_norm) + phase_y)

    # nodal line intensity: bright where W is near zero
    W_abs = np.abs(W)
    W_max = np.percentile(W_abs, 98) + 1e-9
    W_norm = W_abs / W_max

    sigma = 0.06
    nodal = np.exp(-(W_norm ** 2) / (2 * sigma ** 2))

    # scattered sand glow around nodal lines
    nodal_img = Image.fromarray((nodal * 255).astype(np.uint8), "L")
    glow = np.array(
        nodal_img.filter(ImageFilter.GaussianBlur(size // 150)),
        dtype=np.float64
    ) / 255.0

    # dark plate base
    plate_color = np.array([18, 18, 24], dtype=np.float64)
    sand_color = np.array([220, 210, 185], dtype=np.float64)
    bright_sand = np.array([250, 245, 230], dtype=np.float64)

    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        canvas[:, :, ch] = plate_color[ch]

    # diffuse glow layer
    for ch in range(3):
        canvas[:, :, ch] += glow * sand_color[ch] * 0.25

    # sharp nodal lines
    sharp = np.clip(nodal * 1.8, 0, 1)
    for ch in range(3):
        canvas[:, :, ch] = canvas[:, :, ch] * (1.0 - sharp) + bright_sand[ch] * sharp

    # subtle colour variation along nodal lines from local curvature
    # use the sign of W for warm/cool tinting
    warm_tint = np.clip(W / (W_max + 1e-9), -1, 1) * 0.15
    canvas[:, :, 0] += warm_tint * 30 * sharp  # red push on positive side
    canvas[:, :, 2] -= warm_tint * 20 * sharp  # blue push on negative side

    # fine dust scatter
    dust = _perlin_like(size, 28, seed + 77)
    for ch in range(3):
        canvas[:, :, ch] += (dust - 0.5) * 6

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 12. TERRAZZO
# ---------------------------------------------------------------------------

def generate_terrazzo(size=2048, seed=42):
    """Venetian terrazzo -- irregular stone/marble/glass chips embedded
    in polished neutral binder."""
    rng = np.random.RandomState(seed)

    # binder base
    binder_col = np.array([192, 187, 178], dtype=np.float64)
    canvas = np.tile(binder_col, (size, size, 1)).astype(np.float64)

    # binder aggregate texture
    agg = _perlin_like(size, 18, seed + 10)
    for ch in range(3):
        canvas[:, :, ch] += (agg - 0.5) * 12

    # chip palette: marble whites, rose quartz, serpentine, charcoal, brass, sage
    chip_palette = [
        (240, 238, 232),  # marble white
        (235, 230, 225),  # cream marble
        (200, 160, 155),  # rose quartz
        (80, 120, 85),    # serpentine green
        (55, 55, 58),     # charcoal
        (70, 65, 60),     # dark stone
        (195, 175, 110),  # brass fleck
        (160, 175, 155),  # sage
        (180, 130, 115),  # terracotta chip
        (120, 135, 145),  # blue-grey
    ]

    chip_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    chip_draw = ImageDraw.Draw(chip_img)

    n_chips = 450 + rng.randint(0, 150)
    placed = []

    for _ in range(n_chips * 3):
        if len(placed) >= n_chips:
            break

        cx = rng.randint(10, size - 10)
        cy = rng.randint(10, size - 10)
        base_r = rng.uniform(size * 0.008, size * 0.035)

        # reject if overlapping an existing chip
        overlap = False
        for px, py, pr in placed:
            if (cx - px)**2 + (cy - py)**2 < (base_r + pr + 4)**2:
                overlap = True
                break
        if overlap:
            continue

        # irregular convex polygon: sample angles, vary radii
        n_verts = rng.randint(4, 9)
        angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
        radii = base_r * (0.6 + rng.rand(n_verts) * 0.8)
        rot = rng.uniform(0, 2 * math.pi)

        pts = []
        for a, r in zip(angles, radii):
            px_v = cx + r * math.cos(a + rot)
            py_v = cy + r * math.sin(a + rot)
            pts.append((px_v, py_v))

        col_idx = rng.randint(0, len(chip_palette))
        col = chip_palette[col_idx]
        jit = rng.randint(-12, 12)
        fill = tuple(max(0, min(255, c + jit)) for c in col)

        chip_draw.polygon(pts, fill=fill + (255,))
        placed.append((cx, cy, base_r))

    # rasterize chips onto canvas
    chip_arr = np.array(chip_img)
    chip_mask = chip_arr[:, :, 3] > 128
    for ch in range(3):
        canvas[:, :, ch] = np.where(chip_mask, chip_arr[:, :, ch].astype(np.float64), canvas[:, :, ch])

    # polished bevel: darken chip edges slightly
    edge_detect = Image.fromarray(chip_arr[:, :, 3], "L")
    inner = np.array(edge_detect.filter(ImageFilter.MinFilter(5)), dtype=np.float64) / 255.0
    outer = np.array(edge_detect, dtype=np.float64) / 255.0
    bevel = np.clip(outer - inner, 0, 1)
    bevel_blur = np.array(
        Image.fromarray((bevel * 255).astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(1.5)
        ), dtype=np.float64
    ) / 255.0
    for ch in range(3):
        canvas[:, :, ch] -= bevel_blur * 25

    # tiny mica/quartz specks scattered everywhere
    speck_noise = _perlin_like(size, 40, seed + 88)
    specks = (speck_noise > 0.92).astype(np.float64) * 80
    for ch in range(3):
        canvas[:, :, ch] += specks

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 13. OP ART WARP (Bridget Riley)
# ---------------------------------------------------------------------------

def generate_op_art_warp(size=2048, seed=42):
    """Bridget Riley-style warped checkerboard that creates an optical
    illusion of depth and movement."""
    rng = np.random.RandomState(seed)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x_n = xx / size
    y_n = yy / size

    # displacement field: 2 radial bulges
    n_bulges = 2 + rng.randint(0, 2)
    dx = np.zeros((size, size), dtype=np.float64)
    dy = np.zeros((size, size), dtype=np.float64)

    for _ in range(n_bulges):
        bx = 0.2 + rng.rand() * 0.6
        by = 0.2 + rng.rand() * 0.6
        amp = 0.06 + rng.rand() * 0.08
        sig = 0.12 + rng.rand() * 0.10

        rx = x_n - bx
        ry = y_n - by
        r2 = rx * rx + ry * ry
        gauss = np.exp(-r2 / (2 * sig * sig))
        dx += amp * rx * gauss / (sig + 1e-9)
        dy += amp * ry * gauss / (sig + 1e-9)

    # warped sampling coordinates
    wx = x_n + dx
    wy = y_n + dy

    # checkerboard with configurable cell count
    cells = 28
    gx = np.floor(wx * cells).astype(np.int32)
    gy = np.floor(wy * cells).astype(np.int32)
    checker = ((gx + gy) % 2).astype(np.float64)

    # anti-aliasing: smooth the checker boundary using fractional distance
    fx = (wx * cells) - np.floor(wx * cells)
    fy = (wy * cells) - np.floor(wy * cells)
    edge_x = np.minimum(fx, 1.0 - fx)
    edge_y = np.minimum(fy, 1.0 - fy)
    edge_dist = np.minimum(edge_x, edge_y)

    # compute local warp magnitude for adaptive AA width
    warp_mag = np.sqrt(dx * dx + dy * dy)
    aa_width = 0.02 + warp_mag * 2.0
    aa = np.clip(edge_dist / aa_width, 0, 1)
    checker_smooth = checker * aa + (1.0 - checker) * (1.0 - aa)
    checker_smooth = np.clip(checker_smooth, 0, 1)

    # palette: deep navy and warm cream (softer than pure B&W for car readability)
    dark = np.array([12, 12, 18], dtype=np.float64)
    light = np.array([235, 232, 222], dtype=np.float64)

    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        canvas[:, :, ch] = dark[ch] + checker_smooth * (light[ch] - dark[ch])

    # subtle depth shading: darken areas of strongest warp (appears recessed)
    depth_shade = 1.0 - warp_mag / (warp_mag.max() + 1e-9) * 0.15
    for ch in range(3):
        canvas[:, :, ch] *= depth_shade

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ===========================================================================
# GOLD & BLACK COLLECTION v2 -- luxury metallic skins
#
# Gold coloring: amber shadows -> bronze midtone -> rich gold -> warm white
# highlight.  NOT yellow.  Real gold shadow = brown, highlight = desaturated.
# ===========================================================================

_GOLD_STOPS = np.array([
    [55, 38, 10],      # 0.00 -- deep amber shadow
    [90, 65, 16],      # 0.15 -- dark brown
    [135, 100, 25],    # 0.35 -- bronze
    [180, 142, 38],    # 0.55 -- rich gold
    [215, 178, 58],    # 0.75 -- bright gold
    [245, 225, 140],   # 0.92 -- warm highlight
    [255, 248, 200],   # 1.00 -- specular peak
], dtype=np.float64)
_GOLD_POS = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 0.92, 1.0])


def _gold_rgb(t):
    """Map t in [0..1] to metallic gold (R, G, B) float arrays."""
    t = np.clip(t, 0, 1)
    r = np.interp(t, _GOLD_POS, _GOLD_STOPS[:, 0])
    g = np.interp(t, _GOLD_POS, _GOLD_STOPS[:, 1])
    b = np.interp(t, _GOLD_POS, _GOLD_STOPS[:, 2])
    return r, g, b


def _brushed_metal(size, angle_deg, fineness, seed):
    """Anisotropic brush texture -- directional scratches.  Returns 0..1."""
    rad = np.radians(angle_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    u = xx * np.cos(rad) + yy * np.sin(rad)
    scratch = np.sin(u / max(1, fineness)) * 0.5 + 0.5
    jitter = _perlin_like(size, 3, seed)
    return np.clip(scratch * 0.4 + jitter * 0.6, 0, 1)


# -- 1 -- MOLTEN GOLD -------------------------------------------------------
# Liquid gold poured over the car.  Thick pools at top, organic flow boundary
# in the middle, drips and rivulets streaming down.  Matte black beneath.

def generate_molten_gold(size=2048, seed=42):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    yn = yy / size

    flow_a = _multi_octave_noise(size, [90, 45, 22], [0.45, 0.35, 0.2], seed)
    flow_b = _multi_octave_noise(size, [60, 30, 15], [0.5, 0.3, 0.2], seed + 40)

    # Warp vertically: gold flows downward, stronger warp near boundary
    warp = (flow_a - 0.45) * size * 0.30 * yn
    wy = np.clip(yy + warp, 0, size - 1).astype(int)
    wxn = np.arange(size, dtype=int)
    flow_warped = flow_b[wy, wxn[np.newaxis, :]]

    # Gold coverage: dense at top, organic fall-off
    y_weight = (1.0 - yn) ** 0.6
    gold_field = y_weight * 0.55 + flow_a * 0.28 + flow_warped * 0.17

    threshold = 0.42
    gold_mask = np.clip((gold_field - threshold) / 0.035, 0, 1)

    # Thin drip tendrils below the main body
    drip_noise = _perlin_like(size, 5, seed + 70)
    drip = np.sin(xx / size * 28 * np.pi * 2 + drip_noise * 9)
    drip_thin = np.clip(drip * 5 - 3.5, 0, 1)
    near_edge = ((gold_field < threshold + 0.08) &
                 (gold_field > threshold - 0.18)).astype(np.float64)
    gold_mask = np.maximum(gold_mask, drip_thin * near_edge * 0.85)

    # Scattered droplets via high-frequency noise
    drop_n = _perlin_like(size, 2, seed + 80)
    drops = ((drop_n > 0.82) & (yn > 0.55)).astype(np.float64)
    gold_mask = np.maximum(gold_mask, drops * 0.75)

    # Soften boundary slightly
    gm_img = Image.fromarray(np.clip(gold_mask * 255, 0, 255).astype(np.uint8))
    gm_img = gm_img.filter(ImageFilter.GaussianBlur(1.8))
    gold_mask = np.array(gm_img, dtype=np.float64) / 255.0

    # Gold brightness: near-edge = bright meniscus, thick areas = deeper
    edge_prox = np.exp(-((gold_field - threshold) / 0.08) ** 2) * 0.15
    base_t = flow_a * 0.4 + flow_b * 0.3 + 0.25 + edge_prox
    micro = _perlin_like(size, 4, seed + 90)
    gold_t = np.clip(base_t + (micro - 0.5) * 0.1, 0.08, 0.95)
    gr, gg, gb = _gold_rgb(gold_t)

    # Black base with subtle warm grain
    bg = 10 + _perlin_like(size, 6, seed + 50) * 6
    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch, gc in enumerate([gr, gg, gb]):
        canvas[:, :, ch] = gold_mask * gc + (1 - gold_mask) * bg

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# -- 2 -- GOLD PLATE ARMOR --------------------------------------------------
# Large angular plates of brushed gold floating on black carbon fiber.
# Each plate has unique brush angle + beveled edges.  Gaps glow faintly.

def generate_gold_plate_armor(size=2048, seed=42):
    rng = np.random.RandomState(seed)
    n_plates = 22 + rng.randint(0, 8)

    # Place Voronoi sites with slight clustering for interesting shapes
    pts = rng.rand(n_plates, 2) * size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cell_id = np.zeros((size, size), dtype=np.int32)
    min_dist = np.full((size, size), 1e18)
    sec_dist = np.full((size, size), 1e18)

    for i in range(n_plates):
        d = np.sqrt((xx - pts[i, 0])**2 + (yy - pts[i, 1])**2)
        further = d >= min_dist
        sec_dist = np.where(further, np.minimum(sec_dist, d), sec_dist)
        update2 = (~further)
        sec_dist = np.where(update2, min_dist, sec_dist)
        cell_id = np.where(update2, i, cell_id)
        min_dist = np.minimum(min_dist, d)

    # Gap width between plates (distance to edge = sec_dist - min_dist)
    edge_dist = sec_dist - min_dist
    gap_w = max(6, size // 200)
    in_gap = edge_dist < gap_w
    gap_blend = np.clip(edge_dist / gap_w, 0, 1)

    # Bevel: bright gold at plate edges
    bevel_zone = max(12, size // 100)
    bevel = np.clip(1.0 - (edge_dist - gap_w) / bevel_zone, 0, 1)
    bevel = bevel * gap_blend  # only inside plates

    # Per-plate brushed texture with unique angle
    plate_angles = rng.uniform(0, 180, n_plates)
    plate_brightness_offset = rng.uniform(-0.06, 0.06, n_plates)
    brush_layer = np.zeros((size, size), dtype=np.float64)
    bright_offset = np.zeros((size, size), dtype=np.float64)
    for i in range(n_plates):
        pmask = cell_id == i
        bm = _brushed_metal(size, plate_angles[i], max(2, size // 600), seed + i * 7)
        brush_layer[pmask] = bm[pmask]
        bright_offset[pmask] = plate_brightness_offset[i]

    # Gold coloring with brush + bevel + per-plate variation
    gold_t = np.clip(brush_layer * 0.35 + 0.4 + bevel * 0.2 + bright_offset, 0.1, 0.92)
    gr, gg, gb = _gold_rgb(gold_t)

    # Black carbon fiber in gaps: fine diagonal crosshatch
    cf_a = np.sin((xx + yy) / 6.0) * 0.5 + 0.5
    cf_b = np.sin((xx - yy) / 6.0) * 0.5 + 0.5
    carbon = 10 + (cf_a * cf_b) * 14

    # Gap glow: faint gold light bleeding from beneath plates
    glow_src = Image.fromarray(
        np.clip((1 - gap_blend) * 200, 0, 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(size // 100))
    glow = np.array(glow_src, dtype=np.float64) / 255.0

    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch, gc in enumerate([gr, gg, gb]):
        plate_px = gc * gap_blend + carbon * (1 - gap_blend)
        canvas[:, :, ch] = plate_px + glow * _GOLD_STOPS[3, ch] * 0.25

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# -- 3 -- DRAGON SCALE ------------------------------------------------------
# Hexagonal scale pattern.  Each scale is 3D-convex (specular highlight top-
# left, shadow bottom-right).  Color gradient from brilliant gold to bronze.

def generate_dragon_scale(size=2048, seed=42):
    rng = np.random.RandomState(seed)
    cell_r = max(20, size // 40)
    row_h = cell_r * np.sqrt(3)
    col_w = cell_r * 2

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # Hex grid: for each pixel find nearest hex center (check 4 candidates)
    row = np.floor(yy / row_h).astype(int)
    x_off = (row % 2).astype(np.float64) * cell_r
    col = np.floor((xx - x_off) / col_w).astype(int)

    best_d2 = np.full((size, size), 1e18)
    best_cx = np.zeros((size, size), dtype=np.float64)
    best_cy = np.zeros((size, size), dtype=np.float64)
    best_id = np.zeros((size, size), dtype=np.int32)

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r2 = row + dr
            c2 = col + dc
            xo2 = (r2 % 2).astype(np.float64) * cell_r
            ccx = c2 * col_w + xo2 + cell_r
            ccy = r2 * row_h + row_h * 0.5
            d2 = (xx - ccx)**2 + (yy - ccy)**2
            closer = d2 < best_d2
            best_d2 = np.where(closer, d2, best_d2)
            best_cx = np.where(closer, ccx, best_cx)
            best_cy = np.where(closer, ccy, best_cy)
            best_id = np.where(closer, (r2 * 1000 + c2) % 100000, best_id)

    dist = np.sqrt(best_d2)
    dx_from_center = xx - best_cx
    dy_from_center = yy - best_cy

    # Scale boundary: gap between hexes
    gap_thresh = cell_r * 0.88
    in_scale = (dist < gap_thresh).astype(np.float64)
    scale_t = np.clip(dist / gap_thresh, 0, 1)

    # 3D curvature: cosine dome
    curvature = np.cos(scale_t * np.pi * 0.5)
    curvature = np.clip(curvature, 0, 1) * in_scale

    # Directional light from upper-left
    light_dir = (-dx_from_center * 0.7 - dy_from_center * 0.7)
    light_dir = light_dir / (cell_r + 1)
    light = np.clip(0.55 + light_dir * 0.35, 0.15, 0.95)

    # Global gradient: brilliant gold at center, darker bronze at edges
    center_dist = np.sqrt((xx - size * 0.5)**2 + (yy - size * 0.5)**2)
    global_grad = 1.0 - np.clip(center_dist / (size * 0.7), 0, 0.6)

    # Per-scale jitter for organic variation
    n_unique = int(best_id.max()) + 1
    scale_jitter = rng.uniform(-0.04, 0.04, max(n_unique, 1))
    jitter_arr = scale_jitter[np.clip(best_id, 0, len(scale_jitter) - 1)]

    gold_t = np.clip(curvature * light * 0.55 + global_grad * 0.35
                     + jitter_arr + 0.05, 0.05, 0.95)
    gr, gg, gb = _gold_rgb(gold_t)

    # Dark gaps
    gap_color = 8
    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch, gc in enumerate([gr, gg, gb]):
        canvas[:, :, ch] = in_scale * gc + (1 - in_scale) * gap_color

    # Rim highlight at scale edges (bright ring just inside boundary)
    rim_band = ((scale_t > 0.75) & (scale_t < 0.88)).astype(np.float64)
    rim_bright = rim_band * in_scale * 0.08
    for ch in range(3):
        canvas[:, :, ch] += rim_bright * _GOLD_STOPS[5, ch]

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# -- 4 -- BLACK MARBLE GOLD -------------------------------------------------
# Deep obsidian marble with flowing gold veins of varying thickness.
# Veins are generated via sin(turbulent_noise * freq) which naturally
# creates organic branching/flowing patterns like real marble.

def generate_black_marble_gold(size=2048, seed=42):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # Turbulent base fields for vein structure
    turb_a = _multi_octave_noise(size, [100, 50, 25, 12], [0.35, 0.3, 0.2, 0.15],
                                 seed)
    turb_b = _multi_octave_noise(size, [80, 40, 20], [0.45, 0.35, 0.2], seed + 60)

    # Warp for more organic flow
    warp_n = _multi_octave_noise(size, [70, 35], [0.6, 0.4], seed + 30)
    warp_str = size * 0.12
    wx = np.clip(xx + (warp_n - 0.5) * warp_str, 0, size - 1).astype(int)
    wy = np.clip(yy + (turb_b - 0.5) * warp_str * 0.7, 0, size - 1).astype(int)
    warped = turb_a[wy, wx]

    # Primary veins: sin(noise * freq) -> ridges at zero-crossings
    vein_raw = np.sin(warped * np.pi * 7)
    veins_thick = 1.0 - np.abs(vein_raw)
    veins_thick = np.clip(veins_thick * 2.5 - 0.8, 0, 1)

    # Secondary fine veins from different noise
    fine_turb = _multi_octave_noise(size, [40, 20, 10], [0.4, 0.35, 0.25],
                                    seed + 120)
    fine_raw = np.sin(fine_turb * np.pi * 14)
    veins_fine = 1.0 - np.abs(fine_raw)
    veins_fine = np.clip(veins_fine * 3 - 1.5, 0, 1) * 0.5

    veins = np.clip(veins_thick + veins_fine, 0, 1)

    # Glow around veins
    vein_img = Image.fromarray(np.clip(veins * 255, 0, 255).astype(np.uint8))
    glow_img = vein_img.filter(ImageFilter.GaussianBlur(size // 80))
    glow = np.array(glow_img, dtype=np.float64) / 255.0

    # Gold color with variation along the vein
    vein_brightness = veins * 0.5 + warped * 0.35 + 0.12
    vein_brightness = np.clip(vein_brightness, 0.08, 0.95)
    gr, gg, gb = _gold_rgb(vein_brightness)

    # Obsidian base: near-black with very subtle depth
    obs_depth = _perlin_like(size, 30, seed + 200) * 6
    canvas = np.zeros((size, size, 3), dtype=np.float64)
    canvas[:, :, 0] = 10 + obs_depth
    canvas[:, :, 1] = 8 + obs_depth * 0.7
    canvas[:, :, 2] = 6 + obs_depth * 0.4

    # Composite gold veins and glow
    for ch, gc in enumerate([gr, gg, gb]):
        canvas[:, :, ch] = canvas[:, :, ch] * (1 - veins) + gc * veins
        canvas[:, :, ch] += glow * _GOLD_STOPS[3, ch] * 0.12

    # Subtle marble clouding in the black areas
    cloud = _perlin_like(size, 50, seed + 150)
    cloud_mask = (1 - veins) * (1 - glow * 2)
    cloud_mask = np.clip(cloud_mask, 0, 1)
    for ch in range(3):
        canvas[:, :, ch] += cloud * 8 * cloud_mask

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# -- 5 -- GOLD TEMPEST ------------------------------------------------------
# Dynamic speed streaks + particle bursts flowing across the car.  Long gold
# streaks made from heavily stretched noise, interspersed with bright
# particle clusters.  Creates aggressive forward-motion energy.

def generate_gold_tempest(size=2048, seed=42):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    # Anisotropic noise: generate at low resolution then stretch horizontally
    streak_h = max(size // 8, 64)
    streak_w = size
    raw_noise = Image.effect_noise((streak_h, streak_w), rng.randint(8, 16))
    stretched = raw_noise.resize((size, size), Image.Resampling.BICUBIC)
    streak_base = np.array(stretched, dtype=np.float64) / 255.0

    # Second layer for variety
    raw2 = Image.effect_noise((streak_h * 2, streak_w), rng.randint(8, 16))
    stretched2 = raw2.resize((size, size), Image.Resampling.BICUBIC)
    streak2 = np.array(stretched2, dtype=np.float64) / 255.0

    # Warp streaks for organic flow
    warp_n = _perlin_like(size, 40, seed + 10)
    warp_v = (warp_n - 0.5) * size * 0.08
    wy = np.clip(yy + warp_v, 0, size - 1).astype(int)
    wxn = np.arange(size, dtype=int)
    streak_warped = streak_base[wy, wxn[np.newaxis, :]]

    # Combine and threshold for crisp streaks
    combined = streak_warped * 0.6 + streak2 * 0.4
    long_streaks = np.clip((combined - 0.52) / 0.06, 0, 1)

    # Speed fade: stronger from left (front of car) to right
    speed_fade = np.clip(xx / size * 1.3 + 0.1, 0, 1)
    speed_fade = speed_fade ** 0.7
    long_streaks *= speed_fade

    # Particle bursts: clusters of bright points
    burst_noise = _perlin_like(size, 8, seed + 50)
    spark_noise = _perlin_like(size, 2, seed + 60)
    bursts = ((burst_noise > 0.6) & (spark_noise > 0.75)).astype(np.float64)
    # Enlarge particles slightly
    burst_img = Image.fromarray(np.clip(bursts * 255, 0, 255).astype(np.uint8))
    burst_img = burst_img.filter(ImageFilter.MaxFilter(3))
    burst_img = burst_img.filter(ImageFilter.GaussianBlur(1.2))
    bursts = np.array(burst_img, dtype=np.float64) / 255.0
    bursts *= speed_fade * 0.8

    # Combine streaks and bursts
    gold_mask = np.clip(long_streaks + bursts, 0, 1)

    # Gold coloring: streaks brighter at leading edge, particles are hot
    streak_t = np.clip(combined * 0.5 + speed_fade * 0.3 + 0.15, 0.1, 0.9)
    burst_t = np.clip(bursts * 1.5 + 0.4, 0.3, 0.98)
    gold_t = np.where(bursts > long_streaks, burst_t, streak_t)

    # Brush texture on streaks
    brush = _brushed_metal(size, 0, max(2, size // 800), seed + 70)
    gold_t = np.clip(gold_t + (brush - 0.5) * 0.08 * gold_mask, 0.05, 0.98)

    gr, gg, gb = _gold_rgb(gold_t)

    # Background: very dark with subtle motion blur texture
    bg_streak = _perlin_like(size, 10, seed + 100)
    bg = 8 + bg_streak * 4
    canvas = np.zeros((size, size, 3), dtype=np.float64)
    for ch, gc in enumerate([gr, gg, gb]):
        canvas[:, :, ch] = gold_mask * gc + (1 - gold_mask) * bg

    # Subtle warm glow behind dense streak areas
    glow_src = Image.fromarray(
        np.clip(gold_mask * 200, 0, 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(size // 60))
    glow = np.array(glow_src, dtype=np.float64) / 255.0
    for ch in range(3):
        canvas[:, :, ch] += glow * _GOLD_STOPS[2, ch] * 0.08 * (1 - gold_mask)

    return _make_rgba(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Skin builder: wrap a flat pattern into a full car skin
# ---------------------------------------------------------------------------

def build_skin(name, pattern_func, tire_config=None, **kwargs):
    """Generate a flat pattern and composite it onto the car."""
    import sys
    sys.stdout.write(f"\n{'='*60}\n  Generating: {name}\n{'='*60}\n")
    sys.stdout.flush()

    engine = ProSkinEngine(team_name=name, full_skin=True)

    sys.stdout.write("  Generating pattern...\n"); sys.stdout.flush()
    pattern = pattern_func(size=engine.size, **kwargs)
    engine.diffuse = Image.composite(pattern, engine.diffuse, engine.paint_mask)

    if tire_config is not None:
        from tire_customizer import customize_details
        _orig_build_details = engine._build_details_texture

        def _custom_build_details(base_sizes):
            result = _orig_build_details(base_sizes)
            return customize_details(result, **tire_config)

        engine._build_details_texture = _custom_build_details

    sys.stdout.write("  Saving...\n"); sys.stdout.flush()
    engine.save()

    sys.stdout.write(f"  Done -> out/{name}.zip\n"); sys.stdout.flush()
    return Path(f"out/{name}.zip")


# ---------------------------------------------------------------------------
# Livery builder: per-island panel painting (for deliberate racing liveries)
# ---------------------------------------------------------------------------

def _load_livery_font(size_px):
    """Load a bold font for livery text."""
    for p in [
        Path.home() / "Library/Fonts/DejaVuSansCondensed-Bold.ttf",
        Path.home() / "Library/Fonts/DejaVuSans-Bold.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]:
        if p.exists():
            return ImageFont.truetype(str(p), size_px)
    return ImageFont.load_default()


def _render_island_text(engine, text_labels):
    """Render text labels onto specific islands of the diffuse texture.

    text_labels : list of dict
        Each: {"text": str, "island": int, "color": RGB,
               "size": float (0..1 fraction of island width),
               "offset_y": float (0..1, vertical position within island bbox)}
    """
    sz = engine.size
    for lbl in text_labels:
        iid = lbl["island"]
        mask = engine._island_masks.get(iid)
        if mask is None:
            continue
        island_info = engine._geo.islands.get(iid)
        if not island_info:
            continue
        x0, y0, x1, y1 = island_info.bbox
        iw, ih = x1 - x0, y1 - y0

        font_h = int(ih * lbl.get("size", 0.35))
        font = _load_livery_font(max(12, font_h))
        color = lbl.get("color", (255, 255, 255))
        if len(color) == 3:
            color = color + (255,)

        txt_layer = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        text = lbl["text"]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        off_y = lbl.get("offset_y", 0.5)
        tx = x0 + (iw - tw) // 2
        ty = y0 + int((ih - th) * off_y)
        draw.text((tx, ty), text, fill=color, font=font)

        engine.diffuse = Image.composite(txt_layer, engine.diffuse, mask)


def build_livery_skin(name, island_colors, stripe_islands=None,
                      stripe_line_w=10, stripe_gap_w=5,
                      stripe_color=None, stripe_angle=0,
                      text_labels=None, finish_overrides=None,
                      tire_config=None):
    """Build a skin by painting individual UV islands with specific colors,
    and optionally overlaying a stripe pattern on designated islands.

    Parameters
    ----------
    island_colors : dict
        Maps island_id (int) -> RGB/RGBA tuple (base fill).
    stripe_islands : list of int, optional
        Island IDs that receive the gold stripe pattern overlay.
    stripe_line_w, stripe_gap_w : int
        Width of each stripe and the gap between them (pixels at 2048).
    stripe_color : tuple, optional
        RGB/RGBA for the stripes (defaults to _GOLD).
    stripe_angle : float
        Rotation angle in degrees for the stripes (0 = horizontal).
    text_labels : list of dict, optional
        Text rendered onto specific islands.
    finish_overrides : dict, optional
        Maps island_id -> finish alpha.
    tire_config : dict, optional
        Passed to tire_customizer.customize_details().
    """
    import sys
    sys.stdout.write(f"\n{'='*60}\n  Generating livery: {name}\n{'='*60}\n")
    sys.stdout.flush()

    engine = ProSkinEngine(team_name=name, full_skin=True)
    engine.load_uv_geometry()
    sz = engine.size

    # Paint each island with its assigned color
    sys.stdout.write("  Painting panels...\n"); sys.stdout.flush()
    for island_id, color in island_colors.items():
        mask = engine._island_masks.get(island_id)
        if mask is None:
            continue
        if len(color) == 3:
            color = color + (255,)
        fill = Image.new("RGBA", (sz, sz), color)
        engine.diffuse = Image.composite(fill, engine.diffuse, mask)

    # Overlay contour-following stripes on designated islands.
    # Uses distance transform so lines follow each panel's shape
    # (V-shapes on nose, parallel on body, curved on sidepods).
    if stripe_islands:
        from scipy.ndimage import distance_transform_edt
        sc = stripe_color or _GOLD
        if len(sc) == 3:
            sc = sc + (255,)
        period = stripe_line_w + stripe_gap_w
        sys.stdout.write(
            f"  Contouring {len(stripe_islands)} islands "
            f"(line={stripe_line_w}px, gap={stripe_gap_w}px)...\n"
        )
        sys.stdout.flush()
        for iid in stripe_islands:
            mask = engine._island_masks.get(iid)
            if mask is None:
                continue
            mask_arr = np.array(mask, dtype=np.float64) / 255.0
            dist = distance_transform_edt(mask_arr > 0.5)
            phase = dist % period
            gold_band = (phase < stripe_line_w).astype(np.uint8) * 255
            # Also exclude the outermost pixel ring (clean edge)
            gold_band[mask_arr < 0.5] = 0
            band_mask = Image.fromarray(gold_band, "L")
            gold_fill = Image.new("RGBA", (sz, sz), sc)
            engine.diffuse = Image.composite(
                gold_fill, engine.diffuse, band_mask)

    # Text labels on specific islands
    if text_labels:
        sys.stdout.write(f"  Rendering {len(text_labels)} text labels...\n")
        sys.stdout.flush()
        _render_island_text(engine, text_labels)

    # Override finish alphas (controls in-game specular/gloss)
    if finish_overrides:
        for island_id, alpha_val in finish_overrides.items():
            engine._island_finish_alphas[island_id] = alpha_val

    # Tire customization
    if tire_config is not None:
        from tire_customizer import customize_details
        _orig_build_details = engine._build_details_texture

        def _custom_build_details(base_sizes):
            result = _orig_build_details(base_sizes)
            return customize_details(result, **tire_config)

        engine._build_details_texture = _custom_build_details

    sys.stdout.write("  Saving...\n"); sys.stdout.flush()
    engine.save()

    sys.stdout.write(f"  Done -> out/{name}.zip\n"); sys.stdout.flush()
    return Path(f"out/{name}.zip")


# ===========================================================================
# LIVERY DEFINITIONS
# ===========================================================================

# Rich metallic gold -- NOT yellow.  Shadows are brown, highlights are warm.
_GOLD = (205, 150, 25)
_GOLD_LIGHT = (225, 175, 42)
_BLACK = (10, 10, 10)
_NEAR_BLACK = (20, 18, 16)

_FINISH_GLOSS = 0x68
_FINISH_MATTE = 0xA0
_FINISH_CARBON = 0xB0


def _make_stripe_pattern(size, line_w, gap_w, color, angle_deg=0):
    """Create a repeating stripe pattern (gold lines on transparent).

    Returns an RGBA image: gold stripes with full alpha, transparent gaps.
    """
    period = line_w + gap_w
    if len(color) == 3:
        color = color + (255,)

    if abs(angle_deg) < 1:
        # Horizontal stripes (fast path)
        pat = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pat)
        y = 0
        while y < size:
            draw.rectangle([0, y, size, y + line_w - 1], fill=color)
            y += period
        return pat

    # Angled stripes: draw on an oversized canvas, rotate, crop
    diag = int(size * 1.5)
    pat = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pat)
    y = 0
    while y < diag:
        draw.rectangle([0, y, diag, y + line_w - 1], fill=color)
        y += period
    pat = pat.rotate(angle_deg, resample=Image.Resampling.BILINEAR,
                     expand=False, center=(diag // 2, diag // 2))
    ox = (diag - size) // 2
    oy = (diag - size) // 2
    return pat.crop((ox, oy, ox + size, oy + size))


def _jps_gold_black_livery():
    """JPS-style gold & black livery.

    The GOLD panels are NOT solid -- they have dense parallel gold stripes
    on a black base, creating a textured/lined metallic appearance.
    """
    # ALL islands start black
    colors = {}
    for iid in range(1, 28):
        colors[iid] = _BLACK
    colors[4] = _NEAR_BLACK
    for iid in [19, 20, 24, 25]:
        colors[iid] = _NEAR_BLACK

    # Islands that get gold contour stripe treatment
    gold_stripe_islands = [
        # HERO
        1,          # MAIN_BODY_SIDE
        2,          # TOP_BODY
        3,          # NOSE
        5, 6,       # SIDEPOD L/R
        # SECONDARY
        7,          # REAR_SECTION
        12, 13,     # FENDER L/R
        # ACCENT
        8,          # FRONT_WING
        9, 10,      # SIDE_SKIRT L/R
        15, 16,     # REAR_WING_ENDPLATE L/R
        21, 22, 23, # NOSE_DETAIL
    ]

    # Stripe parameters: line width and gap at 2048
    stripe_line_w = 14
    stripe_gap_w = 3

    # Text labels
    text_labels = [
        {"text": "AURA", "island": 7, "color": _GOLD_LIGHT,
         "size": 0.38, "offset_y": 0.35},
        {"text": "AURA", "island": 1, "color": _GOLD_LIGHT,
         "size": 0.06, "offset_y": 0.48},
    ]

    # Finish
    finish = {}
    for iid in range(1, 28):
        if iid in gold_stripe_islands:
            finish[iid] = _FINISH_GLOSS
        else:
            finish[iid] = _FINISH_MATTE

    return colors, gold_stripe_islands, stripe_line_w, stripe_gap_w, \
        text_labels, finish


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    colors, gold_islands, line_w, gap_w, text_labels, finish = \
        _jps_gold_black_livery()
    results = []
    try:
        zp = build_livery_skin(
            "black_gold_aura",
            island_colors=colors,
            stripe_islands=gold_islands,
            stripe_line_w=line_w,
            stripe_gap_w=gap_w,
            stripe_color=_GOLD,
            stripe_angle=0,
            text_labels=text_labels,
            finish_overrides=finish,
        )
        results.append(zp)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Generated {len(results)} skins in {elapsed:.1f}s:")
    for r in results:
        print(f"  {r}")

