import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops
from typing import Tuple, Optional, List, Dict

# =============================================================================
# COLOR & MATERIAL UTILITIES
# =============================================================================

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    h = hex_code.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

def create_gradient(size: int, color1: Tuple[int,int,int], color2: Tuple[int,int,int], direction: str = "vertical") -> Image.Image:
    """Creates a smooth linear gradient."""
    base = Image.new("RGBA", (size, size), color1 + (255,))
    top = Image.new("RGBA", (size, size), color2 + (0,))
    
    mask = Image.new("L", (size, size))
    m_data = []
    if direction == "vertical":
        for y in range(size):
            row_val = int(255 * (y / size))
            m_data.extend([row_val] * size)
    else: # horizontal
        row = [int(255 * (x / size)) for x in range(size)]
        m_data = row * size
        
    mask.putdata(m_data)
    top.putalpha(mask)
    return Image.alpha_composite(base, top)

def apply_material_finish(diffuse: Image.Image, finish_type: str) -> Image.Image:
    """
    Generates a Details.dds (Specular/Gloss map) based on the Diffuse and finish type.
    
    TrackMania Details.dds channels:
    - RGB: Lightmap/Self-Illum (usually ignored or black for standard paint)
    - Alpha: Specular Power (Glossiness). 0 = Matte, 255 = Chrome/Mirror.
    """
    w, h = diffuse.size
    # Default: Create a black RGB layer (no self-illum)
    details = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    
    # Extract brightness from diffuse to modulate gloss (darker areas usually less glossy)
    luma = diffuse.convert("L")
    
    if finish_type == "matte":
        # Matte: Low alpha (10-30), slight variation
        a = luma.point(lambda p: int(10 + p * 0.1)) 
    elif finish_type == "satin":
        # Satin: Medium alpha (60-120), smooth
        a = luma.point(lambda p: int(60 + p * 0.2))
    elif finish_type == "gloss":
        # Gloss: High alpha (180-220), sharp reflections
        a = luma.point(lambda p: int(180 + p * 0.15))
    elif finish_type == "metallic":
        # Metallic: High alpha, but sensitive to color
        a = luma.point(lambda p: int(200 + p * 0.2))
    elif finish_type == "carbon":
        # Carbon: Weaver pattern in alpha
        carbon_pat = _generate_carbon_pattern(w)
        a = carbon_pat.convert("L").point(lambda p: int(p * 0.6)) # darker gloss
    else:
        a = Image.new("L", (w, h), 128)
        
    details.putalpha(a)
    return details

# =============================================================================
# PATTERN GENERATORS
# =============================================================================

def _generate_carbon_pattern(size: int, scale: int = 8) -> Image.Image:
    """Generates a seamless carbon fiber weave pattern."""
    # Create a small tile
    tile_size = scale * 2
    tile = Image.new("L", (tile_size, tile_size), 40)
    d = ImageDraw.Draw(tile)
    
    # Weave pattern
    d.rectangle([0, 0, scale, scale], fill=90)
    d.rectangle([scale, scale, tile_size, tile_size], fill=90)
    
    # Tile it
    return ImageOps.fit(tile, (size, size), method=Image.Resampling.NEAREST, centering=(0,0)) 

def blend_overlay(base: Image.Image, top: Image.Image) -> Image.Image:
    """Applies Overlay blend mode."""
    return ImageChops.overlay(base, top)

def blend_multiply(base: Image.Image, top: Image.Image) -> Image.Image:
    """Applies Multiply blend mode."""
    return ImageChops.multiply(base, top)

def colorize_pattern(pattern: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    """
    Colorizes a greyscale pattern (white = color, black = black).
    Preserves alpha.
    """
    if pattern.mode != "RGBA":
        pattern = pattern.convert("RGBA")
    
    # Split alpha
    r, g, b, a = pattern.split()
    gray = pattern.convert("L")
    
    colored = ImageOps.colorize(gray, "black", color)
    colored.putalpha(a)
    return colored


def blend_screen(base: Image.Image, top: Image.Image) -> Image.Image:
    """Applies Screen blend mode."""
    return ImageChops.screen(base, top)


def generate_topo_lines(size: int, color: Tuple[int, int, int], density: int = 18) -> Image.Image:
    """
    Generates topographic contour lines by thresholding smooth noise into bands.
    Returns RGBA with transparent background and colored lines.
    """
    # Smooth noise
    noise = Image.effect_noise((size // 3, size // 3), 14).resize((size, size), Image.Resampling.BICUBIC).convert("L")
    # Create contour bands
    step = max(6, int(255 / max(6, density)))
    bands = noise.point(lambda p: 255 if (p % step) < 2 else 0).filter(ImageFilter.GaussianBlur(0.6))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fill = Image.new("RGBA", (size, size), color + (220,))
    img.paste(fill, (0, 0), bands)
    return img


def generate_camo(size: int, palette: List[Tuple[int, int, int]], blobs: int = 220) -> Image.Image:
    """
    Generates simple camo blobs using layered noise thresholds.
    Returns RGBA (opaque blobs, transparent background).
    """
    rng = random.Random(1234)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base_noise = Image.effect_noise((size // 4, size // 4), 20).resize((size, size), Image.Resampling.BICUBIC).convert("L")
    base_noise = base_noise.filter(ImageFilter.GaussianBlur(2))

    # 3 layers by thresholds
    thresholds = sorted([rng.randint(70, 140), rng.randint(120, 180), rng.randint(160, 220)])
    for i, thr in enumerate(thresholds):
        col = palette[i % len(palette)]
        mask = base_noise.point(lambda p: 255 if p > thr else 0).filter(ImageFilter.GaussianBlur(3))
        layer = Image.new("RGBA", (size, size), col + (200,))
        img.paste(layer, (0, 0), mask)

    return img


def generate_glitch_bars(size: int, color: Tuple[int, int, int], strength: float = 1.0) -> Image.Image:
    """
    Generates glitchy horizontal bars / slices.
    """
    rng = random.Random(4242)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bars = int(20 * strength) + 8
    for _ in range(bars):
        y = rng.randrange(0, size)
        h = rng.randrange(max(2, size // 400), max(6, size // 120))
        x0 = rng.randrange(0, int(size * 0.7))
        w = rng.randrange(int(size * 0.15), int(size * 0.6))
        a = rng.randrange(60, 180)
        d.rectangle([x0, y, x0 + w, y + h], fill=color + (a,))
    return img

def generate_halftone(size: int, color: Tuple[int,int,int], density: float = 0.5) -> Image.Image:
    """Generates a clean halftone dot matrix."""
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    d = ImageDraw.Draw(img)
    
    step = size // 40
    radius = int(step * density * 0.8)
    
    for y in range(0, size, step):
        for x in range(0, size, step):
            # Hexagonal offset for better look
            offset_x = (step // 2) if (y // step) % 2 == 1 else 0
            
            cx = x + offset_x
            cy = y
            
            d.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=color + (255,))
            
    return img

def generate_circuit_traces(size: int, color: Tuple[int,int,int], density: str = "medium") -> Image.Image:
    """Generates the 'Tech-Luxe' circuit pattern."""
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    d = ImageDraw.Draw(img)
    
    grid = size // (60 if density == "high" else 40)
    num_traces = 100 if density == "high" else 50
    
    rng = random.Random(42) # Deterministic for consistent quality
    
    for _ in range(num_traces):
        sx = rng.randint(0, size // grid) * grid
        sy = rng.randint(0, size // grid) * grid
        
        points = [(sx, sy)]
        curr_x, curr_y = sx, sy
        
        for _ in range(rng.randint(3, 8)):
            dx, dy = rng.choice([(0,1), (0,-1), (1,0), (-1,0)])
            length = rng.randint(2, 6) * grid
            curr_x += dx * length
            curr_y += dy * length
            points.append((curr_x, curr_y))
            
        d.line(points, fill=color + (200,), width=max(2, size//400))
        # End cap (Via)
        r = max(3, size//300)
        d.ellipse([curr_x-r, curr_y-r, curr_x+r, curr_y+r], fill=color + (255,))
        d.ellipse([curr_x-r/2, curr_y-r/2, curr_x+r/2, curr_y+r/2], fill=(0,0,0,0)) # Hole
        
    return img

def generate_kintsugi_cracks(size: int, color: Tuple[int,int,int]) -> Image.Image:
    """Generates organic gold cracks."""
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    d = ImageDraw.Draw(img)
    
    # We simulate lightning/cracks using midpoint displacement or just jagged lines
    rng = random.Random(101)
    
    for _ in range(15): # Number of main cracks
        x = rng.randint(0, size)
        y = rng.randint(0, size)
        points = [(x,y)]
        
        angle = rng.uniform(0, math.pi*2)
        
        for _ in range(rng.randint(10, 25)):
            dist = rng.randint(20, 100)
            angle += rng.uniform(-0.5, 0.5)
            x += math.cos(angle) * dist
            y += math.sin(angle) * dist
            points.append((x,y))
            
            # Branching?
            if rng.random() > 0.7:
                 # Minimal branching logic here if needed
                 pass
                 
        d.line(points, fill=color + (255,), width=max(3, size//300))
        
    # Glow effect
    blur = img.filter(ImageFilter.GaussianBlur(3))
    return Image.alpha_composite(blur, img)


# =============================================================================
# NOVEL PATTERN GENERATORS (2026)
# =============================================================================

def generate_suminagashi(
    size: int,
    colors: List[Tuple[int, int, int]],
    seed: int = 42,
    num_drops: int = 12,
    rings_per_drop: int = 18,
    warp_strength: float = 1.0,
) -> Image.Image:
    """
    Generates a Suminagashi (Japanese floating-ink marbling) pattern.

    V2 -- high contrast rewrite:
    - Much higher ring frequency for thin, crisp alternating bands
    - Hard quantization (floor) for discrete color bands, not smooth gradients
    - 3-octave warp field for dramatic organic flow
    - Post-process contrast boost

    Returns RGBA (fully opaque).
    """
    rng = random.Random(seed)
    n_colors = max(2, len(colors))

    # Coordinate grid
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)

    # --- Warp field (3 octaves for rich, organic deformation) ---
    def _noise_octave(res: int, sigma: int) -> np.ndarray:
        return np.array(
            Image.effect_noise((max(1, res), max(1, res)), sigma)
            .resize((size, size), Image.Resampling.BICUBIC)
            .convert("L"),
            dtype=np.float64,
        ) / 255.0

    pot = (_noise_octave(max(1, size // 4), int(rng.uniform(8, 14))) * 0.50
           + _noise_octave(max(1, size // 6), int(rng.uniform(10, 18))) * 0.30
           + _noise_octave(max(1, size // 10), int(rng.uniform(12, 22))) * 0.20)

    gy, gx = np.gradient(pot)
    warp_px = warp_strength * size * 0.35  # Much stronger warp than v1
    wx = xs + gx * warp_px
    wy = ys + gy * warp_px

    # --- Drop concentric rings at random centers ---
    centers = []
    for _ in range(num_drops):
        cx = rng.uniform(size * -0.05, size * 1.05)
        cy = rng.uniform(size * -0.05, size * 1.05)
        centers.append((cx, cy))

    arr = np.zeros((size, size), dtype=np.float64)
    for cx, cy in centers:
        dist = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
        # High frequency -> thin bands. Random per drop for natural variation.
        freq = rng.uniform(0.18, 0.35)
        phase = rng.uniform(0, 2 * math.pi)
        rings = np.sin(dist * freq + phase)
        arr += rings

    # --- Quantize into hard color bands ---
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-9:
        arr = (arr - mn) / (mx - mn)

    # Floor quantization: each pixel maps to a discrete color index
    band_idx = np.clip(np.floor(arr * n_colors).astype(int), 0, n_colors - 1)

    # Slight local variation: within each band, use fractional part for subtle shading
    frac = (arr * n_colors) - np.floor(arr * n_colors)
    # Compress frac toward center for mostly-flat bands with soft edges
    frac = 0.5 + (frac - 0.5) * 0.3

    img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    img_rgba[:, :, 3] = 255

    for i in range(n_colors):
        mask = band_idx == i
        next_i = (i + 1) % n_colors
        for ch in range(3):
            val = colors[i][ch] * (1.0 - frac) + colors[next_i][ch] * frac
            img_rgba[:, :, ch] = np.where(mask, np.clip(val, 0, 255).astype(np.uint8), img_rgba[:, :, ch])

    img = Image.fromarray(img_rgba, "RGBA")

    # --- Post-process: boost contrast + slight sharpen for DXT readability ---
    from PIL import ImageEnhance
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=4))

    return img


def generate_moire_interference(
    size: int,
    line_color: Tuple[int, int, int],
    accent_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (10, 10, 10),
    seed: int = 42,
    num_grids: int = 2,
    line_density: int = 80,
) -> Image.Image:
    """
    Generates a moire interference pattern from overlapping radial/concentric grids.

    V2 -- high contrast rewrite:
    - Thin bright lines on deep dark background (not grey wash)
    - Sharp line profiles using pow() to narrow sine peaks
    - Wider center offsets for dramatic large-scale interference fringes
    - Strong accent glow at constructive interference peaks
    - Gamma curve to crush blacks and pop highlights

    Returns RGBA (fully opaque).
    """
    rng = random.Random(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)

    # Accumulate interference from multiple grid pairs
    acc = np.zeros((size, size), dtype=np.float64)

    for g in range(num_grids):
        # Wider spread for dramatic interference bands
        cx = size * rng.uniform(0.15, 0.85)
        cy = size * rng.uniform(0.15, 0.85)

        angle = np.arctan2(ys - cy, xs - cx)
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        density = line_density + rng.randint(-8, 8)

        # Radial wedge pattern -- thin bright lines using power sharpening
        radial_raw = np.sin(angle * density) * 0.5 + 0.5
        radial = np.power(radial_raw, 3.0)  # Sharpen: narrows peaks

        # Concentric ring pattern -- also sharpened
        ring_freq = rng.uniform(0.035, 0.065)
        ring_raw = np.sin(dist * ring_freq) * 0.5 + 0.5
        ring = np.power(ring_raw, 2.5)

        # Multiply radial and ring: only where both are bright -> line crossings
        layer = radial * 0.65 + ring * 0.35
        acc += layer

    # Normalize
    mn, mx = acc.min(), acc.max()
    if mx - mn > 1e-9:
        acc = (acc - mn) / (mx - mn)

    # Gamma curve: crush darks, pop brights. This is the key contrast move.
    gamma = 0.45  # < 1.0 brightens peaks, darkens lows after we invert the scale
    acc = np.power(acc, gamma)

    # Re-normalize after gamma
    mn2, mx2 = acc.min(), acc.max()
    if mx2 - mn2 > 1e-9:
        acc = (acc - mn2) / (mx2 - mn2)

    # Apply a second contrast squeeze: push bottom 30% to near-black
    acc = np.clip((acc - 0.30) / 0.70, 0, 1)

    # Build image
    img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    img_rgba[:, :, 3] = 255

    for ch in range(3):
        img_rgba[:, :, ch] = np.clip(
            bg_color[ch] * (1.0 - acc) + line_color[ch] * acc, 0, 255
        ).astype(np.uint8)

    # Strong accent glow at interference peaks (top 20%)
    peak_thresh = 0.80
    peak_strength = np.clip((acc - peak_thresh) / (1.0 - peak_thresh), 0, 1)
    # Blur for glow halo
    peak_img = Image.fromarray((peak_strength * 255).astype(np.uint8), "L")
    peak_img = peak_img.filter(ImageFilter.GaussianBlur(radius=max(3, size // 80)))
    peak_arr = np.array(peak_img, dtype=np.float64) / 255.0

    for ch in range(3):
        glow = peak_arr * accent_color[ch]
        img_rgba[:, :, ch] = np.clip(
            img_rgba[:, :, ch].astype(np.float64) + glow, 0, 255
        ).astype(np.uint8)

    img = Image.fromarray(img_rgba, "RGBA")

    # Final contrast + sharpen
    from PIL import ImageEnhance
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=3))

    return img


def generate_palimpsest(
    size: int,
    base_color: Tuple[int, int, int],
    spray_colors: List[Tuple[int, int, int]],
    grid_color: Tuple[int, int, int],
    veil_warm: Tuple[int, int, int],
    veil_cool: Tuple[int, int, int],
    seed: int = 42,
) -> Image.Image:
    """
    Generates a Palimpsest (layered urban abstraction) pattern, inspired by
    Julie Mehretu's BMW Art Car technique.

    Layers (bottom to top):
    1. Faded architectural/city-grid underlay (orthogonal + diagonal lines)
    2. Rotated fine dot grid (NOT standard halftone -- regular spacing, tilted)
    3. Bold gestural spray-paint arcs (bezier curves with feathered edges)
    4. Translucent radial color veil (warm-to-cool gradient wash)

    Returns RGBA (fully opaque).
    """
    rng = random.Random(seed)
    img = Image.new("RGBA", (size, size), (*base_color, 255))

    # ---- Layer 1: Architectural grid (faded city plan) ----
    grid_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    grid_opacity = 38  # Very faded
    gc = (*grid_color, grid_opacity)
    line_w = max(1, size // 600)

    # Major orthogonal grid
    spacing = max(20, size // 16)
    offset_x = rng.randint(0, spacing // 2)
    offset_y = rng.randint(0, spacing // 2)
    for x in range(offset_x, size, spacing):
        gd.line([(x, 0), (x, size)], fill=gc, width=line_w)
    for y in range(offset_y, size, spacing):
        gd.line([(0, y), (size, y)], fill=gc, width=line_w)

    # Diagonal roads (3-6 random diagonals at varying angles)
    for _ in range(rng.randint(3, 6)):
        x0 = rng.randint(0, size)
        y0 = rng.choice([0, size])
        angle = rng.uniform(20, 70)
        length = size * 1.5
        x1 = x0 + math.cos(math.radians(angle)) * length
        y1 = y0 + math.sin(math.radians(angle)) * length * (1 if y0 == 0 else -1)
        gd.line([(x0, y0), (x1, y1)], fill=(*grid_color, grid_opacity + 10), width=line_w + 1)

    # Occasional block rectangles (building footprints)
    for _ in range(rng.randint(6, 14)):
        bx = rng.randint(0, size)
        by = rng.randint(0, size)
        bw = rng.randint(size // 30, size // 12)
        bh = rng.randint(size // 30, size // 12)
        gd.rectangle([bx, by, bx + bw, by + bh], outline=(*grid_color, grid_opacity + 5), width=line_w)

    img = Image.alpha_composite(img, grid_layer)

    # ---- Layer 2: Rotated fine dot grid ----
    dot_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dot_layer)
    dot_spacing = max(8, size // 50)
    dot_r = max(1, size // 500)
    dot_opacity = 55
    dot_c = (*grid_color, dot_opacity)

    # Generate on a larger canvas then rotate
    pad = int(size * 0.5)
    big = Image.new("RGBA", (size + pad * 2, size + pad * 2), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    for y in range(0, size + pad * 2, dot_spacing):
        for x in range(0, size + pad * 2, dot_spacing):
            bd.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=dot_c)

    rot_angle = rng.uniform(10, 18)
    big = big.rotate(rot_angle, expand=False, resample=Image.Resampling.BICUBIC)
    # Crop back to size
    cx = (big.width - size) // 2
    cy = (big.height - size) // 2
    dot_layer = big.crop((cx, cy, cx + size, cy + size))
    img = Image.alpha_composite(img, dot_layer)

    # ---- Layer 3: Gestural spray-paint arcs ----
    for i, spray_c in enumerate(spray_colors):
        spray_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sd = ImageDraw.Draw(spray_layer)
        spray_opacity = rng.randint(100, 180)
        arc_width = max(8, int(size * rng.uniform(0.02, 0.06)))

        # Bezier-like arc: draw a series of connected line segments along a curve
        # Control points
        p0 = (rng.uniform(size * -0.1, size * 0.3), rng.uniform(size * 0.2, size * 0.8))
        p1 = (rng.uniform(size * 0.2, size * 0.5), rng.uniform(size * -0.1, size * 0.4))
        p2 = (rng.uniform(size * 0.5, size * 0.8), rng.uniform(size * 0.6, size * 1.1))
        p3 = (rng.uniform(size * 0.7, size * 1.1), rng.uniform(size * 0.1, size * 0.7))

        pts = []
        steps = 60
        for t_i in range(steps + 1):
            t = t_i / steps
            # Cubic bezier
            u = 1 - t
            x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
            y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
            pts.append((x, y))

        sd.line(pts, fill=(*spray_c, spray_opacity), width=arc_width, joint="curve")

        # Feather the spray with a blur for soft edges
        spray_layer = spray_layer.filter(ImageFilter.GaussianBlur(radius=max(2, arc_width // 3)))

        img = Image.alpha_composite(img, spray_layer)

    # ---- Layer 4: Radial color veil (warm-to-cool gradient wash) ----
    veil = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    veil_arr = np.zeros((size, size, 4), dtype=np.uint8)
    cy_v, cx_v = size * 0.45, size * 0.5
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dist = np.sqrt((xs - cx_v) ** 2 + (ys - cy_v) ** 2)
    max_dist = np.sqrt(cx_v ** 2 + cy_v ** 2) * 1.2
    t = np.clip(dist / max_dist, 0, 1)
    veil_opacity = 50

    for ch in range(3):
        veil_arr[:, :, ch] = np.clip(
            veil_warm[ch] * (1.0 - t) + veil_cool[ch] * t, 0, 255
        ).astype(np.uint8)
    veil_arr[:, :, 3] = veil_opacity
    veil = Image.fromarray(veil_arr, "RGBA")
    img = Image.alpha_composite(img, veil)

    return img


# =============================================================================
# V2 PATTERN GENERATORS (aesthetics overhaul)
# =============================================================================

def generate_carbon_v2(
    size: int,
    base_tone: int = 35,
    scale: int = 12,
    seed: int = 42,
) -> Image.Image:
    """Proper 2x2 twill carbon-fiber weave with per-thread brightness variation.

    Returns RGBA with transparent background; bright threads on dark gaps.
    Much higher fidelity than the V1 checkerboard tile.
    """
    rng = random.Random(seed)
    cell = max(4, scale)
    tile_w = cell * 4
    tile_h = cell * 4

    tile = Image.new("L", (tile_w, tile_h), base_tone)
    d = ImageDraw.Draw(tile)

    warp_bright = base_tone + 55
    weft_bright = base_tone + 40
    gap_dark = max(0, base_tone - 15)

    for row in range(4):
        for col in range(4):
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell - 1, y0 + cell - 1
            phase = (row + col) % 4
            if phase < 2:
                jitter = rng.randint(-6, 6)
                d.rectangle([x0, y0, x1, y1], fill=min(255, warp_bright + jitter))
            elif phase == 2:
                jitter = rng.randint(-5, 5)
                d.rectangle([x0, y0, x1, y1], fill=min(255, weft_bright + jitter))
            else:
                d.rectangle([x0, y0, x1, y1], fill=gap_dark)

    # Gap lines between threads
    gap_w = max(1, cell // 6)
    for i in range(5):
        pos = i * cell
        d.rectangle([pos, 0, pos + gap_w - 1, tile_h - 1], fill=gap_dark)
        d.rectangle([0, pos, tile_w - 1, pos + gap_w - 1], fill=gap_dark)

    reps_x = math.ceil(size / tile_w) + 1
    reps_y = math.ceil(size / tile_h) + 1
    tiled = Image.new("L", (tile_w * reps_x, tile_h * reps_y))
    for ty in range(reps_y):
        for tx in range(reps_x):
            tiled.paste(tile, (tx * tile_w, ty * tile_h))
    tiled = tiled.crop((0, 0, size, size))

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    colored = ImageOps.colorize(tiled, (0, 0, 0), (base_tone + 60, base_tone + 60, base_tone + 60))
    img.paste(colored.convert("RGBA"), (0, 0))
    return img


def _voronoi_cells(size: int, num_cells: int, seed: int) -> np.ndarray:
    """Returns a (size, size) int array of cell indices via brute-force Voronoi."""
    rng = np.random.RandomState(seed)
    cx = rng.randint(0, size, num_cells).astype(np.float64)
    cy = rng.randint(0, size, num_cells).astype(np.float64)

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    cell_map = np.zeros((size, size), dtype=np.int32)
    min_dist = np.full((size, size), 1e18, dtype=np.float64)

    for i in range(num_cells):
        d = (xs - cx[i]) ** 2 + (ys - cy[i]) ** 2
        closer = d < min_dist
        cell_map[closer] = i
        min_dist[closer] = d[closer]

    return cell_map


def generate_camo_v2(
    size: int,
    palette: List[Tuple[int, int, int]],
    cell_count: int = 80,
    seed: int = 42,
) -> Image.Image:
    """Voronoi-cell based organic camouflage with soft edge blending.

    Returns RGBA (opaque cells, transparent background).
    """
    cell_map = _voronoi_cells(size, cell_count, seed)
    n_colors = max(1, len(palette))

    img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    img_rgba[:, :, 3] = 200

    for i in range(cell_count):
        mask = cell_map == i
        col = palette[i % n_colors]
        for ch in range(3):
            img_rgba[:, :, ch] = np.where(mask, col[ch], img_rgba[:, :, ch])

    img = Image.fromarray(img_rgba, "RGBA")
    img = img.filter(ImageFilter.GaussianBlur(max(2, size // 300)))
    return img


def generate_racing_stripes(
    size: int,
    colors: List[Tuple[int, int, int]],
    stripe_widths: Optional[List[float]] = None,
    angle: float = 0.0,
    gap: float = 0.01,
) -> Image.Image:
    """Parametric multi-stripe system (Martini, Gulf, Le Mans style).

    Parameters
    ----------
    colors : list of RGB tuples for each stripe
    stripe_widths : normalized widths (0..1), defaults to equal
    angle : rotation in degrees (0 = vertical stripes)
    gap : normalized gap between stripes

    Returns RGBA with transparent background outside stripes.
    """
    n = len(colors)
    if stripe_widths is None:
        stripe_widths = [0.06] * n

    pad = int(size * 0.5)
    canvas_sz = size + pad * 2
    canvas = Image.new("RGBA", (canvas_sz, canvas_sz), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    center = canvas_sz / 2.0
    total_w = sum(stripe_widths) + gap * max(0, n - 1)
    start = center - (total_w * size) / 2.0
    cursor = start

    for i, (col, w) in enumerate(zip(colors, stripe_widths)):
        x0 = int(cursor)
        x1 = int(cursor + w * size)
        d.rectangle([x0, 0, x1, canvas_sz], fill=col + (230,))
        cursor = x1 + gap * size

    if abs(angle) > 0.1:
        canvas = canvas.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)

    cx = (canvas.width - size) // 2
    cy = (canvas.height - size) // 2
    return canvas.crop((cx, cy, cx + size, cy + size))


def generate_hex_tessellation(
    size: int,
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    cell_size: int = 60,
    seed: int = 42,
) -> Image.Image:
    """Hexagonal grid with per-cell color jitter between color1 and color2.

    Returns RGBA (fully opaque hexagons).
    """
    rng = random.Random(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    r = max(8, cell_size // 2)
    dx = r * 3
    dy = int(r * math.sqrt(3))

    for row in range(-1, size // dy + 2):
        for col in range(-1, size // dx + 2):
            cx = int(col * dx + (r * 1.5 if row % 2 else 0))
            cy = int(row * dy)

            t = rng.random()
            cr = int(color1[0] * (1 - t) + color2[0] * t)
            cg = int(color1[1] * (1 - t) + color2[1] * t)
            cb = int(color1[2] * (1 - t) + color2[2] * t)

            pts = []
            for a in range(6):
                angle = math.radians(60 * a + 30)
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                pts.append((px, py))
            d.polygon(pts, fill=(cr, cg, cb, 240))

            outline_alpha = rng.randint(30, 80)
            d.polygon(pts, outline=(0, 0, 0, outline_alpha))

    return img


def generate_metallic_flake(
    size: int,
    base_color: Tuple[int, int, int],
    flake_density: float = 0.5,
    seed: int = 42,
) -> Image.Image:
    """Fine sparkle noise simulating metallic-flake / candy paint.

    Returns RGBA (fully opaque). The flake is subtle brightness variation
    over the base color -- designed to look alive in motion via mipmap aliasing.
    """
    rng_np = np.random.RandomState(seed)

    noise = rng_np.normal(0, 1, (size, size)).astype(np.float64)
    # High-frequency flake: no blur, just clip to sparkle points
    amplitude = 25 + int(35 * flake_density)
    flake = np.clip(noise * amplitude, -amplitude, amplitude)

    img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    img_rgba[:, :, 3] = 255
    for ch in range(3):
        img_rgba[:, :, ch] = np.clip(
            base_color[ch] + flake, 0, 255
        ).astype(np.uint8)

    return Image.fromarray(img_rgba, "RGBA")


def generate_weathering(
    size: int,
    intensity: float = 0.5,
    seed: int = 42,
) -> Image.Image:
    """Layered weathering: paint chips, micro-scratches, UV fade zones.

    Returns RGBA with transparent background (overlay on top of skin).
    intensity: 0.0 (pristine) .. 1.0 (heavily weathered).
    """
    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # --- Paint chips (small irregular bright spots revealing primer) ---
    chip_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d_chip = ImageDraw.Draw(chip_layer)
    num_chips = int(60 * intensity) + 5
    primer = (180, 170, 155)  # warm grey primer underneath
    for _ in range(num_chips):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        rx = rng.randint(2, max(3, int(size * 0.008 * intensity)))
        ry = rng.randint(2, max(3, int(size * 0.006 * intensity)))
        alpha = rng.randint(80, 180)
        d_chip.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=primer + (alpha,))
    chip_layer = chip_layer.filter(ImageFilter.GaussianBlur(0.8))
    img = Image.alpha_composite(img, chip_layer)

    # --- Micro-scratches (thin bright lines) ---
    scratch_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d_scr = ImageDraw.Draw(scratch_layer)
    num_scratches = int(40 * intensity) + 3
    for _ in range(num_scratches):
        x0 = rng.randint(0, size)
        y0 = rng.randint(0, size)
        angle = rng.uniform(0, math.pi)
        length = rng.randint(size // 40, size // 8)
        x1 = x0 + int(math.cos(angle) * length)
        y1 = y0 + int(math.sin(angle) * length)
        alpha = rng.randint(40, 120)
        d_scr.line([(x0, y0), (x1, y1)], fill=(200, 200, 200, alpha), width=1)
    img = Image.alpha_composite(img, scratch_layer)

    # --- UV fade zone (top area gets slightly lighter / desaturated) ---
    fade_arr = np.zeros((size, size, 4), dtype=np.uint8)
    ys = np.arange(size, dtype=np.float64)
    fade_t = np.clip(1.0 - ys / (size * 0.6), 0, 1) ** 2
    fade_strength = int(40 * intensity)
    for y_idx in range(size):
        val = int(fade_t[y_idx] * fade_strength)
        fade_arr[y_idx, :, 0] = val
        fade_arr[y_idx, :, 1] = val
        fade_arr[y_idx, :, 2] = val
        fade_arr[y_idx, :, 3] = int(fade_t[y_idx] * fade_strength * 1.5)
    fade_img = Image.fromarray(fade_arr, "RGBA")
    fade_img = fade_img.filter(ImageFilter.GaussianBlur(size // 40))
    img = Image.alpha_composite(img, fade_img)

    return img


def generate_voronoi_shatter(
    size: int,
    colors: List[Tuple[int, int, int]],
    cell_count: int = 50,
    seed: int = 42,
    edge_color: Tuple[int, int, int] = (220, 220, 220),
) -> Image.Image:
    """Shattered glass / crystal pattern using Voronoi cells with bright edges.

    Returns RGBA (fully opaque).
    """
    cell_map = _voronoi_cells(size, cell_count, seed)
    n_colors = max(1, len(colors))
    rng_np = np.random.RandomState(seed + 1)
    color_assign = rng_np.randint(0, n_colors, cell_count)

    img_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    img_rgba[:, :, 3] = 255

    for i in range(cell_count):
        mask = cell_map == i
        col = colors[color_assign[i]]
        brightness_jitter = rng_np.randint(-15, 15)
        for ch in range(3):
            img_rgba[:, :, ch] = np.where(
                mask,
                np.clip(col[ch] + brightness_jitter, 0, 255),
                img_rgba[:, :, ch],
            )

    img = Image.fromarray(img_rgba, "RGBA")

    # Draw edges at cell boundaries
    edges = np.zeros((size, size), dtype=np.uint8)
    # Horizontal boundaries
    edges[:-1, :] = np.where(cell_map[:-1, :] != cell_map[1:, :], 255, edges[:-1, :])
    # Vertical boundaries
    edges[:, :-1] = np.where(cell_map[:, :-1] != cell_map[:, 1:], 255, edges[:, :-1])

    edge_mask = Image.fromarray(edges, "L")
    edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(0.6))

    edge_layer = Image.new("RGBA", (size, size), edge_color + (0,))
    edge_layer.putalpha(edge_mask)
    img = Image.alpha_composite(img, edge_layer)

    return img


def _fbm_noise(size: int, octaves: int, rng: np.random.RandomState) -> np.ndarray:
    """Fractal Brownian motion via interpolated random grids.
    Returns a float64 array normalized to [0, 1]."""
    result = np.zeros((size, size), dtype=np.float64)
    amp = 1.0
    for i in range(octaves):
        grid_dim = max(4, 4 * (2 ** i))
        if grid_dim > size:
            grid_dim = size
        noise = rng.rand(grid_dim + 1, grid_dim + 1)
        noise_img = Image.fromarray((noise * 255).astype(np.uint8), "L")
        noise_up = np.array(
            noise_img.resize((size, size), Image.BICUBIC), dtype=np.float64
        ) / 255.0
        result += noise_up * amp
        amp *= 0.5
    lo, hi = result.min(), result.max()
    return (result - lo) / (hi - lo + 1e-9)


def _turbulence_field(size: int, octaves: int, rng: np.random.RandomState) -> np.ndarray:
    """Turbulence: sum of |noise| at increasing frequencies (sharper ridges)."""
    result = np.zeros((size, size), dtype=np.float64)
    freq = 1.0
    for i in range(octaves):
        grid_dim = max(4, 4 * (2 ** i))
        if grid_dim > size:
            grid_dim = size
        noise = rng.rand(grid_dim + 1, grid_dim + 1)
        noise_img = Image.fromarray((noise * 255).astype(np.uint8), "L")
        noise_up = np.array(
            noise_img.resize((size, size), Image.BICUBIC), dtype=np.float64
        ) / 255.0
        result += np.abs(noise_up - 0.5) * 2.0 / freq
        freq *= 2.07
    return result


def generate_weaponized_115(
    size: int,
    green_peak: Tuple[int, int, int] = (65, 255, 100),
    green_mid: Tuple[int, int, int] = (20, 160, 40),
    dark_color: Tuple[int, int, int] = (5, 8, 4),
    vein_freq: float = 4.0,
    turb_amplitude: float = 10.0,
    octaves: int = 7,
    sharpness: float = 0.28,
    warp_strength: float = 0.0,
    threshold: float = 0.62,
    hotspot_strength: float = 0.08,
    seed: int = 115,
) -> Image.Image:
    """Weaponized 115 camo using turbulent marble noise.

    Produces organic flowing regions of bright neon green and deep black,
    like a radioactive mineral with swirling veins -- the classic marble
    noise algorithm: sin(coord + amplitude * turbulence).

    Parameters
    ----------
    green_peak : RGB for the brightest green areas
    green_mid  : RGB for mid-green transition zone
    dark_color : RGB for the dark vein interiors
    vein_freq  : base frequency of the sine wave (controls vein count)
    turb_amplitude : strength of turbulence distortion (higher = more chaotic)
    octaves    : number of noise octaves (detail levels)
    sharpness  : controls the green/black boundary steepness (lower = sharper)
    warp_strength : domain warping intensity (0 = off, 0.3+ = fluid/chaotic)
    threshold  : dark/green cutoff (lower = more green coverage, default 0.62)
    hotspot_strength : brightness of random hot specks in green areas
    seed       : RNG seed
    """
    rng = np.random.RandomState(seed)

    # -- Step 1: turbulence field (multi-octave abs-noise) --
    turb = _turbulence_field(size, octaves, np.random.RandomState(seed))

    # -- Step 2: coordinate grids --
    y_coord = np.linspace(0, 2 * math.pi * vein_freq, size).reshape(-1, 1)
    x_coord = np.linspace(0, 2 * math.pi * vein_freq, size).reshape(1, -1)
    y_grid = np.broadcast_to(y_coord, (size, size)).copy()
    x_grid = np.broadcast_to(x_coord, (size, size)).copy()

    # -- Step 2b: domain warping (distort coordinates for fluid look) --
    if warp_strength > 0:
        warp_x = _fbm_noise(size, 5, np.random.RandomState(seed + 500))
        warp_y = _fbm_noise(size, 5, np.random.RandomState(seed + 600))
        warp_scale = warp_strength * 2 * math.pi * vein_freq
        x_grid += (warp_x - 0.5) * warp_scale
        y_grid += (warp_y - 0.5) * warp_scale

        # Second level of warping for extra chaos (warp-on-warp)
        warp_x2 = _fbm_noise(size, 4, np.random.RandomState(seed + 700))
        warp_y2 = _fbm_noise(size, 4, np.random.RandomState(seed + 800))
        x_grid += (warp_x2 - 0.5) * warp_scale * 0.5
        y_grid += (warp_y2 - 0.5) * warp_scale * 0.5

    # -- Step 3: marble veins from two directions using warped coords --
    asc_noise = _fbm_noise(size, 3, np.random.RandomState(seed + 77))
    ascent = -0.5 + asc_noise * 2.0

    t1 = y_grid + ascent * x_grid + turb_amplitude * turb
    marble_a = np.sin(t1)

    turb2 = _turbulence_field(size, octaves - 1, np.random.RandomState(seed + 42))
    t2 = x_grid * 0.8 + y_grid * 0.4 + turb_amplitude * 0.75 * turb2
    marble_b = np.sin(t2)

    # Third layer at yet another angle for full isotropy
    turb3 = _turbulence_field(size, octaves - 2, np.random.RandomState(seed + 99))
    t3 = x_grid * 0.3 - y_grid * 0.9 + turb_amplitude * 0.6 * turb3
    marble_c = np.sin(t3)

    raw = marble_a * 0.40 + marble_b * 0.35 + marble_c * 0.25

    # -- Step 4: shape to [0, 1] with controllable boundary sharpness --
    raw_01 = 0.5 * (raw + 1.0)
    raw_01 = np.sqrt(raw_01)

    marble_val = np.clip((raw_01 - threshold) / sharpness, 0.0, 1.0)

    # Detail texture layer
    detail = _fbm_noise(size, 5, np.random.RandomState(seed + 200))
    marble_val = np.clip(marble_val + (detail - 0.5) * 0.12, 0.0, 1.0)

    # -- Step 5: color mapping (three-stop: dark -> mid green -> peak green) --
    img_arr = np.zeros((size, size, 4), dtype=np.uint8)
    img_arr[:, :, 3] = 255

    for ch in range(3):
        d, m, p = float(dark_color[ch]), float(green_mid[ch]), float(green_peak[ch])
        val = np.where(
            marble_val < 0.4,
            d + (marble_val / 0.4) * (m - d),
            m + ((marble_val - 0.4) / 0.6) * (p - m),
        )
        img_arr[:, :, ch] = np.clip(val, 0, 255).astype(np.uint8)

    # -- Step 6: baked glow -- wide green halos at green/dark boundaries --
    # (Illum.dds doesn't work in Stadium, so glow must live in the Diffuse)
    green_channel = img_arr[:, :, 1].astype(np.float64)
    dark_mask_raw = np.clip(1.0 - marble_val * 2.5, 0.0, 1.0)
    dark_img = Image.fromarray((dark_mask_raw * 255).astype(np.uint8), "L")
    # Wide blur for broad glow bleed
    glow_wide = np.array(
        dark_img.filter(ImageFilter.GaussianBlur(18)), dtype=np.float64
    ) / 255.0
    # Narrow blur for sharp inner edge
    glow_narrow = np.array(
        dark_img.filter(ImageFilter.GaussianBlur(6)), dtype=np.float64
    ) / 255.0
    # Edge glow = where green meets dark (green side only)
    green_side = (marble_val > 0.15).astype(np.float64)
    edge_glow = (glow_wide * 0.6 + glow_narrow * 0.4) * green_side
    glow_strength = 0.35
    for ch in range(3):
        img_arr[:, :, ch] = np.clip(
            img_arr[:, :, ch].astype(np.float64) + edge_glow * green_peak[ch] * glow_strength,
            0, 255,
        ).astype(np.uint8)

    # -- Step 7: bright hotspots for depth variation --
    hot = _fbm_noise(size, 4, np.random.RandomState(seed + 300))
    hot_mask = (marble_val > 0.35) * hot
    for ch in range(3):
        # Push hottest spots towards white-green (not just brighter green)
        peak_boost = min(255, int(green_peak[ch] * 1.2 + 40))
        img_arr[:, :, ch] = np.clip(
            img_arr[:, :, ch].astype(np.float64) + hot_mask * peak_boost * hotspot_strength,
            0, 255,
        ).astype(np.uint8)

    # -- Step 7b: radioactive bright specks (scattered particles) --
    speck_noise = _fbm_noise(size, 6, np.random.RandomState(seed + 400))
    speck_mask = (speck_noise > 0.78) & (marble_val > 0.25)
    speck_brightness = speck_noise * speck_mask * hotspot_strength * 6.0
    for ch in range(3):
        # Specks push towards white-green for maximum visibility
        boost = min(255, int(green_peak[ch] * 0.5 + 160))
        img_arr[:, :, ch] = np.clip(
            img_arr[:, :, ch].astype(np.float64) + speck_brightness * boost,
            0, 255,
        ).astype(np.uint8)

    # -- Step 8: fine noise for surface grit --
    noise = rng.randint(-6, 6, (size, size), dtype=np.int16)
    final = img_arr.astype(np.int16)
    for ch in range(3):
        final[:, :, ch] = np.clip(final[:, :, ch] + noise, 0, 255)

    # -- Step 9: encode per-pixel finish alpha --
    # Low alpha (bright/emissive look) on green, moderate alpha (wet/reflective) on dark
    # Per TMNF: black alpha = bright/matte, white alpha = dull/reflective
    green_val = final[:, :, 1].astype(np.float64) / 255.0
    alpha_dark = 0xA0   # dark veins: moderate reflection (wet look)
    alpha_green = 0x50  # green areas: low alpha = bright, almost self-luminous
    finish_alpha = alpha_dark + (alpha_green - alpha_dark) * np.clip(green_val * 1.5, 0, 1)
    final[:, :, 3] = np.clip(finish_alpha, 0, 255).astype(np.int16)

    return Image.fromarray(final.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# Psychedelic pattern generators
# ---------------------------------------------------------------------------

def _hsv_to_rgb(h, s, v):
    """Vectorized HSV->RGB. h:[0,360], s/v:[0,1]. Returns (r,g,b) uint8 arrays."""
    h = np.asarray(h, dtype=np.float64) % 360.0
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = v * s
    hp = h / 60.0
    x = c * (1 - np.abs(hp % 2 - 1))
    m = v - c
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    for lo, hi, rc, gc, bc in [
        (0, 1, c, x, 0), (1, 2, x, c, 0), (2, 3, 0, c, x),
        (3, 4, 0, x, c), (4, 5, x, 0, c), (5, 6, c, 0, x),
    ]:
        mask = (hp >= lo) & (hp < hi)
        r[mask] = rc if np.isscalar(rc) else rc[mask]
        g[mask] = gc if np.isscalar(gc) else gc[mask]
        b[mask] = bc if np.isscalar(bc) else bc[mask]
    return (
        np.clip((r + m) * 255, 0, 255).astype(np.uint8),
        np.clip((g + m) * 255, 0, 255).astype(np.uint8),
        np.clip((b + m) * 255, 0, 255).astype(np.uint8),
    )


def generate_warp_grid(
    size: int,
    grid_count: int = 22,
    warp_amount: float = 0.18,
    line_base: float = 0.04,
    line_var: float = 0.10,
    glow_strength: float = 0.40,
    hue_center: Optional[float] = None,
    hue_spread: float = 300.0,
    seed: int = 42,
) -> Image.Image:
    """Neon grid on dark void with glow bloom, variable line thickness,
    bright intersection nodes, subtle background texture, and particle specks.

    line_base/line_var : line width = line_base + noise * line_var
    glow_strength : bloom intensity around lines (0 = off, 0.4 = normal, 0.7+ = bold)
    hue_center : if set, locks colors around this hue (0-360). None = full rainbow.
    hue_spread : how wide the hue range is (degrees). 300 = rainbow, 40 = mono-ish.
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size

    # -- warp coordinates --
    wx = _fbm_noise(size, 6, np.random.RandomState(seed))
    wy = _fbm_noise(size, 6, np.random.RandomState(seed + 100))
    xw = x + (wx - 0.5) * warp_amount
    yw = y + (wy - 0.5) * warp_amount

    # -- variable line width via noise --
    width_noise = _fbm_noise(size, 4, np.random.RandomState(seed + 300))
    lw = line_base + width_noise * line_var

    gx = np.abs(np.sin(xw * math.pi * grid_count))
    gy = np.abs(np.sin(yw * math.pi * grid_count))

    line_x = np.clip(1.0 - gx / lw, 0, 1) ** 1.8
    line_y = np.clip(1.0 - gy / lw, 0, 1) ** 1.8
    lines = np.maximum(line_x, line_y)

    # -- bright intersection nodes --
    intersect = line_x * line_y
    node_boost = np.clip(intersect * 3.0, 0, 1)
    lines = np.clip(lines + node_boost * 0.6, 0, 1)

    # -- anti-alias: slight blur on raw line mask --
    lines_img = Image.fromarray(
        np.clip(lines * 255, 0, 255).astype(np.uint8), "L"
    ).filter(ImageFilter.GaussianBlur(0.8))
    lines = np.array(lines_img, dtype=np.float64) / 255.0

    # -- color mapping --
    hue_noise = _fbm_noise(size, 4, np.random.RandomState(seed + 200))
    if hue_center is not None:
        hue = (hue_center - hue_spread / 2 + hue_noise * hue_spread) % 360.0
    else:
        hue = (hue_noise * hue_spread + xw * 80.0 + yw * 60.0) % 360.0
    sat = np.full((size, size), 0.95)

    # -- subtle background texture --
    bg_noise = _fbm_noise(size, 5, np.random.RandomState(seed + 400))
    bg_val = 0.015 + bg_noise * 0.025

    val = np.where(lines > 0.05, lines * 0.95, bg_val)

    rc, gc, bc = _hsv_to_rgb(hue, sat, val)
    img_arr = np.stack([rc, gc, bc], axis=-1).astype(np.float64)

    # -- glow bloom: neon halo around lines --
    lum = np.clip(lines * 255, 0, 255).astype(np.uint8)
    lum_pil = Image.fromarray(lum, "L")
    glow_wide = np.array(
        lum_pil.filter(ImageFilter.GaussianBlur(max(2, size // 60))),
        dtype=np.float64) / 255.0
    glow_mid = np.array(
        lum_pil.filter(ImageFilter.GaussianBlur(max(1, size // 160))),
        dtype=np.float64) / 255.0
    glow = glow_wide * 0.45 + glow_mid * 0.55

    glow_r, glow_g, glow_b = _hsv_to_rgb(hue, sat, np.ones_like(hue))
    for ch, gc_arr in enumerate([glow_r, glow_g, glow_b]):
        colored_glow = glow * gc_arr.astype(np.float64)
        img_arr[:, :, ch] = np.clip(
            img_arr[:, :, ch] + colored_glow * glow_strength, 0, 255)

    # -- particle specks along lines --
    speck_noise = _fbm_noise(size, 6, np.random.RandomState(seed + 500))
    speck_mask = (speck_noise > 0.80) & (lines > 0.15)
    speck_val = speck_noise * speck_mask.astype(np.float64) * 180.0
    for ch in range(3):
        img_arr[:, :, ch] = np.clip(img_arr[:, :, ch] + speck_val, 0, 255)

    # -- surface grit --
    grit = rng.randint(-4, 4, (size, size), dtype=np.int16)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    for ch in range(3):
        out[:, :, ch] = np.clip(
            img_arr[:, :, ch].astype(np.int16) + grit, 0, 255).astype(np.uint8)
    out[:, :, 3] = 255
    return Image.fromarray(out, "RGBA")
