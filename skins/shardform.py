#!/usr/bin/env python3
"""
SHARDFORM -- low-poly crystal tessellation.

Poisson-disk samples are scattered across every UV island, Delaunay-
triangulated, and each triangle is filled with a flat colour picked
from a per-variant palette.  Shading uses a pseudo-3D Lambert value
(triangle centroid relative to image centre vs a virtual light dir)
so the panels read as faceted crystal under stage lighting.

Several triangles get bumped to an emissive "light leak" fill for a
prism pop, then the whole frame is bloomed and the panel seams are
rimmed in the hero colour.

Variants differ only by palette + emissive colours.  No tire mods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter

from skin_canvas import SkinCanvas
from skins._gaming_common import install_robust_masks


# ---------------------------------------------------------------------
# Variant palettes -- each has a light-to-dark hull palette, emissive
# pops, and a rim colour for the panel seams.
# ---------------------------------------------------------------------

PRESETS = {
    # original prism (pushed saturation vs v1)
    "Shardform": {
        "hull": [
            (8, 14, 28),   (14, 22, 40),   (24, 40, 64),
            (30, 70, 100), (50, 130, 170), (90, 200, 230),
            (180, 240, 250), (230, 250, 255),
        ],
        "emissive": [(40, 235, 255), (255, 70, 210), (140, 100, 255)],
        "rim": (60, 230, 255),
        "seed": 9,
    },
    # molten / ember tessellation
    "Shardform_Ember": {
        "hull": [
            (10, 4, 2),   (22, 8, 4),    (50, 16, 6),
            (110, 32, 10), (200, 70, 18), (250, 140, 30),
            (255, 210, 90), (255, 245, 180),
        ],
        "emissive": [(255, 180, 40), (255, 80, 40), (255, 230, 120)],
        "rim": (255, 120, 40),
        "seed": 21,
    },
    # toxic / acid lab
    "Shardform_Toxic": {
        "hull": [
            (4, 10, 4),    (10, 22, 8),    (18, 44, 16),
            (36, 90, 28),  (80, 160, 40),  (160, 230, 60),
            (210, 255, 130), (240, 255, 200),
        ],
        "emissive": [(160, 255, 40), (255, 240, 60), (60, 255, 180)],
        "rim": (160, 255, 60),
        "seed": 33,
    },
    # royal / imperial violet
    "Shardform_Royal": {
        "hull": [
            (8, 4, 18),    (16, 8, 30),    (30, 14, 52),
            (60, 24, 90),  (110, 40, 150), (180, 70, 220),
            (230, 160, 255), (245, 220, 255),
        ],
        "emissive": [(255, 210, 80), (240, 80, 240), (120, 100, 255)],
        "rim": (210, 120, 255),
        "seed": 41,
    },
    # arctic / ice
    "Shardform_Arctic": {
        "hull": [
            (4, 8, 16),    (8, 18, 34),    (18, 40, 70),
            (40, 90, 140), (90, 160, 220), (160, 220, 250),
            (220, 245, 255), (255, 255, 255),
        ],
        "emissive": [(200, 240, 255), (120, 200, 255), (255, 255, 255)],
        "rim": (180, 230, 255),
        "seed": 55,
    },

    # --- Cyberpunk variants ------------------------------------------------
    # Neotokyo -- Blade Runner 2049 neon: hot pink + electric cyan on black.
    "Shardform_Neotokyo": {
        "hull": [
            (6, 4, 14),    (16, 6, 28),    (38, 10, 60),
            (80, 18, 100), (160, 30, 140), (240, 60, 180),
            (255, 120, 220), (255, 220, 255),
        ],
        "emissive": [(255, 40, 180), (40, 240, 255), (255, 240, 60)],
        "rim": (255, 50, 200),
        "seed": 67,
    },

    # Synthwave -- 1980s retro grid: indigo/violet base, hot-pink + electric
    # blue emissives, deliberately bright/saturated.
    "Shardform_Synthwave": {
        "hull": [
            (8, 4, 22),    (18, 8, 44),    (40, 14, 80),
            (80, 28, 140), (140, 50, 200), (220, 90, 255),
            (255, 140, 230), (255, 210, 250),
        ],
        "emissive": [(255, 60, 180), (80, 120, 255), (255, 200, 80)],
        "rim": (255, 100, 220),
        "seed": 79,
    },

    # Matrix -- pure code-rain: jet black with cascading acid-green shards
    # and rare white glitch emissives.
    "Shardform_Matrix": {
        "hull": [
            (4, 8, 4),    (8, 20, 8),      (16, 40, 14),
            (28, 80, 20), (60, 140, 30),   (120, 220, 50),
            (180, 255, 120), (240, 255, 200),
        ],
        "emissive": [(120, 255, 60), (220, 255, 180), (255, 255, 255)],
        "rim": (120, 255, 60),
        "seed": 89,
    },

    # Infrared -- thermal signature: deep crimson base ramping to white-hot
    # core.  Gaming "enemy heat-vision" vibe.
    "Shardform_Infrared": {
        "hull": [
            (12, 4, 4),   (28, 8, 6),      (70, 14, 8),
            (140, 28, 10), (220, 60, 20),  (255, 140, 40),
            (255, 220, 120), (255, 250, 220),
        ],
        "emissive": [(255, 240, 160), (255, 80, 40), (255, 160, 20)],
        "rim": (255, 80, 20),
        "seed": 97,
    },

    # Venom -- chartreuse cyber-viper: near-black olive, toxic yellow-green,
    # with rare violet emissive pops.
    "Shardform_Venom": {
        "hull": [
            (6, 10, 2),   (14, 22, 4),     (30, 50, 10),
            (60, 100, 20), (120, 180, 30), (200, 255, 60),
            (230, 255, 140), (250, 255, 220),
        ],
        "emissive": [(220, 255, 40), (150, 60, 255), (40, 255, 160)],
        "rim": (200, 255, 50),
        "seed": 103,
    },

    # Hologram -- iridescent rainbow shift.  Hull cycles through full
    # spectrum (cyan -> magenta -> yellow) so individual triangles look
    # like glass shards refracting light.
    "Shardform_Hologram": {
        "hull": [
            (8, 8, 20),     (30, 20, 70),   (30, 90, 180),
            (80, 200, 220), (220, 90, 220), (255, 160, 100),
            (255, 230, 160), (240, 240, 255),
        ],
        "emissive": [(40, 230, 255), (255, 80, 230), (255, 230, 60)],
        "rim": (220, 140, 255),
        "seed": 113,
    },
}


LIGHT_DIR = np.array([-0.4, -0.6], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


def _poisson_disk(mask_arr, min_dist, rng, boost_centroid=1.0):
    """Poisson-disk sample points inside a boolean mask using spatial grid."""
    H, W = mask_arr.shape
    ys, xs = np.where(mask_arr > 128)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cx, cy = xs.mean(), ys.mean()
    cell = max(1, int(min_dist / math.sqrt(2)))
    gw = W // cell + 1
    gh = H // cell + 1
    grid = -np.ones((gh, gw), dtype=np.int32)
    pts = []

    def _accept(x, y):
        gx, gy = int(x / cell), int(y / cell)
        for ix in range(max(0, gx - 2), min(gw, gx + 3)):
            for iy in range(max(0, gy - 2), min(gh, gy + 3)):
                pi = grid[iy, ix]
                if pi >= 0:
                    px, py = pts[pi]
                    if (px - x) ** 2 + (py - y) ** 2 < min_dist ** 2:
                        return False
        return True

    start = rng.integers(0, len(xs))
    p0 = (float(xs[start]), float(ys[start]))
    pts.append(p0)
    grid[int(p0[1] / cell), int(p0[0] / cell)] = 0
    active = [0]

    while active:
        idx = active[rng.integers(0, len(active))]
        px, py = pts[idx]
        placed = False
        for _ in range(24):
            r = min_dist * (1 + rng.random())
            theta = rng.uniform(0, 2 * math.pi)
            nx = px + r * math.cos(theta)
            ny = py + r * math.sin(theta)
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if mask_arr[int(ny), int(nx)] <= 128:
                continue
            if boost_centroid != 1.0:
                d = math.hypot(nx - cx, ny - cy)
                local_min = min_dist * (1.0 + (1.0 - boost_centroid)
                                         * (d / (0.5 * max(W, H))))
                gx, gy = int(nx / cell), int(ny / cell)
                ok = True
                for ix in range(max(0, gx - 2), min(gw, gx + 3)):
                    for iy in range(max(0, gy - 2), min(gh, gy + 3)):
                        pi = grid[iy, ix]
                        if pi >= 0:
                            qx, qy = pts[pi]
                            if (qx - nx) ** 2 + (qy - ny) ** 2 < local_min ** 2:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    continue
            else:
                if not _accept(nx, ny):
                    continue
            pts.append((nx, ny))
            grid[int(ny / cell), int(nx / cell)] = len(pts) - 1
            active.append(len(pts) - 1)
            placed = True
            break
        if not placed:
            active.remove(idx)
    return np.array(pts, dtype=np.float32)


def _add_island_edge_points(mask_arr, rng, spacing=28):
    img = Image.fromarray(mask_arr, "L")
    edges = np.array(img.filter(ImageFilter.FIND_EDGES))
    ys, xs = np.where(edges > 64)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    n = max(4, len(xs) // spacing)
    idx = rng.choice(len(xs), size=n, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def _fill_triangles(points, tri, palette, emissive, rng, size):
    """Rasterise every triangle that has a centroid inside the canvas.
    No mask gating -- we want full-canvas coverage so no UV gap is bare."""
    W = H = size
    img_cx, img_cy = W / 2, H / 2
    palette_arr = np.array(palette, dtype=np.float32)

    fill_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_img)
    edge_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_img)
    glow_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_img)

    edge_col = tuple(int(c * 0.65) for c in palette[-2]) + (160,)

    for s_idx in range(len(tri.simplices)):
        s = tri.simplices[s_idx]
        v = points[s]
        centroid = v.mean(axis=0)
        cx, cy = int(centroid[0]), int(centroid[1])
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        ax, ay = v[0]; bx, by = v[1]; cx_, cy_ = v[2]
        area2 = abs((bx - ax) * (cy_ - ay) - (cx_ - ax) * (by - ay))
        if area2 < 40:
            continue
        d = centroid - np.array([img_cx, img_cy])
        norm = np.linalg.norm(d)
        if norm > 1e-3:
            d = d / norm
        lambert = max(0.35, 0.7 + 0.45 * np.dot(-d, LIGHT_DIR))
        lambert = min(lambert, 1.2)
        dist_norm = min(1.0, norm / (0.55 * max(W, H)))
        pal_idx = int(np.clip(
            (1.0 - dist_norm * 0.55) * (len(palette) - 1)
            + rng.normal(0, 1.0),
            0, len(palette) - 1,
        ))
        base = palette_arr[pal_idx]
        shaded = np.clip(base * lambert, 0, 255).astype(np.uint8)
        verts = [tuple(p) for p in v.tolist()]
        fill_draw.polygon(verts, fill=(int(shaded[0]), int(shaded[1]),
                                        int(shaded[2]), 255))

        if rng.random() < 0.08:
            em = emissive[int(rng.integers(0, len(emissive)))]
            glow_draw.polygon(verts, fill=em + (220,))

        for i in range(3):
            a = v[i]; b = v[(i + 1) % 3]
            edge_draw.line([tuple(a), tuple(b)],
                           fill=edge_col, width=1)

    return fill_img, edge_img, glow_img


def build_pattern(preset_name, size=2048):
    """Generate a full-canvas triangulation.  We tessellate the ENTIRE
    2048x2048 frame (not just UV islands) so every UV unwrap coordinate --
    including the fenders and any edge bleed -- lands on a coloured
    triangle.  If we only painted inside islands, the enhance-pipeline
    contrast boost would clamp our dark base to pure black, and
    ProSkinEngine._fill_uv_gaps would then flood-fill the rim colour
    across every 'gap' -- that was why the fenders rendered solid yellow.
    """
    p = PRESETS[preset_name]
    rng = np.random.default_rng(p["seed"])

    # Uniform Poisson-disk across the whole canvas.  ~1 point per
    # 700 sq-px matches the density we liked inside islands before.
    full_mask = np.full((size, size), 255, dtype=np.uint8)
    target_total = int(size * size / 700)
    min_dist = max(12, int(math.sqrt(size * size / target_total * 0.9)))
    pts = _poisson_disk(full_mask, min_dist, rng, boost_centroid=1.0)

    corners = np.array([[0, 0], [size - 1, 0],
                        [size - 1, size - 1], [0, size - 1]],
                       dtype=np.float32)
    pts = np.vstack([pts, corners])

    tri = Delaunay(pts)

    fill_img, edge_img, glow_img = _fill_triangles(
        pts, tri, p["hull"], p["emissive"], rng, size,
    )

    # base canvas: palette's darkest colour (still bright enough that the
    # enhance pipeline's contrast boost won't clamp it to pure black)
    dark = np.array(p["hull"][0], dtype=np.float64)
    dark = np.maximum(dark, 24.0)  # survive (x-128)*1.22+128 without clamping
    base = np.tile(dark[None, None, :], (size, size, 1)).astype(np.float32)

    fill_arr = np.array(fill_img, dtype=np.float32)
    fa = fill_arr[:, :, 3:4] / 255.0
    out = base * (1 - fa) + fill_arr[:, :, :3] * fa

    ed_arr = np.array(edge_img, dtype=np.float32)
    ea = ed_arr[:, :, 3:4] / 255.0
    out = out * (1 - ea * 0.7) + ed_arr[:, :, :3] * (ea * 0.7)

    gl_arr = np.array(glow_img, dtype=np.float32) / 255.0
    glow_rgb = gl_arr[:, :, :3] * gl_arr[:, :, 3:4]
    bloom = np.stack([
        gaussian_filter(glow_rgb[:, :, c], sigma=9.0) for c in range(3)
    ], axis=-1)
    out = np.clip(out + bloom * 255 * 1.4, 0, 255)
    out = out * (1 - gl_arr[:, :, 3:4]) + gl_arr[:, :, :3] * 255 * gl_arr[:, :, 3:4]

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def build(preset_name=None):
    targets = [preset_name] if preset_name else list(PRESETS.keys())
    for name in targets:
        print(f"  Building shardform: {name}...")
        pattern = build_pattern(name)
        sc = SkinCanvas()
        install_robust_masks(sc)
        sc.set_image(pattern)
        sc.set_finish(0x70)
        sc.set_finish_role("hero", 0x50)
        sc.set_finish_role("neutral", 0x95)
        sc.save(name)
        print(f"    -> out/{name}.zip")


if __name__ == "__main__":
    build()
