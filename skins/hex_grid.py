#!/usr/bin/env python3
"""
HEX GRID -- Glowing hexagonal cell network.

A honeycomb lattice with dark cells and bright glowing edges.
Cell brightness varies via noise for organic depth.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from skin_canvas import SkinCanvas

PRESETS = {
    "Hex_Grid": {
        "edge_color": (0, 255, 200),
        "glow_color": (0, 100, 80),
        "bg": (4, 8, 10),
        "cell_fill": (10, 18, 22),
        "cell_size": 80.0,
        "edge_width": 3.5,
        "glow_width": 16.0,
        "finish": 0x6C,
        "seed": 42,
    },
}


def _hex_distance(xx, yy, cell_size):
    """Distance from each pixel to the nearest hexagonal cell edge."""
    s = cell_size
    h = s * np.sqrt(3.0) / 2.0

    col = np.floor(xx / (1.5 * s)).astype(int)
    row_offset = (col % 2) * h
    row = np.floor((yy - row_offset) / (2.0 * h)).astype(int)

    best_dist = np.full_like(xx, 1e9)
    for dc in range(-1, 3):
        for dr in range(-1, 3):
            c = col + dc
            r = row + dr
            ro = (c % 2) * h
            cx = c * 1.5 * s
            cy = r * 2.0 * h + ro

            dx = np.abs(xx - cx)
            dy = np.abs(yy - cy)

            d_hex = np.maximum(dx / s, (dx / s) * 0.5 + (dy / h) * 0.5)
            d_edge = np.abs(d_hex - 1.0) * s
            best_dist = np.minimum(best_dist, d_edge)

    return best_dist


def build_pattern(preset_name, size=2048):
    p = PRESETS[preset_name]
    cs = p["cell_size"]
    ew = p["edge_width"]
    gw = p["glow_width"]

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dist = _hex_distance(xx, yy, cs)

    core = np.clip(1.0 - dist / ew, 0.0, 1.0)
    glow = np.clip(1.0 - dist / gw, 0.0, 1.0) ** 2

    rng = np.random.default_rng(p["seed"])
    cell_noise = gaussian_filter(rng.random((size, size)), sigma=cs * 0.6, mode="wrap")
    cell_noise = (cell_noise - cell_noise.min()) / (cell_noise.max() - cell_noise.min() + 1e-12)
    cell_var = 0.6 + 0.4 * cell_noise

    bg = np.array(p["bg"], dtype=np.float64)
    cf = np.array(p["cell_fill"], dtype=np.float64)
    gc = np.array(p["glow_color"], dtype=np.float64)
    ec = np.array(p["edge_color"], dtype=np.float64)

    rgb = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        base = bg[ch] + (cf[ch] - bg[ch]) * cell_var
        base = base * (1.0 - glow) + gc[ch] * glow
        base = base * (1.0 - core) + ec[ch] * core
        rgb[:, :, ch] = base

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def build(preset_name=None):
    targets = {preset_name: PRESETS[preset_name]} if preset_name else PRESETS
    for name in targets:
        print(f"  Building hex grid: {name}...")
        pattern = build_pattern(name)
        c = SkinCanvas()
        c.set_image(pattern)
        c.set_finish(PRESETS[name]["finish"])
        c.set_finish_role("neutral", "matte")
        c.save(name)
        print(f"    -> out/{name}.zip")


if __name__ == "__main__":
    build()
