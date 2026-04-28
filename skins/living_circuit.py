#!/usr/bin/env python3
"""
LIVING CIRCUIT -- Branching circuit traces on dark substrate.

Dark PCB-like surface with thin bright traces that branch and
connect organically, glowing at junction nodes.  Built from
a recursive L-system-inspired random walk that deposits traces
onto the texture, then blurred for glow.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
from skin_canvas import SkinCanvas

PRESETS = {
    "Living_Circuit": {
        "trace_color": (0, 255, 180),
        "glow_color": (0, 100, 80),
        "node_color": (180, 255, 230),
        "bg": (6, 10, 12),
        "substrate": (12, 18, 22),
        "n_seeds": 25,
        "max_steps": 300,
        "branch_prob": 0.06,
        "turn_prob": 0.15,
        "trace_width": 2,
        "finish": 0x55,
        "seed": 42,
    },
}


def build_pattern(preset_name, size=2048):
    p = PRESETS[preset_name]
    rng = np.random.default_rng(p["seed"])

    trace_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(trace_img)

    node_img = Image.new("L", (size, size), 0)
    ndraw = ImageDraw.Draw(node_img)

    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    step_size = 12
    tw = p["trace_width"]

    walkers = []
    for _ in range(p["n_seeds"]):
        x = rng.integers(50, size - 50)
        y = rng.integers(50, size - 50)
        d = rng.integers(0, 4)
        walkers.append((x, y, d, p["max_steps"]))
        ndraw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=255)

    while walkers:
        new_walkers = []
        for wx, wy, wd, steps_left in walkers:
            if steps_left <= 0:
                continue

            dx, dy = dirs[wd]
            nx = wx + dx * step_size
            ny = wy + dy * step_size

            if 0 <= nx < size and 0 <= ny < size:
                draw.line([(wx, wy), (nx, ny)], fill=200, width=tw)
                new_walkers.append((nx, ny, wd, steps_left - 1))

                if rng.random() < p["turn_prob"]:
                    new_dir = (wd + rng.choice([-1, 1])) % 4
                    new_walkers[-1] = (nx, ny, new_dir, steps_left - 1)
                    ndraw.ellipse([nx - 3, ny - 3, nx + 3, ny + 3], fill=180)

                if rng.random() < p["branch_prob"] and steps_left > 30:
                    branch_dir = (wd + rng.choice([-1, 1])) % 4
                    new_walkers.append((nx, ny, branch_dir, steps_left // 2))
                    ndraw.ellipse([nx - 5, ny - 5, nx + 5, ny + 5], fill=255)
            else:
                new_dir = (wd + rng.choice([-1, 1])) % 4
                new_walkers.append((wx, wy, new_dir, steps_left - 1))

        walkers = new_walkers

    trace_arr = np.array(trace_img, dtype=np.float64) / 255.0
    node_arr = np.array(node_img, dtype=np.float64) / 255.0

    glow = gaussian_filter(trace_arr, sigma=6.0)
    glow = np.clip(glow / (glow.max() + 1e-12), 0, 1)

    node_glow = gaussian_filter(node_arr, sigma=8.0)
    node_glow = np.clip(node_glow / (node_glow.max() + 1e-12), 0, 1)

    bg = np.array(p["bg"], dtype=np.float64)
    sub = np.array(p["substrate"], dtype=np.float64)
    gc = np.array(p["glow_color"], dtype=np.float64)
    tc = np.array(p["trace_color"], dtype=np.float64)
    nc = np.array(p["node_color"], dtype=np.float64)

    grid_noise = np.array(gaussian_filter(
        np.random.default_rng(p["seed"] + 999).random((size, size)),
        sigma=3), dtype=np.float64)
    grid_noise = (grid_noise - grid_noise.min()) / (grid_noise.max() - grid_noise.min() + 1e-12)

    rgb = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        base = bg[ch] + (sub[ch] - bg[ch]) * grid_noise * 0.5
        base = base * (1 - glow) + gc[ch] * glow
        base = base * (1 - trace_arr) + tc[ch] * trace_arr
        base = base * (1 - node_glow * 0.5) + nc[ch] * node_glow * 0.5
        base = base * (1 - node_arr) + nc[ch] * node_arr
        rgb[:, :, ch] = base

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def build(preset_name=None):
    targets = {preset_name: PRESETS[preset_name]} if preset_name else PRESETS
    for name in targets:
        print(f"  Building living circuit: {name}...")
        pattern = build_pattern(name)
        c = SkinCanvas()
        c.set_image(pattern)
        c.set_finish(PRESETS[name]["finish"])
        c.set_finish_role("neutral", "matte")
        c.save(name)
        print(f"    -> out/{name}.zip")


if __name__ == "__main__":
    build()
