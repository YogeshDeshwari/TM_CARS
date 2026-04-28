#!/usr/bin/env python3
"""
ABYSSAL -- Deep underwater with caustic light.

The car appears submerged in deep ocean.  Dark teal-blue base with
bright dancing caustic light patterns on top, subtle depth fog
gradient from front to back, and scattered bright particle specks
simulating suspended matter in the water column.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from skin_canvas import SkinCanvas

PRESETS = {
    "Abyssal": {
        "deep_color": (2, 12, 25),
        "mid_color": (5, 40, 65),
        "caustic_color": (120, 230, 220),
        "particle_color": (180, 255, 245),
        "base_scale": 45.0,
        "caustic_octaves": 5,
        "n_particles": 800,
        "finish": 0x60,
        "seed": 42,
    },
}


def _smooth_noise(size, scale, seed=42):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.random((size, size)), sigma=scale, mode="wrap")


def build_pattern(preset_name, size=2048):
    p = PRESETS[preset_name]
    seed = p["seed"]
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    depth_grad = (yy / size) * 0.3 + 0.7

    combined = np.ones((size, size), dtype=np.float64)
    freq = 1.0
    for i in range(p["caustic_octaves"]):
        n = _smooth_noise(size, p["base_scale"] / freq, seed + i * 137)
        n = (n - n.min()) / (n.max() - n.min() + 1e-12)
        ridge = 1.0 - np.abs(2.0 * n - 1.0)
        combined *= 0.25 + 0.75 * ridge
        freq *= 1.8

    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
    combined = np.power(combined, 2.0)
    combined *= depth_grad

    deep = np.array(p["deep_color"], dtype=np.float64)
    mid = np.array(p["mid_color"], dtype=np.float64)
    caus = np.array(p["caustic_color"], dtype=np.float64)

    depth_t = np.clip((yy / size) * 0.5 + 0.2, 0, 1)
    base = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        base[:, :, ch] = deep[ch] * (1 - depth_t) + mid[ch] * depth_t

    rgb = np.zeros((size, size, 3), dtype=np.float64)
    for ch in range(3):
        rgb[:, :, ch] = base[:, :, ch] * (1 - combined) + caus[ch] * combined

    pc = np.array(p["particle_color"], dtype=np.float64)
    px = rng.integers(0, size, p["n_particles"])
    py = rng.integers(0, size, p["n_particles"])
    pr = rng.uniform(1.0, 3.5, p["n_particles"])
    pb = rng.uniform(0.3, 1.0, p["n_particles"])

    for x, y, r, bright in zip(px, py, pr, pb):
        y0 = max(0, int(y - r * 2))
        y1 = min(size, int(y + r * 2) + 1)
        x0 = max(0, int(x - r * 2))
        x1 = min(size, int(x + r * 2) + 1)
        dy = np.arange(y0, y1)[:, None] - y
        dx = np.arange(x0, x1)[None, :] - x
        d = np.sqrt(dy ** 2 + dx ** 2)
        falloff = np.clip(1.0 - d / (r * 2), 0, 1) ** 2 * bright
        for ch in range(3):
            rgb[y0:y1, x0:x1, ch] += pc[ch] * falloff

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def build(preset_name=None):
    targets = {preset_name: PRESETS[preset_name]} if preset_name else PRESETS
    for name in targets:
        print(f"  Building abyssal: {name}...")
        pattern = build_pattern(name)
        c = SkinCanvas()
        c.set_image(pattern)
        c.set_finish(PRESETS[name]["finish"])
        c.set_finish_role("neutral", "matte")
        c.save(name)
        print(f"    -> out/{name}.zip")


if __name__ == "__main__":
    build()
