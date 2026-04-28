#!/usr/bin/env python3
"""
STAR TRAILS -- Long-exposure night sky photography.

Concentric circular arcs of starlight sweeping across a deep
navy-black sky, as if the car captured hours of Earth's rotation.
Each arc is a thin luminous streak varying in brightness and hue,
centered around the celestial pole.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
from skin_canvas import SkinCanvas

PRESETS = {
    "Star_Trails": {
        "sky_dark": (3, 5, 18),
        "sky_mid": (8, 12, 35),
        "center": (0.3, 0.35),
        "n_stars": 600,
        "arc_width": 2,
        "min_arc_deg": 15,
        "max_arc_deg": 120,
        "hue_spread": 0.12,
        "base_hue": 0.6,
        "bright_range": (0.4, 1.0),
        "seed": 42,
    },
}


def _hsv_pixel(h, s, v):
    """Single HSV to RGB."""
    h6 = (h % 1.0) * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def build_pattern(preset_name, size=2048):
    p = PRESETS[preset_name]
    rng = np.random.default_rng(p["seed"])

    sky = np.zeros((size, size, 3), dtype=np.float64)
    sd = np.array(p["sky_dark"], dtype=np.float64)
    sm = np.array(p["sky_mid"], dtype=np.float64)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = p["center"][0] * size
    cy = p["center"][1] * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.7)
    dist = np.clip(dist, 0, 1)

    for ch in range(3):
        sky[:, :, ch] = sd[ch] + (sm[ch] - sd[ch]) * (1 - dist) * 0.3

    sky_noise = rng.random((size, size)) * 4.0
    for ch in range(3):
        sky[:, :, ch] += sky_noise

    trail_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(trail_layer)

    for _ in range(p["n_stars"]):
        r = rng.uniform(20, size * 0.85)
        start_angle = rng.uniform(0, 360)
        arc_len = rng.uniform(p["min_arc_deg"], p["max_arc_deg"])

        brightness = rng.uniform(*p["bright_range"])
        hue = p["base_hue"] + rng.uniform(-p["hue_spread"], p["hue_spread"])
        sat = rng.uniform(0.15, 0.6)

        rv, gv, bv = _hsv_pixel(hue, sat, brightness)
        color = (int(rv * 255), int(gv * 255), int(bv * 255), int(brightness * 220 + 35))

        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, start_angle, start_angle + arc_len,
                 fill=color, width=p["arc_width"])

    bright_stars = 40
    for _ in range(bright_stars):
        r = rng.uniform(30, size * 0.8)
        start_angle = rng.uniform(0, 360)
        arc_len = rng.uniform(p["max_arc_deg"] * 0.7, p["max_arc_deg"] * 1.3)
        hue = p["base_hue"] + rng.uniform(-0.05, 0.05)
        rv, gv, bv = _hsv_pixel(hue, 0.1, 1.0)
        color = (int(rv * 255), int(gv * 255), int(bv * 255), 255)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, start_angle, start_angle + arc_len,
                 fill=color, width=p["arc_width"] + 1)

    trail_arr = np.array(trail_layer, dtype=np.float64)
    trail_rgb = trail_arr[:, :, :3]
    trail_a = trail_arr[:, :, 3] / 255.0

    glow_r = gaussian_filter(trail_rgb[:, :, 0] * trail_a, sigma=3.0)
    glow_g = gaussian_filter(trail_rgb[:, :, 1] * trail_a, sigma=3.0)
    glow_b = gaussian_filter(trail_rgb[:, :, 2] * trail_a, sigma=3.0)

    for ch, glow in enumerate([glow_r, glow_g, glow_b]):
        sky[:, :, ch] += glow * 0.4

    for ch in range(3):
        sky[:, :, ch] = sky[:, :, ch] * (1 - trail_a) + trail_rgb[:, :, ch] * trail_a

    static_stars = 300
    sx = rng.integers(0, size, static_stars)
    sy = rng.integers(0, size, static_stars)
    sb = rng.uniform(0.3, 1.0, static_stars)
    for x, y, b in zip(sx, sy, sb):
        for ch in range(3):
            sky[y, x, ch] = min(255, sky[y, x, ch] + b * 200)

    rgb = np.clip(sky, 0, 255).astype(np.uint8)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def build(preset_name=None):
    targets = {preset_name: PRESETS[preset_name]} if preset_name else PRESETS
    for name in targets:
        print(f"  Building star trails: {name}...")
        pattern = build_pattern(name)
        c = SkinCanvas()
        c.set_image(pattern)
        c.save(name)
        print(f"    -> out/{name}.zip")


if __name__ == "__main__":
    build()
