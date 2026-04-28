#!/usr/bin/env python3
"""
VISCERA -- Anatomical horror gore skin.  Dark translucent red-black
base as if looking through skin, dense vein/artery network, ribcage
painted on the sides, skull on wing, raven-wing ProjShad.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
from skin_canvas import SkinCanvas
from skins.gore_utils import (
    make_raven_projshad, draw_veins, draw_skull,
)


def _draw_ribcage(size):
    """Draw a stylized ribcage overlay."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = size // 2
    bone_color = (190, 180, 160, 140)
    outline_color = (60, 10, 10, 120)

    spine_top = int(size * 0.15)
    spine_bot = int(size * 0.75)
    spine_w = int(size * 0.015)
    draw.rectangle([cx - spine_w, spine_top, cx + spine_w, spine_bot],
                   fill=bone_color, outline=outline_color, width=1)

    n_ribs = 8
    for i in range(n_ribs):
        ry = spine_top + int((spine_bot - spine_top) * (i + 0.5) / n_ribs)
        rib_len = int(size * (0.18 + 0.08 * np.sin(np.pi * i / n_ribs)))
        curve = int(size * 0.04 * (1 + i / n_ribs))

        for sx in [-1, 1]:
            pts = []
            for t in np.linspace(0, 1, 20):
                x = cx + sx * rib_len * t
                y = ry + curve * np.sin(np.pi * t) * (1 - 0.3 * t)
                pts.append((int(x), int(y)))
            draw.line(pts, fill=bone_color, width=max(2, size // 400))

    return img


def build():
    size = 2048
    print("  Building Viscera...")

    base = np.zeros((size, size, 3), dtype=np.float64)
    rng = np.random.default_rng(333)

    tissue = gaussian_filter(rng.random((size, size)), sigma=30)
    tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min() + 1e-12)

    base[:, :, 0] = 35 + 30 * tissue
    base[:, :, 1] = 5 + 8 * tissue
    base[:, :, 2] = 6 + 6 * tissue

    depth = gaussian_filter(rng.random((size, size)), sigma=80)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-12)
    for ch in range(3):
        base[:, :, ch] *= (0.5 + 0.5 * depth)

    membrane = gaussian_filter(rng.random((size, size)), sigma=12)
    membrane = (membrane - membrane.min()) / (membrane.max() - membrane.min() + 1e-12)
    base[:, :, 0] += membrane * 15

    base_rgb = np.clip(base, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(
        np.dstack([base_rgb, np.full((size, size), 255, dtype=np.uint8)]),
        "RGBA"
    )

    arteries = draw_veins(size, n_seeds=35, color=(180, 20, 10),
                          max_depth=7, seed=333)
    capillaries = draw_veins(size, n_seeds=60, color=(120, 40, 30),
                             max_depth=5, seed=444)

    art_arr = np.array(arteries, dtype=np.float64)
    art_glow = np.zeros_like(art_arr)
    for ch in range(3):
        art_glow[:, :, ch] = gaussian_filter(art_arr[:, :, ch], sigma=5)
    art_glow[:, :, 3] = gaussian_filter(art_arr[:, :, 3], sigma=5)
    glow_img = Image.fromarray(np.clip(art_glow, 0, 255).astype(np.uint8), "RGBA")

    base_img = Image.alpha_composite(base_img, glow_img)
    base_img = Image.alpha_composite(base_img, capillaries)
    base_img = Image.alpha_composite(base_img, arteries)

    ribs = _draw_ribcage(size)
    base_img = Image.alpha_composite(base_img, ribs)

    c = SkinCanvas()
    c.set_image(base_img)

    skull_img = draw_skull(180, color=(210, 200, 180), outline=(60, 15, 10))
    for wing_id in [15, 16]:
        c.paste(skull_img, island=wing_id, scale=0.6, clip_to_island=True)

    projshad = make_raven_projshad(512)
    c.save("Viscera", projshad=projshad)
    print("    -> out/Viscera.zip")


if __name__ == "__main__":
    build()
