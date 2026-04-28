#!/usr/bin/env python3
"""
CARRION -- Grungy gore skin.  Dried blood base, fresh splatters,
claw scratches, skull on the rear wing, raven-wing ProjShad.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageChops
from skin_canvas import SkinCanvas
from skins.gore_utils import (
    make_raven_projshad, draw_blood_splatters,
    draw_claw_scratches, draw_skull,
)


def build():
    size = 2048
    print("  Building Carrion...")

    base = np.zeros((size, size, 3), dtype=np.float64)
    yy = np.linspace(0, 1, size).reshape(size, 1)
    base[:, :, 0] = 28 + 22 * yy
    base[:, :, 1] = 4 + 4 * yy
    base[:, :, 2] = 4 + 3 * yy

    rng = np.random.default_rng(42)
    grime = gaussian_filter(rng.random((size, size)), sigma=60)
    grime = (grime - grime.min()) / (grime.max() - grime.min() + 1e-12)
    for ch in range(3):
        base[:, :, ch] *= (0.7 + 0.3 * grime)

    dark_pools = gaussian_filter(rng.random((size, size)), sigma=120)
    dark_pools = (dark_pools - dark_pools.min()) / (dark_pools.max() - dark_pools.min() + 1e-12)
    dark_mask = np.clip((dark_pools - 0.5) * 3, 0, 1)
    for ch in range(3):
        base[:, :, ch] *= (1 - dark_mask * 0.5)

    base_rgb = np.clip(base, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(
        np.dstack([base_rgb, np.full((size, size), 255, dtype=np.uint8)]),
        "RGBA"
    )

    splatters = draw_blood_splatters(size, n_splatters=80,
                                     color=(160, 10, 5), seed=42)
    fresh_splatters = draw_blood_splatters(size, n_splatters=25,
                                           color=(200, 20, 10), seed=99)
    scratches = draw_claw_scratches(size, n_scratches=12,
                                     color=(200, 190, 170), seed=42)

    base_img = Image.alpha_composite(base_img, splatters)
    base_img = Image.alpha_composite(base_img, fresh_splatters)
    base_img = Image.alpha_composite(base_img, scratches)

    c = SkinCanvas()
    c.set_image(base_img)

    skull_img = draw_skull(180, color=(200, 190, 170), outline=(50, 5, 5))
    for wing_id in [15, 16]:
        c.paste(skull_img, island=wing_id, scale=0.6, clip_to_island=True)

    projshad = make_raven_projshad(512)
    c.save("Carrion", projshad=projshad)
    print("    -> out/Carrion.zip")


if __name__ == "__main__":
    build()
