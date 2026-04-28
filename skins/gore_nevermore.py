#!/usr/bin/env python3
"""
NEVERMORE -- Gothic elegance gore skin.  Matte black with dark
purple-blue raven feather iridescence, arterial red veins creeping
across the surface, blood drips from edges, skull on wing,
raven-wing ProjShad.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageChops
from skin_canvas import SkinCanvas
from skins.gore_utils import (
    make_raven_projshad, draw_blood_splatters,
    draw_veins, draw_skull,
)


def build():
    size = 2048
    print("  Building Nevermore...")

    base = np.zeros((size, size, 3), dtype=np.float64)
    rng = np.random.default_rng(77)

    n1 = gaussian_filter(rng.random((size, size)), sigma=80)
    n1 = (n1 - n1.min()) / (n1.max() - n1.min() + 1e-12)
    n2 = gaussian_filter(rng.random((size, size)), sigma=50)
    n2 = (n2 - n2.min()) / (n2.max() - n2.min() + 1e-12)

    base[:, :, 0] = 8 + 12 * n1
    base[:, :, 1] = 5 + 6 * n2
    base[:, :, 2] = 14 + 20 * n1

    irid = gaussian_filter(rng.random((size, size)), sigma=40)
    irid = (irid - irid.min()) / (irid.max() - irid.min() + 1e-12)
    base[:, :, 0] += irid * 8
    base[:, :, 1] += irid * 3
    base[:, :, 2] += irid * 18

    feather_noise = gaussian_filter(rng.random((size, size)), sigma=6)
    feather_noise = (feather_noise - feather_noise.min()) / (feather_noise.max() - feather_noise.min() + 1e-12)
    for ch in range(3):
        base[:, :, ch] *= (0.85 + 0.15 * feather_noise)

    base_rgb = np.clip(base, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(
        np.dstack([base_rgb, np.full((size, size), 255, dtype=np.uint8)]),
        "RGBA"
    )

    veins = draw_veins(size, n_seeds=45, color=(160, 15, 10),
                       max_depth=7, seed=77)
    vein_arr = np.array(veins, dtype=np.float64)
    vein_glow = np.zeros_like(vein_arr)
    for ch in range(3):
        vein_glow[:, :, ch] = gaussian_filter(vein_arr[:, :, ch], sigma=4)
    vein_glow[:, :, 3] = gaussian_filter(vein_arr[:, :, 3], sigma=4)
    glow_img = Image.fromarray(np.clip(vein_glow, 0, 255).astype(np.uint8), "RGBA")

    base_img = Image.alpha_composite(base_img, glow_img)
    base_img = Image.alpha_composite(base_img, veins)

    drips = draw_blood_splatters(size, n_splatters=30,
                                  color=(130, 8, 5), seed=177)
    base_img = Image.alpha_composite(base_img, drips)

    c = SkinCanvas()
    c.set_image(base_img)

    skull_img = draw_skull(180, color=(180, 170, 155), outline=(30, 0, 10))
    for wing_id in [15, 16]:
        c.paste(skull_img, island=wing_id, scale=0.6, clip_to_island=True)

    projshad = make_raven_projshad(512)
    c.save("Nevermore", projshad=projshad)
    print("    -> out/Nevermore.zip")


if __name__ == "__main__":
    build()
