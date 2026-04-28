"""
TM2020 skin preset registry.

Each preset is a function that returns a SkinSpec. Presets are registered
with @register and can be listed/fetched by name.

Pattern generators are imported from the parent TMNF skin_utils module.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

# Add parent dir so we can import TMNF utilities
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from skin_utils import (
    hex_to_rgb,
    create_gradient,
    generate_warp_grid,
    generate_aurora,
    generate_acid_neon,
    generate_bismuth_crystal,
    generate_voronoi_shatter,
    generate_damascus_steel,
    generate_circuit_traces,
    generate_halftone,
    generate_racing_stripes,
    generate_hex_tessellation,
    generate_carbon_v2,
    generate_metallic_flake,
)

from .engine import SkinSpec
from .materials import (
    PBRMaterial,
    GLOSS_PAINT,
    MATTE_PAINT,
    SATIN_PAINT,
    METALLIC_PAINT,
    CHROME,
    BRUSHED_METAL,
    CARBON_FIBER,
    RUBBER,
    PLASTIC,
)


# -- Registry --------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], SkinSpec]] = {}


def register(name: str, description: str = ""):
    """Decorator to register a preset function."""
    def wrapper(fn: Callable[[], SkinSpec]) -> Callable[[], SkinSpec]:
        _REGISTRY[name] = fn
        fn._preset_name = name
        fn._preset_desc = description
        return fn
    return wrapper


def get_preset(name: str) -> SkinSpec:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_presets() -> List[Tuple[str, str]]:
    return [
        (name, getattr(fn, "_preset_desc", ""))
        for name, fn in sorted(_REGISTRY.items())
    ]


# -- Helper patterns (TM2020-specific) ------------------------------------

def _diagonal_stripes(
    size: int,
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    stripe_width: int = 80,
) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x + y) // stripe_width) % 2 == 0:
                arr[y, x] = (*color1, 255)
            else:
                arr[y, x] = (*color2, 255)
    return Image.fromarray(arr, "RGBA")


def _split_livery(
    size: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
    split_ratio: float = 0.45,
    feather: int = 0,
) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    split_y = int(size * split_ratio)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, split_y], fill=top_color + (255,))
    draw.rectangle([0, split_y, size, size], fill=bottom_color + (255,))
    if feather > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=feather))
    return img


def _gradient_sweep(
    size: int,
    colors: List[Tuple[int, int, int]],
    angle_deg: float = 0.0,
) -> Image.Image:
    """Multi-stop gradient at an arbitrary angle."""
    n = len(colors)
    if n < 2:
        return Image.new("RGBA", (size, size), colors[0] + (255,))

    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255

    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = size / 2, size / 2

    ys, xs = np.mgrid[0:size, 0:size]
    proj = (xs - cx) * cos_a + (ys - cy) * sin_a
    proj_min, proj_max = proj.min(), proj.max()
    if proj_max > proj_min:
        t = (proj - proj_min) / (proj_max - proj_min)
    else:
        t = np.zeros_like(proj)

    t_scaled = t * (n - 1)
    for c in range(3):
        channel = np.zeros((size, size), dtype=np.float32)
        for i in range(n - 1):
            local_t = np.clip(t_scaled - i, 0, 1)
            mask = (t_scaled >= i) & (t_scaled < i + 1)
            if i == n - 2:
                mask = t_scaled >= i
            c1 = colors[i][c]
            c2 = colors[i + 1][c]
            channel += mask * (c1 * (1 - local_t) + c2 * local_t)
        arr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


# =========================================================================
# PRESETS
# =========================================================================

@register("midnight_chrome", "Deep navy with chrome accent stripes")
def _midnight_chrome() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        base = Image.new("RGBA", (size, size), (10, 12, 30, 255))
        stripes = generate_racing_stripes(
            size, (180, 200, 220), count=2, width=60
        )
        return Image.alpha_composite(base, stripes)

    return SkinSpec(
        name="TM2020_Midnight_Chrome",
        skin_color_fn=skin_fn,
        skin_material=PBRMaterial(roughness=30, metalness=40, clearcoat=240, dirt=100),
        glass_tint=(5, 5, 15),
        seed=100,
    )


@register("acid_warp", "Psychedelic warp grid with glossy clearcoat")
def _acid_warp() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_warp_grid(size, seed=42)

    return SkinSpec(
        name="TM2020_Acid_Warp",
        skin_color_fn=skin_fn,
        skin_material=GLOSS_PAINT,
        clearcoat_edge_falloff=True,
        seed=42,
    )


@register("neon_aurora", "Aurora borealis effect with metallic finish")
def _neon_aurora() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_aurora(
            size,
            palette=[(0, 255, 128), (0, 128, 255), (180, 0, 255)],
            seed=77,
        )

    return SkinSpec(
        name="TM2020_Neon_Aurora",
        skin_color_fn=skin_fn,
        skin_material=METALLIC_PAINT,
        glass_tint=(0, 30, 20),
        seed=77,
    )


@register("matte_stealth", "Flat black stealth with subtle hex pattern")
def _matte_stealth() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        base = Image.new("RGBA", (size, size), (20, 20, 22, 255))
        hex_pat = generate_hex_tessellation(size, (35, 35, 38))
        return Image.alpha_composite(base, hex_pat)

    return SkinSpec(
        name="TM2020_Matte_Stealth",
        skin_color_fn=skin_fn,
        skin_material=MATTE_PAINT,
        dirt_amount=200,
        glass_tint=(10, 10, 10),
        seed=200,
    )


@register("crimson_metallic", "Deep red metallic paint, classic sports car")
def _crimson_metallic() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        base = _gradient_sweep(
            size,
            [(120, 0, 10), (180, 10, 20), (140, 0, 15)],
            angle_deg=30.0,
        )
        flake = generate_metallic_flake(size, (200, 30, 40))
        flake.putalpha(Image.new("L", (size, size), 40))
        return Image.alpha_composite(base, flake)

    return SkinSpec(
        name="TM2020_Crimson_Metallic",
        skin_color_fn=skin_fn,
        skin_material=METALLIC_PAINT,
        clearcoat_edge_falloff=True,
        seed=300,
    )


@register("bismuth_iridescent", "Bismuth crystal pattern with chrome finish")
def _bismuth() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_bismuth_crystal(size, seed=55)

    return SkinSpec(
        name="TM2020_Bismuth_Iridescent",
        skin_color_fn=skin_fn,
        skin_material=PBRMaterial(
            roughness=20, metalness=200, clearcoat=255, dirt=60
        ),
        glass_tint=(15, 10, 25),
        seed=55,
    )


@register("carbon_racing", "Carbon fiber body with neon accent stripe")
def _carbon_racing() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        carbon = generate_carbon_v2(size, base_color=(25, 25, 28))
        stripe = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(stripe)
        y_center = size // 2
        stripe_h = size // 16
        draw.rectangle(
            [0, y_center - stripe_h, size, y_center + stripe_h],
            fill=(0, 255, 180, 220),
        )
        return Image.alpha_composite(carbon, stripe)

    return SkinSpec(
        name="TM2020_Carbon_Racing",
        skin_color_fn=skin_fn,
        skin_material=CARBON_FIBER,
        glass_tint=(0, 20, 15),
        seed=400,
    )


@register("damascus_blade", "Damascus steel pattern, brushed metal finish")
def _damascus() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_damascus_steel(size, seed=88)

    return SkinSpec(
        name="TM2020_Damascus_Blade",
        skin_color_fn=skin_fn,
        skin_material=BRUSHED_METAL,
        glass_tint=(10, 10, 8),
        seed=88,
    )


@register("voronoi_shatter", "Shattered glass / voronoi pattern")
def _voronoi() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_voronoi_shatter(
            size,
            palette=[(0, 200, 255), (0, 100, 200), (10, 20, 40)],
            seed=66,
        )

    return SkinSpec(
        name="TM2020_Voronoi_Shatter",
        skin_color_fn=skin_fn,
        skin_material=GLOSS_PAINT,
        seed=66,
    )


@register("circuit_neon", "Circuit board traces with neon glow")
def _circuit_neon() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        base = Image.new("RGBA", (size, size), (5, 5, 10, 255))
        traces = generate_circuit_traces(size, (0, 255, 120), density="dense")
        return Image.alpha_composite(base, traces)

    return SkinSpec(
        name="TM2020_Circuit_Neon",
        skin_color_fn=skin_fn,
        skin_material=PBRMaterial(
            roughness=50, metalness=60, clearcoat=200, dirt=80
        ),
        glass_tint=(0, 30, 10),
        seed=500,
    )


@register("sunset_gradient", "Warm sunset gradient, satin finish")
def _sunset() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return _gradient_sweep(
            size,
            [(255, 60, 20), (255, 140, 0), (255, 200, 50), (200, 80, 180)],
            angle_deg=135.0,
        )

    return SkinSpec(
        name="TM2020_Sunset_Gradient",
        skin_color_fn=skin_fn,
        skin_material=SATIN_PAINT,
        glass_tint=(30, 15, 10),
        seed=600,
    )


@register("arctic_white", "Clean white with blue tinted glass")
def _arctic() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return Image.new("RGBA", (size, size), (245, 245, 248, 255))

    return SkinSpec(
        name="TM2020_Arctic_White",
        skin_color_fn=skin_fn,
        skin_material=PBRMaterial(
            roughness=35, metalness=10, clearcoat=250, dirt=100
        ),
        glass_tint=(10, 15, 30),
        seed=700,
    )


@register("acid_neon_blast", "Acid neon pattern, extreme vibrancy")
def _acid_neon_blast() -> SkinSpec:
    def skin_fn(size: int) -> Image.Image:
        return generate_acid_neon(size, seed=33)

    return SkinSpec(
        name="TM2020_Acid_Neon_Blast",
        skin_color_fn=skin_fn,
        skin_material=GLOSS_PAINT,
        glass_tint=(10, 0, 20),
        seed=33,
    )
