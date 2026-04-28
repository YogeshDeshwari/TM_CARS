"""
PBR material system for Trackmania 2020 skins.

TM2020 uses a physically-based rendering pipeline with separate texture maps:
    Skin_R:     R channel = Roughness,  G channel = Metalness
    Skin_CoatR: Grayscale = Clearcoat (varnish / glitter paint)
    Skin_DirtMask: Grayscale = where dirt accumulates

This module provides Material presets and functions to generate the PBR maps
from a high-level description (e.g., "glossy metallic red paint").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class PBRMaterial:
    """
    Per-pixel PBR material descriptor.

    roughness : 0 = mirror smooth, 255 = fully rough/matte
    metalness : 0 = dielectric (paint/plastic), 255 = full metal
    clearcoat : 0 = no clearcoat, 255 = thick varnish layer
    dirt      : 0 = clean, 255 = maximum dirt accumulation
    """
    roughness: int = 80
    metalness: int = 0
    clearcoat: int = 200
    dirt: int = 128

    def roughness_metalness_image(self, size: int) -> Image.Image:
        """Generate a flat Skin_R texture: R=roughness, G=metalness, B=0."""
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[:, :, 0] = self.roughness
        arr[:, :, 1] = self.metalness
        arr[:, :, 3] = 255
        return Image.fromarray(arr, "RGBA")

    def clearcoat_image(self, size: int) -> Image.Image:
        """Generate a flat Skin_CoatR texture (grayscale)."""
        arr = np.full((size, size, 4), 255, dtype=np.uint8)
        arr[:, :, 0] = self.clearcoat
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        return Image.fromarray(arr, "RGBA")

    def dirt_mask_image(self, size: int) -> Image.Image:
        """Generate a flat Skin_DirtMask texture (grayscale)."""
        arr = np.full((size, size, 4), 255, dtype=np.uint8)
        arr[:, :, 0] = self.dirt
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        return Image.fromarray(arr, "RGBA")


# -- Named material presets ------------------------------------------------

MATTE_PAINT = PBRMaterial(roughness=200, metalness=0, clearcoat=30, dirt=160)
SATIN_PAINT = PBRMaterial(roughness=120, metalness=0, clearcoat=120, dirt=140)
GLOSS_PAINT = PBRMaterial(roughness=40, metalness=0, clearcoat=230, dirt=120)
METALLIC_PAINT = PBRMaterial(roughness=60, metalness=180, clearcoat=220, dirt=100)
CHROME = PBRMaterial(roughness=10, metalness=255, clearcoat=255, dirt=60)
BRUSHED_METAL = PBRMaterial(roughness=140, metalness=220, clearcoat=80, dirt=100)
CARBON_FIBER = PBRMaterial(roughness=100, metalness=30, clearcoat=180, dirt=80)
RUBBER = PBRMaterial(roughness=230, metalness=0, clearcoat=10, dirt=200)
PLASTIC = PBRMaterial(roughness=160, metalness=0, clearcoat=60, dirt=180)

MATERIAL_PRESETS: Dict[str, PBRMaterial] = {
    "matte": MATTE_PAINT,
    "satin": SATIN_PAINT,
    "gloss": GLOSS_PAINT,
    "metallic": METALLIC_PAINT,
    "chrome": CHROME,
    "brushed_metal": BRUSHED_METAL,
    "carbon": CARBON_FIBER,
    "rubber": RUBBER,
    "plastic": PLASTIC,
}


# -- PBR map generation helpers -------------------------------------------

def generate_roughness_metalness(
    size: int,
    base_material: PBRMaterial,
    *,
    roughness_noise: int = 0,
    metalness_noise: int = 0,
    seed: int = 42,
) -> Image.Image:
    """
    Generate a Skin_R texture with optional per-pixel noise.

    Returns RGBA image where R=roughness, G=metalness, B=0, A=255.
    """
    rng = np.random.RandomState(seed)
    arr = np.zeros((size, size, 4), dtype=np.uint8)

    r = np.full((size, size), base_material.roughness, dtype=np.float32)
    m = np.full((size, size), base_material.metalness, dtype=np.float32)

    if roughness_noise > 0:
        r += rng.uniform(-roughness_noise, roughness_noise, (size, size))
    if metalness_noise > 0:
        m += rng.uniform(-metalness_noise, metalness_noise, (size, size))

    arr[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(m, 0, 255).astype(np.uint8)
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def generate_clearcoat(
    size: int,
    base_value: int = 200,
    *,
    edge_falloff: bool = False,
) -> Image.Image:
    """
    Generate a Skin_CoatR texture. Optionally reduce clearcoat at UV edges
    to simulate wear.
    """
    arr = np.full((size, size, 4), 255, dtype=np.uint8)
    arr[:, :, 0] = base_value
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0

    if edge_falloff:
        falloff = np.ones((size, size), dtype=np.float32)
        margin = size // 16
        for i in range(margin):
            t = i / margin
            falloff[i, :] *= t
            falloff[-(i + 1), :] *= t
            falloff[:, i] *= t
            falloff[:, -(i + 1)] *= t
        arr[:, :, 0] = (arr[:, :, 0].astype(np.float32) * falloff).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


def generate_dirt_mask(
    size: int,
    base_value: int = 128,
    *,
    noise_amount: int = 40,
    seed: int = 99,
) -> Image.Image:
    """
    Generate a Skin_DirtMask with organic noise for realistic dirt variation.
    """
    rng = np.random.RandomState(seed)
    base = np.full((size, size), base_value, dtype=np.float32)
    base += rng.uniform(-noise_amount, noise_amount, (size, size))
    base = np.clip(base, 0, 255)

    arr = np.full((size, size, 4), 255, dtype=np.uint8)
    arr[:, :, 0] = base.astype(np.uint8)
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    img = Image.fromarray(arr, "RGBA")
    return img.filter(ImageFilter.GaussianBlur(radius=size // 128 or 1))


def generate_flat_normal(size: int) -> Image.Image:
    """Generate a flat (no-bump) normal map in OpenGL Y+ convention."""
    arr = np.full((size, size, 4), 255, dtype=np.uint8)
    arr[:, :, 0] = 128  # X = 0 (centered)
    arr[:, :, 1] = 128  # Y = 0 (centered, OpenGL Y+)
    arr[:, :, 2] = 255  # Z = 1 (pointing straight out)
    return Image.fromarray(arr, "RGBA")


def composite_roughness_metalness(
    base: Image.Image,
    regions: Dict[str, Tuple[Image.Image, PBRMaterial]],
) -> Image.Image:
    """
    Build a Skin_R map with different materials per masked region.

    regions: dict of {name: (mask_image_L, material)}
    The mask is L-mode (0..255), white = full material application.
    """
    size = base.size[0]
    result = np.array(base.convert("RGBA"), dtype=np.float32)

    for _name, (mask, mat) in regions.items():
        m = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        result[:, :, 0] = result[:, :, 0] * (1 - m) + mat.roughness * m
        result[:, :, 1] = result[:, :, 1] * (1 - m) + mat.metalness * m

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGBA")
