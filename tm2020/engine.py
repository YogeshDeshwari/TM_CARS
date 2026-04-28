"""
TM2020 skin compositor engine.

Takes a SkinSpec (preset definition) and produces a complete skin zip
containing all required DDS textures + Icon.tga, ready to drop into
Documents/Trackmania/Skins/Models/CarSport/.

Texture pipeline:
    1. Generate Skin_B (basecolor) from pattern layers
    2. Generate Skin_R (roughness/metalness) from material spec
    3. Generate Skin_CoatR (clearcoat) from material spec
    4. Generate Skin_DirtMask from material spec
    5. Optionally generate Details/Wheels/Glass overrides
    6. Render Icon.tga from a scaled-down preview
    7. Pack everything into a zip
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .dds import BCFormat, build_dds_bytes, save_dds
from .materials import (
    PBRMaterial,
    GLOSS_PAINT,
    RUBBER,
    PLASTIC,
    generate_roughness_metalness,
    generate_clearcoat,
    generate_dirt_mask,
    generate_flat_normal,
)

SKIN_SIZE = 2048
DETAIL_SIZE = 2048
WHEEL_SIZE = 1024
GLASS_SIZE = 512
ICON_SIZE = 64


@dataclass
class SkinSpec:
    """
    Complete specification for a TM2020 skin.

    name:           Output filename (without .zip)
    skin_color_fn:  Callable(size) -> RGBA Image for Skin_B
    skin_material:  PBR material for the body paint
    details_color_fn: Optional Callable(size) -> RGBA Image for Details_B
    details_material: PBR material for structural parts
    wheels_color_fn:  Optional Callable(size) -> RGBA Image for Wheels_B
    wheels_material:  PBR material for wheels
    glass_tint:     RGB tuple for glass color (luminosity -> opacity)
    roughness_noise: Per-pixel roughness variation
    metalness_noise: Per-pixel metalness variation
    clearcoat_edge_falloff: Reduce clearcoat at UV edges for wear
    dirt_amount:    Base dirt mask value (0=clean, 255=filthy)
    seed:           Random seed for reproducibility
    """
    name: str
    skin_color_fn: Callable[[int], Image.Image]
    skin_material: PBRMaterial = field(default_factory=lambda: GLOSS_PAINT)
    details_color_fn: Optional[Callable[[int], Image.Image]] = None
    details_material: PBRMaterial = field(default_factory=lambda: PLASTIC)
    wheels_color_fn: Optional[Callable[[int], Image.Image]] = None
    wheels_material: PBRMaterial = field(default_factory=lambda: RUBBER)
    glass_tint: Tuple[int, int, int] = (20, 20, 25)
    roughness_noise: int = 8
    metalness_noise: int = 4
    clearcoat_edge_falloff: bool = False
    dirt_amount: int = 128
    seed: int = 42


class TM2020SkinEngine:
    """Generates a complete TM2020 skin from a SkinSpec."""

    def __init__(self, spec: SkinSpec):
        self.spec = spec

    def generate(self, out_dir: Path) -> Path:
        """
        Generate all textures and pack into a zip.

        Returns the path to the output zip file.
        """
        spec = self.spec
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- Skin textures (body paint) ------------------------------------
        skin_b = spec.skin_color_fn(SKIN_SIZE)
        skin_r = generate_roughness_metalness(
            SKIN_SIZE,
            spec.skin_material,
            roughness_noise=spec.roughness_noise,
            metalness_noise=spec.metalness_noise,
            seed=spec.seed,
        )
        skin_coat = generate_clearcoat(
            SKIN_SIZE,
            spec.skin_material.clearcoat,
            edge_falloff=spec.clearcoat_edge_falloff,
        )
        skin_dirt = generate_dirt_mask(
            SKIN_SIZE,
            spec.dirt_amount,
            seed=spec.seed + 1,
        )

        # -- Details textures (structural parts) ---------------------------
        if spec.details_color_fn:
            details_b = spec.details_color_fn(DETAIL_SIZE)
        else:
            details_b = self._default_details_basecolor(DETAIL_SIZE)

        details_r = generate_roughness_metalness(
            DETAIL_SIZE,
            spec.details_material,
            seed=spec.seed + 10,
        )
        details_n = generate_flat_normal(DETAIL_SIZE)
        details_dirt = generate_dirt_mask(
            DETAIL_SIZE, spec.dirt_amount,
            noise_amount=50, seed=spec.seed + 11,
        )
        details_i = self._default_illumination(DETAIL_SIZE)

        # -- Wheels textures -----------------------------------------------
        if spec.wheels_color_fn:
            wheels_b = spec.wheels_color_fn(WHEEL_SIZE)
        else:
            wheels_b = self._default_wheels_basecolor(WHEEL_SIZE)

        wheels_r = generate_roughness_metalness(
            WHEEL_SIZE,
            spec.wheels_material,
            seed=spec.seed + 20,
        )
        wheels_n = generate_flat_normal(WHEEL_SIZE)
        wheels_dirt = generate_dirt_mask(
            WHEEL_SIZE, 180, seed=spec.seed + 21,
        )

        # -- Glass textures ------------------------------------------------
        glass_d = self._glass_basecolor(GLASS_SIZE, spec.glass_tint)
        glass_i = self._glass_illumination(GLASS_SIZE)

        # -- Icon ----------------------------------------------------------
        icon = self._render_icon(skin_b)

        # -- Pack into zip -------------------------------------------------
        zip_path = out_dir / f"{spec.name}.zip"
        textures = {
            "Skin_B.dds":           (skin_b,      BCFormat.BC1),
            "Skin_R.dds":           (skin_r,      BCFormat.BC5),
            "Skin_CoatR.dds":       (skin_coat,   BCFormat.BC4),
            "Skin_DirtMask.dds":    (skin_dirt,   BCFormat.BC4),
            "Details_B.dds":        (details_b,   BCFormat.BC1),
            "Details_R.dds":        (details_r,   BCFormat.BC5),
            "Details_I.dds":        (details_i,   BCFormat.BC3),
            "Details_N.dds":        (details_n,   BCFormat.BC5),
            "Details_DirtMask.dds": (details_dirt, BCFormat.BC4),
            "Wheels_B.dds":         (wheels_b,    BCFormat.BC1),
            "Wheels_R.dds":         (wheels_r,    BCFormat.BC5),
            "Wheels_N.dds":         (wheels_n,    BCFormat.BC5),
            "Wheels_DirtMask.dds":  (wheels_dirt, BCFormat.BC4),
            "Glass_D.dds":          (glass_d,     BCFormat.BC1),
            "Glass_I.dds":          (glass_i,     BCFormat.BC5),
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, (img, fmt) in textures.items():
                print(f"  Compressing {name} ({fmt.value})...")
                dds_bytes = build_dds_bytes(img, fmt, mipmaps=True)
                zf.writestr(name, dds_bytes)

            # Icon as TGA
            tga_buf = io.BytesIO()
            icon.save(tga_buf, format="TGA")
            zf.writestr("Icon.tga", tga_buf.getvalue())

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  -> {zip_path.name} ({size_mb:.1f} MB)")
        return zip_path

    # -- Default texture generators ----------------------------------------

    def _default_details_basecolor(self, size: int) -> Image.Image:
        """Dark gray structural parts."""
        return Image.new("RGBA", (size, size), (40, 40, 42, 255))

    def _default_wheels_basecolor(self, size: int) -> Image.Image:
        """Near-black rubber with slight warm tint."""
        return Image.new("RGBA", (size, size), (30, 28, 26, 255))

    def _default_illumination(self, size: int) -> Image.Image:
        """Black (no glow) with alpha=0 (inactive illumination role)."""
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    def _glass_basecolor(
        self, size: int, tint: Tuple[int, int, int]
    ) -> Image.Image:
        """Glass tint. Darker = more opaque glass."""
        return Image.new("RGBA", (size, size), tint + (255,))

    def _glass_illumination(self, size: int) -> Image.Image:
        """Default: no glass glow."""
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    def _render_icon(self, skin_b: Image.Image) -> Image.Image:
        """Downscale the body paint to a 64x64 icon."""
        return skin_b.resize(
            (ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS
        ).convert("RGBA")
