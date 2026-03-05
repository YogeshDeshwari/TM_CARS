"""
Composable layer-stack compositor for TMNF/TMUF car skins.

Core idea: every visual element (base coat, accent stripe, pattern overlay,
sponsor decal, dirt ...) is an independent *layer* with its own blend mode,
opacity, spatial mask, and -- critically -- its own **finish** specification.

The stack flattens into two outputs:
  - Diffuse  (RGBA) -- color + game finish alpha
  - Details  (RGBA) -- RGB black, alpha = specular power per the TM convention

This decouples "what color" from "what material" at the layer level, which is
the single biggest upgrade over the old approach of deriving finish from
diffuse brightness after the fact.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Finish descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finish:
    """Per-layer material/finish specification.

    Parameters
    ----------
    type : str
        Semantic name: matte / satin / gloss / metallic / carbon / brushed.
    gloss : int
        Base specular-power alpha (0 = full matte, 255 = chrome mirror).
    variation : int
        Random noise amplitude added to *gloss* for sparkle / flake effects.
    pattern : str or None
        Optional micro-pattern baked into the finish alpha:
        "twill"     -- carbon-fiber 2x2 weave
        "brushed_h" -- horizontal brushed metal
        "brushed_v" -- vertical brushed metal
        None        -- flat (no pattern)
    """

    type: str = "satin"
    gloss: int = 100
    variation: int = 0
    pattern: Optional[str] = None


# Convenience presets
FINISH_MATTE    = Finish("matte",    gloss=20,  variation=4)
FINISH_SATIN    = Finish("satin",    gloss=100, variation=6)
FINISH_GLOSS    = Finish("gloss",    gloss=200, variation=3)
FINISH_METALLIC = Finish("metallic", gloss=210, variation=20)
FINISH_CARBON   = Finish("carbon",   gloss=55,  variation=8, pattern="twill")
FINISH_BRUSHED  = Finish("brushed",  gloss=170, variation=12, pattern="brushed_h")


# ---------------------------------------------------------------------------
# Single layer
# ---------------------------------------------------------------------------

@dataclass
class SkinLayer:
    tag: str
    content: Image.Image          # RGBA
    mask: Optional[Image.Image]   # L-mode or None (= full coverage)
    blend: str                    # normal / screen / overlay / multiply
    opacity: float                # 0.0 .. 1.0
    finish: Finish


# ---------------------------------------------------------------------------
# Layer stack
# ---------------------------------------------------------------------------

class SkinLayerStack:
    """Ordered list of layers with a shared paint_mask boundary."""

    def __init__(self, size: int = 2048, paint_mask: Optional[Image.Image] = None):
        self.size = size
        self.paint_mask: Image.Image = (
            paint_mask if paint_mask is not None
            else Image.new("L", (size, size), 255)
        )
        self._layers: List[SkinLayer] = []

    # -- mutators -----------------------------------------------------------

    def add(
        self,
        tag: str,
        image: Image.Image,
        *,
        mask: Optional[Image.Image] = None,
        blend: str = "normal",
        opacity: float = 1.0,
        finish: Optional[Finish] = None,
    ) -> None:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        if mask is not None and mask.mode != "L":
            mask = mask.convert("L")
        self._layers.append(SkinLayer(
            tag=tag,
            content=image,
            mask=mask,
            blend=blend,
            opacity=opacity,
            finish=finish or FINISH_SATIN,
        ))

    def insert(self, index: int, tag: str, image: Image.Image, **kw) -> None:
        """Insert a layer at a specific position (0 = bottom)."""
        before = self._layers[index:]
        self._layers = self._layers[:index]
        self.add(tag, image, **kw)
        self._layers.extend(before)

    @property
    def layers(self) -> List[SkinLayer]:
        return list(self._layers)

    # -- flatten (the main event) -------------------------------------------

    def flatten(
        self,
        clearcoat_sweep: Optional[str] = None,
        fresnel_boost: int = 0,
    ) -> Tuple[Image.Image, Image.Image]:
        """Composite all layers into Diffuse + Details.

        Parameters
        ----------
        clearcoat_sweep : str or None
            Direction for a global clearcoat gradient on the finish alpha.
            "horizontal" / "vertical" / "diagonal" / None.
        fresnel_boost : int
            Extra gloss (0-50) added at paint_mask edges to simulate Fresnel.

        Returns
        -------
        diffuse : Image  (RGBA -- color in RGB, finish alpha in A)
        details : Image  (RGBA -- black RGB, specular power in A)
        """
        sz = self.size
        diffuse = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        finish_alpha = Image.new("L", (sz, sz), 0)

        for layer in self._layers:
            img = layer.content.copy()

            # Apply layer mask (intersected with global paint_mask)
            effective_mask = self._effective_mask(layer.mask)

            # Apply opacity
            if layer.opacity < 1.0:
                a = img.getchannel("A")
                a = a.point(lambda p, o=layer.opacity: int(p * o))
                img.putalpha(a)

            # Restrict to effective mask
            masked = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
            masked.paste(img, (0, 0), effective_mask)

            # Blend onto diffuse
            diffuse = self._blend(diffuse, masked, layer.blend)

            # Build finish alpha for this layer's coverage
            coverage = ImageChops.multiply(
                masked.getchannel("A"),
                effective_mask,
            )
            layer_finish = self._render_finish_alpha(layer.finish, coverage)
            # Layers higher in the stack overwrite finish where they have coverage
            finish_alpha = _alpha_over(finish_alpha, layer_finish, coverage)

        # -- global finish post-effects -------------------------------------
        if clearcoat_sweep:
            sweep = self._make_clearcoat_sweep(clearcoat_sweep)
            sweep_masked = ImageChops.multiply(sweep, self.paint_mask)
            finish_alpha = ImageChops.lighter(finish_alpha, sweep_masked)

        if fresnel_boost > 0:
            edge = self._make_fresnel_edge(fresnel_boost)
            finish_alpha = ImageChops.lighter(finish_alpha, edge)

        # Pack into TM convention: Diffuse alpha = finish, Details alpha = spec
        diffuse.putalpha(ImageChops.multiply(finish_alpha, self.paint_mask))

        details = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
        details.putalpha(ImageChops.multiply(finish_alpha, self.paint_mask))

        return diffuse, details

    # -- internal helpers ---------------------------------------------------

    def _effective_mask(self, layer_mask: Optional[Image.Image]) -> Image.Image:
        if layer_mask is None:
            return self.paint_mask
        return ImageChops.multiply(layer_mask, self.paint_mask)

    @staticmethod
    def _blend(base: Image.Image, top: Image.Image, mode: str) -> Image.Image:
        if mode == "screen":
            return ImageChops.screen(base, top)
        if mode == "overlay":
            return ImageChops.overlay(base, top)
        if mode == "multiply":
            return ImageChops.multiply(base, top)
        return Image.alpha_composite(base, top)

    def _render_finish_alpha(self, finish: Finish, coverage: Image.Image) -> Image.Image:
        """Build an L-mode image with the finish's gloss value + variation + pattern."""
        sz = self.size
        base_val = max(0, min(255, finish.gloss))
        fa = Image.new("L", (sz, sz), base_val)

        if finish.variation > 0:
            noise = Image.effect_noise((sz, sz), finish.variation * 3)
            noise = noise.convert("L")
            noise = noise.point(lambda p, v=finish.variation: int((p - 128) * v / 128))
            fa = ImageChops.add(fa, noise, scale=1, offset=0)

        if finish.pattern == "twill":
            fa = self._apply_twill_pattern(fa, finish.gloss)
        elif finish.pattern in ("brushed_h", "brushed_v"):
            fa = self._apply_brushed_pattern(fa, finish.pattern)

        return fa

    def _apply_twill_pattern(self, base: Image.Image, gloss: int) -> Image.Image:
        """2x2 twill carbon weave modulation on finish alpha."""
        sz = self.size
        cell = 8
        tile = Image.new("L", (cell * 2, cell * 2), gloss)
        d = ImageDraw.Draw(tile)
        bright = min(255, gloss + 30)
        dark = max(0, gloss - 25)
        d.rectangle([0, 0, cell - 1, cell - 1], fill=bright)
        d.rectangle([cell, cell, cell * 2 - 1, cell * 2 - 1], fill=bright)
        d.rectangle([cell, 0, cell * 2 - 1, cell - 1], fill=dark)
        d.rectangle([0, cell, cell - 1, cell * 2 - 1], fill=dark)

        reps_x = math.ceil(sz / tile.width)
        reps_y = math.ceil(sz / tile.height)
        tiled = Image.new("L", (tile.width * reps_x, tile.height * reps_y))
        for ty in range(reps_y):
            for tx in range(reps_x):
                tiled.paste(tile, (tx * tile.width, ty * tile.height))
        tiled = tiled.crop((0, 0, sz, sz))
        return ImageChops.multiply(base, tiled.point(lambda p: int(p * 255 / max(1, bright))))

    def _apply_brushed_pattern(self, base: Image.Image, direction: str) -> Image.Image:
        """Directional 1D noise for brushed-metal finish."""
        sz = self.size
        if direction == "brushed_h":
            stripe = Image.effect_noise((sz, 1), 40).resize((sz, sz), Image.Resampling.NEAREST)
        else:
            stripe = Image.effect_noise((1, sz), 40).resize((sz, sz), Image.Resampling.NEAREST)
        stripe = stripe.convert("L").point(lambda p: int(128 + (p - 128) * 0.3))
        return ImageChops.multiply(base, stripe)

    def _make_clearcoat_sweep(self, direction: str) -> Image.Image:
        """Subtle gradient in finish alpha simulating body-panel curvature."""
        sz = self.size
        grad = Image.new("L", (sz, sz), 0)
        px = grad.load()
        for y in range(sz):
            for x in range(sz):
                if direction == "horizontal":
                    t = x / max(1, sz - 1)
                elif direction == "vertical":
                    t = y / max(1, sz - 1)
                else:
                    t = (x + y) / max(1, 2 * (sz - 1))
                bell = math.sin(t * math.pi)
                px[x, y] = int(bell * 35)
        return grad.filter(ImageFilter.GaussianBlur(sz // 64))

    def _make_fresnel_edge(self, boost: int) -> Image.Image:
        """Gloss boost at paint_mask boundaries (Fresnel approximation)."""
        edges = self.paint_mask.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(max(3, self.size // 200)))
        return edges.point(lambda p: min(255, int(p * boost / 50)))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _alpha_over(bottom: Image.Image, top: Image.Image, top_alpha: Image.Image) -> Image.Image:
    """Composite *top* over *bottom* using *top_alpha* as the blend mask.

    Where top_alpha is 255, result = top; where 0, result = bottom.
    """
    return Image.composite(top, bottom, top_alpha)
