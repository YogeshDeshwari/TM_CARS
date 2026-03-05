"""
Inspiration-based style compiler for TMNF/TMUF skins.

V2 -- layer-stack rewrite.

Now uses the composable SkinLayerStack via ProSkinEngine.add_layer(), giving
each motif its own Finish specification (matte base, glossy accents, etc.).
Also integrates the new V2 patterns (carbon_v2, camo_v2, hex, racing_stripes,
voronoi_shatter, metallic_flake, weathering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import random
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops

import skin_utils
from pro_skin_engine import ProSkinEngine
from layer_stack import (
    Finish,
    FINISH_MATTE,
    FINISH_SATIN,
    FINISH_GLOSS,
    FINISH_METALLIC,
    FINISH_CARBON,
    FINISH_BRUSHED,
)
from palette_lab import sample_vibrant_palette

RGB = Tuple[int, int, int]

_FINISH_MAP = {
    "matte":    FINISH_MATTE,
    "satin":    FINISH_SATIN,
    "gloss":    FINISH_GLOSS,
    "metallic": FINISH_METALLIC,
    "carbon":   FINISH_CARBON,
    "brushed":  FINISH_BRUSHED,
}


def clamp8(x: int) -> int:
    return max(0, min(255, int(x)))


def mix(c1: RGB, c2: RGB, t: float) -> RGB:
    return (
        clamp8(c1[0] * (1 - t) + c2[0] * t),
        clamp8(c1[1] * (1 - t) + c2[1] * t),
        clamp8(c1[2] * (1 - t) + c2[2] * t),
    )


def luma(c: RGB) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StyleRecipe:
    """A parameterized style grammar (unchanged public API)."""

    name: str
    seed: int
    palette: Dict[str, RGB]
    gloss_base: str
    accent_gloss_boost: int
    dirt_amount: float
    motif: str
    motif_stack: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Zone masks
# ---------------------------------------------------------------------------

class ZoneMasks:
    def __init__(self, size: int, paint_mask: Image.Image):
        self.size = size
        self.paint_mask = paint_mask
        self.hood = self._rect(0.00, 0.00, 0.55, 0.55, blur=12)
        self.roof = self._rect(0.30, 0.15, 0.70, 0.45, blur=10)
        self.sides = self._rect(0.00, 0.50, 1.00, 1.00, blur=10)
        self.center_band = self._rect(0.40, 0.00, 0.60, 1.00, blur=8)
        self.lower = self._rect(0.00, 0.75, 1.00, 1.00, blur=8)

    def _rect(self, x0: float, y0: float, x1: float, y1: float, blur: int) -> Image.Image:
        m = Image.new("L", (self.size, self.size), 0)
        d = ImageDraw.Draw(m)
        d.rectangle([x0 * self.size, y0 * self.size, x1 * self.size, y1 * self.size], fill=255)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        return ImageChops.multiply(m, self.paint_mask)


# ---------------------------------------------------------------------------
# Motif generators (unchanged from V1)
# ---------------------------------------------------------------------------

def motif_galaxy(size: int, rng: random.Random, base: RGB, accent: RGB, highlight: RGB) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    noise = Image.effect_noise((size // 2, size // 2), 18).resize((size, size), Image.Resampling.BICUBIC).convert("L")
    nebula = ImageOps.colorize(noise, "black", accent).convert("RGBA")
    nebula.putalpha(noise.point(lambda p: clamp8((p - 80) * 2)))
    nebula = nebula.filter(ImageFilter.GaussianBlur(10))
    img = Image.alpha_composite(img, nebula)

    noise2 = Image.effect_noise((size // 3, size // 3), 22).resize((size, size), Image.Resampling.BICUBIC).convert("L")
    wisp = ImageOps.colorize(noise2, "black", highlight).convert("RGBA")
    wisp.putalpha(noise2.point(lambda p: clamp8((p - 140) * 3)))
    wisp = wisp.filter(ImageFilter.GaussianBlur(6))
    img = Image.alpha_composite(img, wisp)

    d = ImageDraw.Draw(img)
    star_count = int(size * size / 6000)
    for _ in range(star_count):
        x = rng.randrange(0, size)
        y = rng.randrange(0, size)
        a = rng.randrange(120, 255)
        r = 1 if rng.random() < 0.85 else 2
        col = mix((255, 255, 255), highlight, rng.random() * 0.4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,))
    return img


def motif_leaves(size: int, rng: random.Random, accent: RGB, secondary: RGB) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    count = int(size * size / 50000) + 120
    for _ in range(count):
        x = rng.randrange(0, size)
        y = rng.randrange(0, size)
        s = rng.randrange(max(6, size // 260), max(14, size // 120))
        rot = rng.random()
        col = mix(accent, secondary, rng.random() * 0.6)
        alpha = rng.randrange(70, 200)

        w = int(s * (1.5 + rot))
        h = int(s * (0.7 + (1 - rot)))
        leaf = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(leaf)
        ld.ellipse([0, 0, w - 1, h - 1], fill=col + (alpha,))
        ld.line([(w * 0.2, h * 0.6), (w * 0.85, h * 0.35)], fill=mix(col, (255, 255, 255), 0.25) + (alpha,), width=1)
        leaf = leaf.rotate(rng.randrange(0, 360), expand=True, resample=Image.Resampling.BICUBIC)
        img.alpha_composite(leaf, (x - leaf.width // 2, y - leaf.height // 2))

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return img


def motif_minimal_blocks(size: int, rng: random.Random, accent: RGB, highlight: RGB) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    for _ in range(6):
        x0 = rng.random() * 0.9
        y0 = rng.random() * 0.9
        w = 0.1 + rng.random() * 0.35
        h = 0.02 + rng.random() * 0.12
        col = accent if rng.random() < 0.7 else highlight
        a = 220 if rng.random() < 0.5 else 150
        d.rectangle([x0 * size, y0 * size, (x0 + w) * size, (y0 + h) * size], fill=col + (a,))

    for _ in range(18):
        y = rng.random()
        d.rectangle([0, y * size, size, (y + 0.003) * size], fill=highlight + (rng.randrange(40, 90),))

    return img


def motif_fade_band(size: int, rng: random.Random, c1: RGB, c2: RGB, direction: str = "diagonal") -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    px = mask.load()
    for y in range(size):
        for x in range(size):
            if direction == "horizontal":
                t = x / (size - 1)
            elif direction == "vertical":
                t = y / (size - 1)
            else:
                t = (x + y) / (2 * (size - 1))
            px[x, y] = int(255 * t)
    fill1 = Image.new("RGBA", (size, size), c1 + (255,))
    fill2 = Image.new("RGBA", (size, size), c2 + (255,))
    fill2.putalpha(mask)
    band = Image.alpha_composite(fill1, fill2)
    alpha = mask.filter(ImageFilter.GaussianBlur(18))
    band.putalpha(alpha)
    return band


# ---------------------------------------------------------------------------
# Compiler (V2 -- layer-stack path)
# ---------------------------------------------------------------------------

class StyleCompiler:
    def __init__(self, size: int = 2048):
        self.size = size

    def render(self, recipe: StyleRecipe, team_name: Optional[str] = None) -> ProSkinEngine:
        rng = random.Random(recipe.seed)
        team_name = team_name or recipe.name

        engine = ProSkinEngine(size=self.size, team_name=team_name)
        zones = ZoneMasks(self.size, engine.paint_mask)

        base = recipe.palette["base"]
        secondary = recipe.palette["secondary"]
        accent = recipe.palette["accent"]
        highlight = recipe.palette["highlight"]

        base_finish = _FINISH_MAP.get(recipe.gloss_base, FINISH_SATIN)

        # --- Base coat (gradient for depth) --------------------------------
        base2 = mix(base, secondary, 0.25 if luma(base) < 80 else 0.15)
        grad = skin_utils.create_gradient(self.size, base, base2, direction="vertical")
        engine.add_layer("base_coat", grad, finish=base_finish)

        # --- Build motif stack ---------------------------------------------
        motif_list: List[str]
        if recipe.motif == "mixmatch":
            motif_list = recipe.motif_stack or ["fade", "topo", "halftone", "glitch"]
        else:
            motif_list = [recipe.motif]

        accent_finish = Finish("gloss", gloss=min(255, base_finish.gloss + recipe.accent_gloss_boost), variation=5)
        detail_finish = Finish("satin", gloss=min(255, base_finish.gloss + recipe.accent_gloss_boost // 2), variation=3)

        # Primary layer (structure)
        if "fade" in motif_list:
            fade = motif_fade_band(self.size, rng, secondary, accent, direction=rng.choice(["diagonal", "horizontal"]))
            engine.add_layer(
                "fade_band", fade,
                mask=ImageChops.lighter(zones.center_band, zones.sides),
                opacity=0.9,
                finish=accent_finish,
            )

        # Secondary layers (texture)
        if "topo" in motif_list:
            topo = skin_utils.generate_topo_lines(self.size, mix(accent, highlight, 0.25), density=18)
            engine.add_layer("topo", topo, mask=zones.sides, blend="screen", opacity=0.65, finish=detail_finish)

        if "camo" in motif_list:
            camo = skin_utils.generate_camo(self.size, [base, secondary, mix(accent, base, 0.3)], blobs=220)
            engine.add_layer("camo", camo, mask=zones.sides, blend="overlay", opacity=0.45, finish=base_finish)

        if "camo_v2" in motif_list:
            camo2 = skin_utils.generate_camo_v2(self.size, [base, secondary, mix(accent, base, 0.3)], seed=recipe.seed)
            engine.add_layer("camo_v2", camo2, mask=zones.sides, blend="overlay", opacity=0.50, finish=base_finish)

        if "carbon_v2" in motif_list:
            carbon = skin_utils.generate_carbon_v2(self.size, seed=recipe.seed)
            engine.add_layer("carbon_v2", carbon, mask=zones.lower, opacity=0.85, finish=FINISH_CARBON)

        if "hex" in motif_list:
            hx = skin_utils.generate_hex_tessellation(self.size, mix(base, secondary, 0.4), accent, seed=recipe.seed)
            engine.add_layer("hex", hx, mask=zones.sides, blend="overlay", opacity=0.55, finish=detail_finish)

        if "racing_stripes" in motif_list:
            stripe_angle = rng.choice([0, 15, -15, 30])
            stripes = skin_utils.generate_racing_stripes(
                self.size,
                [accent, highlight, accent],
                stripe_widths=[0.04, 0.015, 0.04],
                angle=stripe_angle,
            )
            engine.add_layer("racing_stripes", stripes, opacity=0.90, finish=FINISH_GLOSS)

        if "voronoi" in motif_list:
            vor = skin_utils.generate_voronoi_shatter(
                self.size, [base, secondary, accent, mix(accent, highlight, 0.3)], seed=recipe.seed,
            )
            engine.add_layer("voronoi", vor, mask=zones.sides, blend="overlay", opacity=0.50, finish=detail_finish)

        if "metallic_flake" in motif_list:
            flake = skin_utils.generate_metallic_flake(self.size, accent, seed=recipe.seed)
            engine.add_layer("metallic_flake", flake, mask=zones.sides, blend="screen", opacity=0.25, finish=FINISH_METALLIC)

        # Detail layers (micro accents)
        if "glitch" in motif_list:
            glitch = skin_utils.generate_glitch_bars(self.size, highlight, strength=1.0)
            engine.add_layer(
                "glitch", glitch,
                mask=ImageChops.lighter(zones.hood, zones.roof),
                blend="screen", opacity=0.55,
                finish=detail_finish,
            )

        if "halftone" in motif_list:
            ht = skin_utils.generate_halftone(self.size, highlight, density=0.12)
            engine.add_layer("halftone", ht, mask=zones.sides, blend="screen", opacity=0.35, finish=detail_finish)

        if "galaxy" in motif_list:
            gal = motif_galaxy(self.size, rng, base, accent, highlight)
            engine.add_layer(
                "galaxy", gal,
                mask=ImageChops.lighter(zones.hood, zones.roof),
                blend="screen", opacity=0.55,
                finish=accent_finish,
            )

        if "leaves" in motif_list:
            lv = motif_leaves(self.size, rng, accent, secondary)
            engine.add_layer(
                "leaves", lv,
                mask=ImageChops.lighter(zones.sides, zones.hood),
                opacity=0.70,
                finish=detail_finish,
            )

        if "minimal_blocks" in motif_list:
            bl = motif_minimal_blocks(self.size, rng, accent, highlight)
            engine.add_layer("minimal_blocks", bl, mask=zones.center_band, opacity=0.90, finish=accent_finish)

        # Weathering (optional, subtle on all skins for realism)
        if "weathering" in motif_list:
            wear = skin_utils.generate_weathering(self.size, intensity=0.3, seed=recipe.seed)
            engine.add_layer("weathering", wear, opacity=0.4, finish=FINISH_MATTE)

        return engine


# ---------------------------------------------------------------------------
# Recipe constructors (unchanged API)
# ---------------------------------------------------------------------------

def make_recipe(
    name: str,
    seed: int,
    base: RGB,
    secondary: RGB,
    accent: RGB,
    highlight: RGB,
    motif: str,
    gloss_base: str = "satin",
    accent_gloss_boost: int = 180,
    dirt_amount: float = 0.35,
) -> StyleRecipe:
    return StyleRecipe(
        name=name,
        seed=seed,
        palette={"base": base, "secondary": secondary, "accent": accent, "highlight": highlight},
        gloss_base=gloss_base,
        accent_gloss_boost=accent_gloss_boost,
        dirt_amount=dirt_amount,
        motif=motif,
    )


def make_mixmatch_recipe(name: str, seed: int, motif_stack: Optional[List[str]] = None) -> StyleRecipe:
    """Convenience: generate a novel vibrant palette + a stacked motif recipe."""
    pal = sample_vibrant_palette(seed)
    motif_stack = motif_stack or ["fade", "topo", "halftone", "glitch"]
    return StyleRecipe(
        name=name,
        seed=seed,
        palette={"base": pal.base, "secondary": pal.secondary, "accent": pal.accent, "highlight": pal.highlight},
        gloss_base="satin",
        accent_gloss_boost=210,
        dirt_amount=0.30,
        motif="mixmatch",
        motif_stack=motif_stack,
    )
