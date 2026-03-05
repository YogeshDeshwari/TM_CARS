"""
Palette generation utilities for high-contrast, vibrant liveries.

V2 -- OKLCH rewrite.

We sample palettes in OKLCH (Oklab cylindrical) colour space, which is
perceptually uniform: equal steps in L/C/H map to equal perceived changes.
This fixes the old HSV issues (uneven vibrancy across hues, muddy gradients,
the yellow ban) without adding any dependency -- the OKLCH math is ~40 lines
of matrix operations + cube roots.

Palette structure (unchanged from V1):
  - base:      dark, low chroma -- recedes, doesn't fight decals
  - secondary: near-base hue, slightly lighter -- adds depth
  - accent:    high chroma, strong contrast vs base -- hero colour
  - highlight: near-white or neon variant of accent -- small details
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# OKLCH  <->  sRGB  (pure math, zero dependencies)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp8(x: float) -> int:
    return max(0, min(255, int(round(x))))


def rgb255_to_oklab(c: RGB) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to OKLab (L in 0..1, a/b unbounded but small)."""
    r = _srgb_to_linear(c[0] / 255.0)
    g = _srgb_to_linear(c[1] / 255.0)
    b = _srgb_to_linear(c[2] / 255.0)

    l_ = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return (L, a, bb)


def oklab_to_rgb255(L: float, a: float, b: float) -> RGB:
    """Convert OKLab to sRGB (0-255), clamping out-of-gamut values."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return (
        _clamp8(_linear_to_srgb(_clamp01(r)) * 255),
        _clamp8(_linear_to_srgb(_clamp01(g)) * 255),
        _clamp8(_linear_to_srgb(_clamp01(bb)) * 255),
    )


def rgb255_to_oklch(c: RGB) -> Tuple[float, float, float]:
    """sRGB (0-255) -> OKLCH  (L 0..1, C 0..~0.4, H 0..360)."""
    L, a, b = rgb255_to_oklab(c)
    C = math.sqrt(a * a + b * b)
    H = math.degrees(math.atan2(b, a)) % 360.0
    return (L, C, H)


def oklch_to_rgb255(L: float, C: float, H: float) -> RGB:
    """OKLCH -> sRGB (0-255)."""
    h_rad = math.radians(H % 360.0)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    return oklab_to_rgb255(L, a, b)


def _in_srgb_gamut(L: float, C: float, H: float, tol: float = 0.002) -> bool:
    """Check whether an OKLCH colour is representable in sRGB."""
    rgb = oklch_to_rgb255(L, C, H)
    back = rgb255_to_oklch(rgb)
    return abs(back[0] - L) < tol and abs(back[1] - C) < 0.01


def _gamut_clamp_chroma(L: float, C: float, H: float) -> float:
    """Binary-search for the maximum in-gamut chroma at given L, H."""
    lo, hi = 0.0, C
    for _ in range(16):
        mid = (lo + hi) * 0.5
        if _in_srgb_gamut(L, mid, H):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Perceptual utilities
# ---------------------------------------------------------------------------

def delta_e_ok(c1: RGB, c2: RGB) -> float:
    """Euclidean distance in OKLab -- a simple, effective perceptual metric."""
    L1, a1, b1 = rgb255_to_oklab(c1)
    L2, a2, b2 = rgb255_to_oklab(c2)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def luma(c: RGB) -> float:
    """BT.709 luminance (kept for backward compat)."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast_ok(a: RGB, b: RGB, min_delta: float = 0.18) -> bool:
    """Perceptual contrast check via delta-E in OKLab.

    Default threshold 0.18 roughly equals the old luma delta of ~90.
    """
    return delta_e_ok(a, b) >= min_delta


def lighten(c: RGB, amount: float = 0.1) -> RGB:
    """Raise lightness in OKLCH by *amount* (0..1 scale)."""
    L, C, H = rgb255_to_oklch(c)
    L = min(1.0, L + amount)
    C = _gamut_clamp_chroma(L, C, H)
    return oklch_to_rgb255(L, C, H)


def darken(c: RGB, amount: float = 0.1) -> RGB:
    """Lower lightness in OKLCH by *amount*."""
    L, C, H = rgb255_to_oklch(c)
    L = max(0.0, L - amount)
    return oklch_to_rgb255(L, C, H)


def gradient_oklch(c1: RGB, c2: RGB, steps: int) -> List[RGB]:
    """Perceptually uniform gradient between two colours (interpolated in OKLab)."""
    L1, a1, b1 = rgb255_to_oklab(c1)
    L2, a2, b2 = rgb255_to_oklab(c2)
    result: List[RGB] = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        L = L1 + (L2 - L1) * t
        a = a1 + (a2 - a1) * t
        b = b1 + (b2 - b1) * t
        result.append(oklab_to_rgb255(L, a, b))
    return result


# ---------------------------------------------------------------------------
# Palette dataclass (unchanged API from V1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    base: RGB
    secondary: RGB
    accent: RGB
    highlight: RGB


# ---------------------------------------------------------------------------
# Harmony helpers (hue in degrees, 0..360)
# ---------------------------------------------------------------------------

_HARMONIES = [
    "complementary",
    "split_complementary",
    "triadic",
    "analogous",
    "tetradic",
    "square",
]


def _pick_harmony_hues(rng: random.Random, mode: str) -> Tuple[float, float]:
    """Returns (h_base, h_accent) in degrees [0..360)."""
    h = rng.uniform(0, 360)
    if mode == "complementary":
        return h, (h + 180) % 360
    if mode == "split_complementary":
        return h, (h + 180 + rng.choice([-30, 30])) % 360
    if mode == "triadic":
        return h, (h + rng.choice([120, 240])) % 360
    if mode == "analogous":
        return h, (h + rng.uniform(25, 50) * rng.choice([-1, 1])) % 360
    if mode == "tetradic":
        return h, (h + rng.choice([90, 270])) % 360
    if mode == "square":
        return h, (h + 90) % 360
    return h, (h + 180) % 360


# ---------------------------------------------------------------------------
# Main palette generator
# ---------------------------------------------------------------------------

def sample_vibrant_palette(seed: int) -> Palette:
    """Generate a vibrant, perceptually balanced palette in OKLCH.

    Drop-in replacement for the old HSV version -- same function signature,
    same Palette return type.  Downstream code sees no difference.
    """
    rng = random.Random(seed)
    harmony = rng.choice(_HARMONIES)
    h_base, h_acc = _pick_harmony_hues(rng, harmony)

    # --- Base: dark, low chroma -------------------------------------------
    base_L = rng.uniform(0.15, 0.25)
    base_C = rng.uniform(0.008, 0.035)
    base_C = _gamut_clamp_chroma(base_L, base_C, h_base)
    base = oklch_to_rgb255(base_L, base_C, h_base)

    # --- Secondary: same hue family, slightly lighter ---------------------
    sec_h = (h_base + rng.uniform(-8, 8)) % 360
    sec_L = rng.uniform(0.25, 0.38)
    sec_C = rng.uniform(0.02, 0.06)
    sec_C = _gamut_clamp_chroma(sec_L, sec_C, sec_h)
    secondary = oklch_to_rgb255(sec_L, sec_C, sec_h)

    # --- Accent: high chroma, medium-high lightness -----------------------
    acc_L = rng.uniform(0.55, 0.80)
    acc_C = rng.uniform(0.12, 0.25)
    acc_C = _gamut_clamp_chroma(acc_L, acc_C, h_acc)
    accent = oklch_to_rgb255(acc_L, acc_C, h_acc)

    # --- Highlight: near-white or neon variant of accent ------------------
    hl_h = (h_acc + rng.uniform(-15, 15)) % 360
    if rng.random() < 0.6:
        hl_L = rng.uniform(0.92, 0.98)
        hl_C = rng.uniform(0.03, 0.10)
    else:
        hl_L = rng.uniform(0.82, 0.95)
        hl_C = rng.uniform(0.12, 0.22)
    hl_C = _gamut_clamp_chroma(hl_L, hl_C, hl_h)
    highlight = oklch_to_rgb255(hl_L, hl_C, hl_h)

    # --- Contrast guardrails (perceptual) ---------------------------------
    if not contrast_ok(base, accent, 0.20):
        base = oklch_to_rgb255(0.15, 0.01, h_base)
        acc_C = _gamut_clamp_chroma(0.78, 0.22, h_acc)
        accent = oklch_to_rgb255(0.78, acc_C, h_acc)

    if not contrast_ok(base, highlight, 0.25):
        hl_C = _gamut_clamp_chroma(0.96, 0.06, hl_h)
        highlight = oklch_to_rgb255(0.96, hl_C, hl_h)

    return Palette(base=base, secondary=secondary, accent=accent, highlight=highlight)


# Alias for explicit naming
sample_vibrant_palette_oklch = sample_vibrant_palette
