"""
Curated skin preset library with real layered livery designs.

Each preset builds a multi-layer composition:
  1. Base colors per UV role (foundation)
  2. Design layers: slashes, swooshes, bands, blocks, patterns, pinstripes
  3. Post-effects: hero gradient, prelight, OKLCH fade

Usage::

    from skin_presets import get_preset, list_presets
    from pro_skin_engine import ProSkinEngine

    engine = ProSkinEngine(team_name="MySkin", full_skin=True)
    engine.load_uv_geometry()

    preset = get_preset("gulf_spirit", engine.size)
    engine.paint_by_role(preset["role_spec"])
    engine.apply_design_layers(preset.get("design_layers", []))
    engine.apply_hero_gradient(**preset.get("hero_gradient", {}))
    engine.apply_prelight(**preset.get("prelight", {}))
    engine.save()
"""

from __future__ import annotations

from typing import Dict, Any, List, Callable, Optional, Tuple

from layer_stack import (
    Finish,
    FINISH_MATTE,
    FINISH_SATIN,
    FINISH_GLOSS,
    FINISH_METALLIC,
    FINISH_CARBON,
    FINISH_BRUSHED,
)
import skin_utils

HERO_ISLANDS = [1, 2, 3, 5, 6]
BODY_SIDES = [1, 5, 6]
TOP_AND_NOSE = [2, 3]
SECONDARY_ISLANDS = [7, 12, 13]
ACCENT_ISLANDS = [8, 9, 10, 15, 16]

_REGISTRY: Dict[str, Callable[[int], Dict[str, Any]]] = {}


def _register(category: str, name: str, description: str):
    def decorator(fn):
        fn._preset_meta = {
            "name": name,
            "category": category,
            "description": description,
        }
        _REGISTRY[name] = fn
        return fn
    return decorator


def _derive_dark(primary: Tuple[int, int, int],
                  fallback: Tuple[int, int, int],
                  factor: float = 0.30,
                  minimum: int = 25) -> Tuple[int, int, int]:
    """Derive a dark shade. Uses fallback if primary is too dark."""
    src = primary if sum(primary) > 80 else fallback
    return (max(minimum, int(src[0] * factor)),
            max(minimum, int(src[1] * factor)),
            max(minimum, int(src[2] * factor)))


def get_preset(name: str, size: int = 2048) -> Dict[str, Any]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    fn = _REGISTRY[name]
    result = fn(size)
    result.setdefault("hero_gradient", {"top_lighten": 0.12, "bottom_darken": 0.15})
    result.setdefault("prelight", {"strength": 0.55})
    result.setdefault("design_layers", [])
    result["meta"] = fn._preset_meta

    rs = result.get("role_spec", {})
    hero_col = rs.get("hero", {}).get("color", (100, 100, 100))
    sec_col = rs.get("secondary", {}).get("color", hero_col)
    acc_col = rs.get("accent", {}).get("color", hero_col)

    if "darken" in rs:
        dc = rs["darken"].get("color", (0, 0, 0))
        if sum(dc) < 50:
            rs["darken"]["color"] = _derive_dark(hero_col, acc_col, 0.30, 20)

    if "neutral" in rs:
        nc = rs["neutral"].get("color", (0, 0, 0))
        if sum(nc) < 50:
            rs["neutral"]["color"] = _derive_dark(sec_col, hero_col, 0.22, 15)

    return result


def list_presets() -> List[Dict[str, str]]:
    out = []
    for name, fn in sorted(_REGISTRY.items()):
        m = fn._preset_meta.copy()
        out.append(m)
    return out


# ===================================================================
# Category 1: Iconic Racing
# ===================================================================

@_register("iconic_racing", "gulf_spirit",
           "Powder blue with bold orange band, navy lower -- Gulf Oil heritage")
def _gulf_spirit(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (108, 190, 218), "finish": Finish(type="gloss", gloss=195, variation=4)},
            "secondary": {"color": (0, 30, 66), "finish": Finish(type="satin", gloss=120)},
            "accent": {"color": (232, 119, 34), "finish": Finish(type="gloss", gloss=200)},
            "darken": {"color": (0, 18, 42), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 12, 18), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "band", "color": (232, 119, 34), "alpha": 230,
             "highlight": (255, 200, 120), "highlight_alpha": 200,
             "band_width": 0.22, "angle": -12.0, "offset_y": -0.05,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=200)},
            {"type": "slash", "color": (0, 30, 66), "alpha": 160,
             "angle": 25.0, "thickness": 0.28, "position": 0.72,
             "feather": 22, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(255, 200, 120), (232, 119, 34)],
             "angle_range": (55, 72), "thickness": 0.006, "alpha": 70,
             "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.12},
        "prelight": {"strength": 0.55},
    }


@_register("iconic_racing", "martini_stripe",
           "White body with navy/red racing stripes, carbon lower -- Martini heritage")
def _martini_stripe(size: int) -> Dict[str, Any]:
    carbon = skin_utils.generate_carbon_v2(size, base_tone=18, scale=6, seed=11)
    return {
        "role_spec": {
            "hero": {"color": (242, 242, 245), "finish": Finish(type="gloss", gloss=200, variation=3)},
            "secondary": {"color": (15, 15, 18), "pattern": carbon, "pattern_opacity": 0.7, "finish": FINISH_CARBON},
            "accent": {"color": (190, 30, 45), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (5, 5, 8), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 14), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "slash", "color": (0, 32, 91), "alpha": 220,
             "angle": 90.0, "thickness": 0.055, "position": 0.44, "feather": 2,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (190, 30, 45), "alpha": 240,
             "angle": 90.0, "thickness": 0.025, "position": 0.50, "feather": 2,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (0, 32, 91), "alpha": 220,
             "angle": 90.0, "thickness": 0.055, "position": 0.56, "feather": 2,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (15, 15, 18), "alpha": 150,
             "angle": 25.0, "thickness": 0.30, "position": 0.75,
             "feather": 20, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 2,
             "colors": [(190, 30, 45)], "thickness": 0.004, "alpha": 60,
             "angle_range": (60, 75), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.06, "bottom_darken": 0.08},
        "prelight": {"strength": 0.50},
    }


@_register("iconic_racing", "jps_gold",
           "Black body with gold diagonal band, gold pinstripes -- JPS Lotus DNA")
def _jps_gold(size: int) -> Dict[str, Any]:
    gold_flake = skin_utils.generate_metallic_flake(
        size, base_color=(200, 170, 55), flake_density=0.7, seed=72)
    return {
        "role_spec": {
            "hero": {"color": (8, 8, 10), "finish": Finish(type="matte", gloss=30, variation=5)},
            "secondary": {"color": (12, 12, 14), "finish": Finish(type="satin", gloss=80)},
            "accent": {"color": (200, 170, 55), "pattern": gold_flake, "pattern_opacity": 0.85,
                        "finish": Finish(type="metallic", gloss=220, variation=25)},
            "darken": {"color": (4, 4, 5), "finish": FINISH_MATTE},
            "neutral": {"color": (5, 5, 6), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "band", "color": (200, 170, 55), "alpha": 210,
             "highlight": (255, 220, 80), "highlight_alpha": 200,
             "band_width": 0.18, "angle": -15.0, "offset_y": -0.05,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=220, variation=25)},
            {"type": "slash", "color": (200, 170, 55), "alpha": 200,
             "angle": 78.0, "thickness": 0.018, "position": 0.40, "feather": 1,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (200, 170, 55), "alpha": 200,
             "angle": 82.0, "thickness": 0.012, "position": 0.62, "feather": 0,
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 5,
             "colors": [(255, 220, 80), (200, 170, 55)],
             "angle_range": (65, 82), "thickness": 0.005, "alpha": 65,
             "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.04, "bottom_darken": 0.06},
        "prelight": {"strength": 0.45},
    }


@_register("iconic_racing", "rothmans_elegance",
           "White/blue diagonal split with gold pinstripe divider -- Rothmans era")
def _rothmans_elegance(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (245, 245, 248), "finish": Finish(type="gloss", gloss=195, variation=3)},
            "secondary": {"color": (0, 40, 120), "finish": Finish(type="gloss", gloss=185)},
            "accent": {"color": (210, 175, 55), "finish": Finish(type="metallic", gloss=215, variation=18)},
            "darken": {"color": (0, 20, 60), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 10, 18), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "blocks", "color": (0, 40, 120), "alpha": 230,
             "color2": (0, 25, 80), "alpha2": 180, "style": "split",
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (210, 175, 55), "alpha": 220,
             "angle": 0.0, "thickness": 0.008, "position": 0.42, "feather": 1,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=215, variation=18)},
            {"type": "slash", "color": (180, 30, 40), "alpha": 180,
             "angle": 0.0, "thickness": 0.004, "position": 0.44, "feather": 0,
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 3,
             "colors": [(210, 175, 55)], "thickness": 0.005, "alpha": 55,
             "angle_range": (70, 85), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.05, "bottom_darken": 0.10},
        "prelight": {"strength": 0.55},
    }


@_register("iconic_racing", "bmw_m_tribute",
           "White body with bold M-stripes, dark lower half, strong contrast")
def _bmw_m_tribute(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (248, 248, 250), "finish": Finish(type="gloss", gloss=190, variation=3)},
            "secondary": {"color": (16, 16, 20), "finish": Finish(type="satin", gloss=100)},
            "accent": {"color": (0, 90, 181), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (10, 10, 14), "finish": FINISH_MATTE},
            "neutral": {"color": (12, 12, 15), "finish": FINISH_MATTE},
        },
        "design_layers": [
            # Dark lower half for strong white/black contrast
            {"type": "slash", "color": (16, 16, 20), "alpha": 200,
             "angle": 25.0, "thickness": 0.35, "position": 0.72,
             "feather": 15, "islands": BODY_SIDES},
            # Bold M-stripes -- wider and more opaque
            {"type": "slash", "color": (0, 90, 181), "alpha": 245,
             "angle": 90.0, "thickness": 0.050, "position": 0.44, "feather": 2,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (100, 45, 155), "alpha": 245,
             "angle": 90.0, "thickness": 0.040, "position": 0.50, "feather": 2,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (220, 25, 25), "alpha": 245,
             "angle": 90.0, "thickness": 0.050, "position": 0.56, "feather": 2,
             "islands": HERO_ISLANDS},
            # Dark swoosh for dynamic lower body
            {"type": "swoosh", "color": (16, 16, 20), "alpha": 190,
             "thickness": 90, "curve_type": "arc", "flip": False,
             "islands": BODY_SIDES},
            # Thin white highlight pinstripes bordering the M-stripes
            {"type": "slash", "color": (255, 255, 255), "alpha": 120,
             "angle": 90.0, "thickness": 0.004, "position": 0.415, "feather": 0,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (255, 255, 255), "alpha": 120,
             "angle": 90.0, "thickness": 0.004, "position": 0.585, "feather": 0,
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 4,
             "colors": [(0, 90, 181), (220, 25, 25), (100, 45, 155)],
             "thickness": 0.005, "alpha": 60,
             "angle_range": (55, 78), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.14},
        "prelight": {"strength": 0.60},
    }


@_register("iconic_racing", "marlboro_edge",
           "Red body with white band + carbon swoosh lower -- classic red/white")
def _marlboro_edge(size: int) -> Dict[str, Any]:
    carbon = skin_utils.generate_carbon_v2(size, base_tone=15, scale=7, seed=33)
    return {
        "role_spec": {
            "hero": {"color": (190, 22, 34), "finish": Finish(type="metallic", gloss=180, variation=12)},
            "secondary": {"color": (14, 14, 16), "pattern": carbon, "pattern_opacity": 0.85, "finish": FINISH_CARBON},
            "accent": {"color": (248, 248, 250), "finish": Finish(type="gloss", gloss=205)},
            "darken": {"color": (6, 4, 5), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 8, 10), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "band", "color": (248, 248, 250), "alpha": 240,
             "highlight": (255, 255, 255), "highlight_alpha": 220,
             "band_width": 0.24, "angle": -10.0, "offset_y": -0.12,
             "islands": HERO_ISLANDS},
            {"type": "swoosh", "color": (14, 14, 16), "alpha": 170,
             "thickness": 70, "curve_type": "arc",
             "islands": BODY_SIDES},
            {"type": "overlay", "image": carbon, "opacity": 0.4,
             "blend": "normal", "islands": [1, 5, 6]},
            {"type": "pinstripes", "count": 4,
             "colors": [(248, 248, 250), (255, 200, 200)],
             "thickness": 0.006, "alpha": 60,
             "angle_range": (50, 70), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.12, "bottom_darken": 0.18},
        "prelight": {"strength": 0.60},
    }


# ===================================================================
# Category 2: Modern Automotive
# ===================================================================

@_register("modern_auto", "stealth_matte",
           "Matte black with neon orange slash + carbon texture + pinstripes")
def _stealth_matte(size: int) -> Dict[str, Any]:
    carbon = skin_utils.generate_carbon_v2(size, base_tone=22, scale=8, seed=77)
    return {
        "role_spec": {
            "hero": {"color": (18, 18, 22), "finish": Finish(type="matte", gloss=25, variation=6)},
            "secondary": {"color": (35, 35, 40), "finish": Finish(type="satin", gloss=100)},
            "accent": {"color": (255, 60, 0), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (6, 6, 8), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 8, 10), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": carbon, "opacity": 0.45,
             "blend": "normal", "islands": HERO_ISLANDS},
            {"type": "slash", "color": (35, 35, 40), "alpha": 100,
             "angle": 25.0, "thickness": 0.28, "position": 0.65,
             "feather": 22, "islands": BODY_SIDES},
            {"type": "slash", "color": (255, 60, 0), "alpha": 220,
             "angle": 55.0, "thickness": 0.10, "position": 0.30,
             "feather": 4, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=210)},
            {"type": "slash", "color": (255, 120, 40), "alpha": 160,
             "angle": 52.0, "thickness": 0.025, "position": 0.26,
             "feather": 2, "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 4,
             "colors": [(255, 60, 0), (255, 120, 40)],
             "thickness": 0.006, "alpha": 70,
             "angle_range": (50, 75), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.03, "bottom_darken": 0.04},
        "prelight": {"strength": 0.40},
    }


@_register("modern_auto", "satin_gunmetal",
           "Gunmetal metallic with white center stripe + arc swoosh + flake")
def _satin_gunmetal(size: int) -> Dict[str, Any]:
    flake = skin_utils.generate_metallic_flake(
        size, base_color=(85, 90, 95), flake_density=0.4, seed=55)
    return {
        "role_spec": {
            "hero": {"color": (85, 90, 95), "pattern": flake, "pattern_opacity": 0.6, "finish": FINISH_BRUSHED},
            "secondary": {"color": (12, 12, 14), "finish": Finish(type="satin", gloss=90)},
            "accent": {"color": (240, 240, 245), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (5, 5, 6), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 8, 10), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "slash", "color": (30, 30, 35), "alpha": 120,
             "angle": 25.0, "thickness": 0.30, "position": 0.68,
             "feather": 20, "islands": BODY_SIDES},
            {"type": "slash", "color": (240, 240, 245), "alpha": 230,
             "angle": 90.0, "thickness": 0.04, "position": 0.50,
             "feather": 2, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=210)},
            {"type": "slash", "color": (200, 40, 30), "alpha": 200,
             "angle": 90.0, "thickness": 0.005, "position": 0.50,
             "feather": 1, "islands": HERO_ISLANDS},
            {"type": "swoosh", "color": (240, 240, 245), "alpha": 130,
             "thickness": 25, "curve_type": "arc", "flip": True,
             "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(240, 240, 245), (200, 40, 30)],
             "thickness": 0.005, "alpha": 55,
             "angle_range": (60, 80), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.08, "bottom_darken": 0.12},
        "prelight": {"strength": 0.55},
    }


@_register("modern_auto", "nardo_grey",
           "Flat grey with neon lime slash + black geometric blocks + pinstripes")
def _nardo_grey(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (148, 148, 150), "finish": Finish(type="matte", gloss=35, variation=4)},
            "secondary": {"color": (14, 14, 16), "finish": Finish(type="satin", gloss=85)},
            "accent": {"color": (130, 255, 5), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (6, 6, 7), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 12), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "slash", "color": (14, 14, 16), "alpha": 160,
             "angle": 25.0, "thickness": 0.30, "position": 0.70,
             "feather": 18, "islands": BODY_SIDES},
            {"type": "slash", "color": (130, 255, 5), "alpha": 230,
             "angle": 50.0, "thickness": 0.08, "position": 0.28,
             "feather": 4, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=210)},
            {"type": "slash", "color": (14, 14, 16), "alpha": 180,
             "angle": 48.0, "thickness": 0.015, "position": 0.24,
             "feather": 1, "islands": HERO_ISLANDS},
            {"type": "blocks", "color": (14, 14, 16), "alpha": 100,
             "color2": (130, 255, 5), "alpha2": 80, "style": "corner",
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 4,
             "colors": [(130, 255, 5), (14, 14, 16)],
             "thickness": 0.006, "alpha": 60,
             "angle_range": (45, 72), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.05, "bottom_darken": 0.08},
        "prelight": {"strength": 0.50},
    }


@_register("modern_auto", "viper_green",
           "BRG with gold pinstripes + carbon sidepods + arc swoosh")
def _viper_green(size: int) -> Dict[str, Any]:
    carbon = skin_utils.generate_carbon_v2(size, base_tone=12, scale=7, seed=88)
    return {
        "role_spec": {
            "hero": {"color": (0, 66, 37), "finish": Finish(type="metallic", gloss=175, variation=14)},
            "secondary": {"color": (10, 10, 12), "pattern": carbon, "pattern_opacity": 0.8, "finish": FINISH_CARBON},
            "accent": {"color": (195, 165, 45), "finish": Finish(type="metallic", gloss=215, variation=20)},
            "darken": {"color": (0, 20, 12), "finish": FINISH_MATTE},
            "neutral": {"color": (6, 10, 8), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": carbon, "opacity": 0.3,
             "blend": "normal", "islands": [5, 6]},
            {"type": "slash", "color": (0, 30, 18), "alpha": 140,
             "angle": 25.0, "thickness": 0.25, "position": 0.68,
             "feather": 20, "islands": BODY_SIDES},
            {"type": "swoosh", "color": (195, 165, 45), "alpha": 170,
             "thickness": 18, "curve_type": "arc",
             "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=215, variation=20)},
            {"type": "slash", "color": (195, 165, 45), "alpha": 210,
             "angle": 90.0, "thickness": 0.006, "position": 0.50,
             "feather": 0, "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 5,
             "colors": [(195, 165, 45), (255, 220, 80)],
             "thickness": 0.005, "alpha": 55,
             "angle_range": (68, 85), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.15},
        "prelight": {"strength": 0.55},
    }


@_register("modern_auto", "frozen_white",
           "Matte white with red diagonal band + grey swoosh lower + pinstripes")
def _frozen_white(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (238, 238, 240), "finish": Finish(type="satin", gloss=130, variation=3)},
            "secondary": {"color": (55, 55, 60), "finish": Finish(type="satin", gloss=100)},
            "accent": {"color": (210, 25, 30), "finish": Finish(type="gloss", gloss=200)},
            "darken": {"color": (25, 25, 28), "finish": FINISH_MATTE},
            "neutral": {"color": (15, 15, 18), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "band", "color": (210, 25, 30), "alpha": 230,
             "highlight": (255, 100, 100), "highlight_alpha": 190,
             "band_width": 0.20, "angle": -15.0, "offset_y": 0.0,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=200)},
            {"type": "swoosh", "color": (55, 55, 60), "alpha": 150,
             "thickness": 65, "curve_type": "arc",
             "islands": BODY_SIDES},
            {"type": "slash", "color": (55, 55, 60), "alpha": 120,
             "angle": 25.0, "thickness": 0.25, "position": 0.72,
             "feather": 18, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(210, 25, 30), (55, 55, 60)],
             "thickness": 0.005, "alpha": 50,
             "angle_range": (55, 75), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.04, "bottom_darken": 0.06},
        "prelight": {"strength": 0.50},
    }


# ===================================================================
# Category 3: Artistic / Experimental
# ===================================================================

@_register("artistic", "hex_chrome",
           "Dark chrome with hex overlay + chrome band + metallic pinstripes")
def _hex_chrome(size: int) -> Dict[str, Any]:
    hex_pat = skin_utils.generate_hex_tessellation(
        size, color1=(60, 65, 75), color2=(110, 115, 130), cell_size=50, seed=42)
    carbon = skin_utils.generate_carbon_v2(size, base_tone=14, scale=6, seed=42)
    return {
        "role_spec": {
            "hero": {"color": (28, 30, 35), "finish": Finish(type="satin", gloss=130, variation=10)},
            "secondary": {"color": (12, 12, 14), "pattern": carbon, "pattern_opacity": 0.7, "finish": FINISH_CARBON},
            "accent": {"color": (200, 205, 215), "finish": Finish(type="metallic", gloss=230, variation=15)},
            "darken": {"color": (5, 5, 6), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 8, 10), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": hex_pat, "opacity": 0.65,
             "blend": "normal", "islands": HERO_ISLANDS},
            {"type": "band", "color": (200, 205, 215), "alpha": 200,
             "highlight": (255, 255, 255), "highlight_alpha": 180,
             "band_width": 0.15, "angle": -20.0, "offset_y": -0.05,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=230, variation=15)},
            {"type": "slash", "color": (8, 8, 10), "alpha": 140,
             "angle": 25.0, "thickness": 0.25, "position": 0.72,
             "feather": 20, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 4,
             "colors": [(200, 205, 215), (255, 255, 255)],
             "thickness": 0.006, "alpha": 65,
             "angle_range": (55, 78), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.06, "bottom_darken": 0.10},
        "prelight": {"strength": 0.50},
    }


@_register("artistic", "shattered_ice",
           "Ice white body covered in high-contrast voronoi shatter + bold cyan accents")
def _shattered_ice(size: int) -> Dict[str, Any]:
    # Primary shatter: high-contrast blue cells on white, bright cyan edges
    shatter_main = skin_utils.generate_voronoi_shatter(
        size, colors=[(80, 140, 190), (40, 90, 160), (180, 220, 245), (20, 70, 140)],
        cell_count=70, seed=17, edge_color=(0, 220, 255))
    # Second shatter layer: finer cracks overlaid for density
    shatter_fine = skin_utils.generate_voronoi_shatter(
        size, colors=[(150, 200, 235), (100, 160, 210), (210, 235, 250)],
        cell_count=120, seed=43, edge_color=(0, 180, 220))
    return {
        "role_spec": {
            "hero": {"color": (220, 235, 248), "finish": Finish(type="gloss", gloss=200, variation=5)},
            "secondary": {"color": (10, 18, 28), "finish": Finish(type="satin", gloss=110)},
            "accent": {"color": (0, 210, 240), "finish": Finish(type="gloss", gloss=220)},
            "darken": {"color": (5, 10, 15), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 12, 18), "finish": FINISH_MATTE},
        },
        "design_layers": [
            # Main shatter: full opacity, covers entire hero area
            {"type": "overlay", "image": shatter_main, "opacity": 0.92,
             "blend": "normal", "islands": HERO_ISLANDS},
            # Fine shatter on top for extra crack density
            {"type": "overlay", "image": shatter_fine, "opacity": 0.40,
             "blend": "screen", "islands": HERO_ISLANDS},
            # Also shatter on secondary islands so the whole car looks cracked
            {"type": "overlay", "image": shatter_main, "opacity": 0.65,
             "blend": "normal", "islands": SECONDARY_ISLANDS},
            # Bold cyan slash cutting through
            {"type": "slash", "color": (0, 220, 255), "alpha": 230,
             "angle": 58.0, "thickness": 0.08, "position": 0.28,
             "feather": 4, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=220)},
            {"type": "slash", "color": (0, 160, 200), "alpha": 170,
             "angle": 55.0, "thickness": 0.02, "position": 0.24,
             "feather": 2, "islands": HERO_ISLANDS},
            # Dark lower body for contrast
            {"type": "slash", "color": (10, 18, 28), "alpha": 160,
             "angle": 25.0, "thickness": 0.22, "position": 0.72,
             "feather": 16, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 4,
             "colors": [(0, 220, 255), (0, 160, 200), (255, 255, 255)],
             "thickness": 0.005, "alpha": 70,
             "angle_range": (50, 72), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.06, "bottom_darken": 0.10},
        "prelight": {"strength": 0.55},
    }


@_register("artistic", "camo_tactical",
           "Olive camo with orange slash + dark blocks + tactical pinstripes")
def _camo_tactical(size: int) -> Dict[str, Any]:
    camo = skin_utils.generate_camo_v2(
        size, palette=[(65, 70, 45), (35, 38, 22), (100, 95, 65), (20, 22, 14)],
        cell_count=90, seed=66)
    return {
        "role_spec": {
            "hero": {"color": (65, 70, 45), "finish": Finish(type="matte", gloss=25, variation=8)},
            "secondary": {"color": (14, 14, 12), "finish": Finish(type="matte", gloss=20)},
            "accent": {"color": (230, 120, 20), "finish": Finish(type="satin", gloss=140)},
            "darken": {"color": (8, 8, 6), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 8), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": camo, "opacity": 0.90,
             "blend": "normal", "islands": HERO_ISLANDS},
            {"type": "slash", "color": (230, 120, 20), "alpha": 220,
             "angle": 55.0, "thickness": 0.08, "position": 0.35,
             "feather": 4, "islands": BODY_SIDES,
             "finish": Finish(type="satin", gloss=140)},
            {"type": "blocks", "color": (14, 14, 12), "alpha": 130,
             "color2": (230, 120, 20), "alpha2": 100, "style": "corner",
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (14, 14, 12), "alpha": 140,
             "angle": 25.0, "thickness": 0.22, "position": 0.72,
             "feather": 15, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(230, 120, 20), (100, 95, 65)],
             "thickness": 0.006, "alpha": 60,
             "angle_range": (45, 65), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.02, "bottom_darken": 0.04},
        "prelight": {"strength": 0.40},
    }


@_register("artistic", "digital_fracture",
           "Deep black with intense neon green glitch, circuits, crossing slashes")
def _digital_fracture(size: int) -> Dict[str, Any]:
    glitch = skin_utils.generate_glitch_bars(size, color=(0, 255, 80), strength=1.0)
    circuit = skin_utils.generate_circuit_traces(size, color=(0, 220, 70), density="high")
    carbon = skin_utils.generate_carbon_v2(size, base_tone=12, scale=6, seed=99)
    return {
        "role_spec": {
            "hero": {"color": (5, 6, 5), "finish": Finish(type="matte", gloss=25, variation=6)},
            "secondary": {"color": (8, 8, 10), "pattern": carbon, "pattern_opacity": 0.70, "finish": FINISH_CARBON},
            "accent": {"color": (0, 255, 80), "finish": Finish(type="gloss", gloss=225)},
            "darken": {"color": (3, 4, 3), "finish": FINISH_MATTE},
            "neutral": {"color": (5, 6, 5), "finish": FINISH_MATTE},
        },
        "design_layers": [
            # Heavy glitch bars across entire body
            {"type": "overlay", "image": glitch, "opacity": 0.80,
             "blend": "screen", "islands": HERO_ISLANDS},
            # Circuit traces on body sides + secondary areas
            {"type": "overlay", "image": circuit, "opacity": 0.60,
             "blend": "screen", "islands": BODY_SIDES},
            {"type": "overlay", "image": circuit, "opacity": 0.40,
             "blend": "screen", "islands": SECONDARY_ISLANDS},
            # Primary neon slash -- thick, bright
            {"type": "slash", "color": (0, 255, 80), "alpha": 240,
             "angle": 65.0, "thickness": 0.10, "position": 0.25,
             "feather": 4, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=225)},
            # Crossing counter-slash for X pattern
            {"type": "slash", "color": (0, 255, 80), "alpha": 180,
             "angle": -55.0, "thickness": 0.05, "position": 0.65,
             "feather": 5, "islands": HERO_ISLANDS},
            # Neon wave swoosh
            {"type": "swoosh", "color": (0, 255, 80), "alpha": 170,
             "thickness": 35, "curve_type": "wave",
             "islands": HERO_ISLANDS},
            # Dense neon pinstripes
            {"type": "pinstripes", "count": 6,
             "colors": [(0, 255, 80), (0, 220, 70), (100, 255, 140)],
             "thickness": 0.006, "alpha": 70,
             "angle_range": (45, 80), "seed": 99, "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.30},
    }


@_register("artistic", "cyber_magenta",
           "Deep black with intense hot-pink glitch, circuits, crossing slashes")
def _cyber_magenta(size: int) -> Dict[str, Any]:
    glitch = skin_utils.generate_glitch_bars(size, color=(255, 0, 110), strength=1.0)
    circuit = skin_utils.generate_circuit_traces(size, color=(255, 50, 150), density="high")
    carbon = skin_utils.generate_carbon_v2(size, base_tone=10, scale=6, seed=41)
    return {
        "role_spec": {
            "hero": {"color": (5, 3, 6), "finish": Finish(type="matte", gloss=25, variation=6)},
            "secondary": {"color": (8, 5, 10), "pattern": carbon, "pattern_opacity": 0.70, "finish": FINISH_CARBON},
            "accent": {"color": (255, 0, 110), "finish": Finish(type="gloss", gloss=225)},
            "darken": {"color": (4, 2, 5), "finish": FINISH_MATTE},
            "neutral": {"color": (5, 3, 6), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": glitch, "opacity": 0.80,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": circuit, "opacity": 0.60,
             "blend": "screen", "islands": BODY_SIDES},
            {"type": "overlay", "image": circuit, "opacity": 0.40,
             "blend": "screen", "islands": SECONDARY_ISLANDS},
            {"type": "slash", "color": (255, 0, 110), "alpha": 240,
             "angle": 65.0, "thickness": 0.10, "position": 0.25,
             "feather": 4, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=225)},
            {"type": "slash", "color": (255, 0, 110), "alpha": 180,
             "angle": -55.0, "thickness": 0.05, "position": 0.65,
             "feather": 5, "islands": HERO_ISLANDS},
            {"type": "swoosh", "color": (255, 0, 110), "alpha": 170,
             "thickness": 35, "curve_type": "wave",
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 6,
             "colors": [(255, 0, 110), (255, 50, 150), (255, 120, 180)],
             "thickness": 0.006, "alpha": 70,
             "angle_range": (45, 80), "seed": 41, "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.30},
    }


@_register("artistic", "cyber_cyan",
           "Dark purple-black with electric cyan swoosh, teal circuits, neon edges")
def _cyber_cyan(size: int) -> Dict[str, Any]:
    glitch = skin_utils.generate_glitch_bars(size, color=(0, 255, 255), strength=0.85)
    circuit = skin_utils.generate_circuit_traces(size, color=(0, 200, 220), density="medium")
    hex_pat = skin_utils.generate_hex_tessellation(
        size, color1=(0, 40, 50), color2=(0, 80, 90), cell_size=45, seed=55)
    return {
        "role_spec": {
            "hero": {"color": (8, 4, 14), "finish": Finish(type="matte", gloss=28, variation=5)},
            "secondary": {"color": (6, 3, 10), "finish": Finish(type="satin", gloss=80)},
            "accent": {"color": (0, 255, 255), "finish": Finish(type="gloss", gloss=225)},
            "darken": {"color": (4, 2, 8), "finish": FINISH_MATTE},
            "neutral": {"color": (5, 3, 9), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": hex_pat, "opacity": 0.55,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": glitch, "opacity": 0.70,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": circuit, "opacity": 0.55,
             "blend": "screen", "islands": BODY_SIDES},
            {"type": "swoosh", "color": (0, 255, 255), "alpha": 210,
             "thickness": 45, "curve_type": "wave",
             "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=225)},
            {"type": "swoosh", "color": (0, 180, 200), "alpha": 150,
             "thickness": 25, "curve_type": "arc", "flip": True,
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (0, 255, 255), "alpha": 200,
             "angle": 70.0, "thickness": 0.04, "position": 0.30,
             "feather": 3, "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 5,
             "colors": [(0, 255, 255), (0, 200, 220), (80, 255, 255)],
             "thickness": 0.005, "alpha": 65,
             "angle_range": (55, 80), "seed": 55, "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.28},
    }


@_register("artistic", "cyber_violet",
           "Black body with ultraviolet halftone dots everywhere, glitch + circuit underlayer")
def _cyber_violet(size: int) -> Dict[str, Any]:
    halftone_bright = skin_utils.generate_halftone(size, color=(180, 40, 255), density=0.22)
    halftone_pink = skin_utils.generate_halftone(size, color=(255, 0, 180), density=0.12)
    glitch = skin_utils.generate_glitch_bars(size, color=(130, 0, 220), strength=0.7)
    circuit = skin_utils.generate_circuit_traces(size, color=(160, 30, 255), density="high")
    carbon = skin_utils.generate_carbon_v2(size, base_tone=10, scale=7, seed=77)
    return {
        "role_spec": {
            "hero": {"color": (5, 3, 8), "finish": Finish(type="matte", gloss=25, variation=5)},
            "secondary": {"color": (6, 4, 10), "pattern": carbon, "pattern_opacity": 0.55, "finish": FINISH_CARBON},
            "accent": {"color": (160, 0, 255), "finish": Finish(type="gloss", gloss=220)},
            "darken": {"color": (3, 2, 5), "finish": FINISH_MATTE},
            "neutral": {"color": (4, 3, 6), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": glitch, "opacity": 0.50,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": circuit, "opacity": 0.35,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": halftone_bright, "opacity": 0.85,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": halftone_bright, "opacity": 0.75,
             "blend": "screen", "islands": BODY_SIDES},
            {"type": "overlay", "image": halftone_bright, "opacity": 0.70,
             "blend": "screen", "islands": SECONDARY_ISLANDS},
            {"type": "overlay", "image": halftone_pink, "opacity": 0.50,
             "blend": "screen", "islands": HERO_ISLANDS},
            {"type": "overlay", "image": halftone_pink, "opacity": 0.40,
             "blend": "screen", "islands": BODY_SIDES},
            {"type": "slash", "color": (160, 0, 255), "alpha": 200,
             "angle": 65.0, "thickness": 0.07, "position": 0.30,
             "feather": 5, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=220)},
            {"type": "slash", "color": (255, 0, 180), "alpha": 150,
             "angle": -50.0, "thickness": 0.04, "position": 0.60,
             "feather": 4, "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.30},
    }


@_register("artistic", "acid_yellow",
           "Black with acid yellow neon, topo lines, angular slashes, electric")
def _acid_yellow(size: int) -> Dict[str, Any]:
    glitch = skin_utils.generate_glitch_bars(size, color=(255, 255, 0), strength=0.85)
    topo = skin_utils.generate_topo_lines(size, color=(220, 220, 0), density=14)
    carbon = skin_utils.generate_carbon_v2(size, base_tone=10, scale=6, seed=33)
    return {
        "role_spec": {
            "hero": {"color": (5, 5, 3), "finish": Finish(type="matte", gloss=25, variation=5)},
            "secondary": {"color": (7, 7, 4), "pattern": carbon, "pattern_opacity": 0.65, "finish": FINISH_CARBON},
            "accent": {"color": (255, 255, 0), "finish": Finish(type="gloss", gloss=225)},
            "darken": {"color": (3, 3, 2), "finish": FINISH_MATTE},
            "neutral": {"color": (4, 4, 3), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": glitch, "opacity": 0.70,
             "blend": "screen", "islands": HERO_ISLANDS},
            # Topo contour lines for unique texture
            {"type": "overlay", "image": topo, "opacity": 0.50,
             "blend": "screen", "islands": BODY_SIDES},
            # Sharp angular slashes at steep angles
            {"type": "slash", "color": (255, 255, 0), "alpha": 240,
             "angle": 72.0, "thickness": 0.08, "position": 0.22,
             "feather": 3, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=225)},
            {"type": "slash", "color": (255, 255, 0), "alpha": 200,
             "angle": 75.0, "thickness": 0.04, "position": 0.45,
             "feather": 2, "islands": HERO_ISLANDS},
            {"type": "slash", "color": (200, 200, 0), "alpha": 160,
             "angle": 68.0, "thickness": 0.06, "position": 0.70,
             "feather": 4, "islands": HERO_ISLANDS},
            # Arc swoosh for organic contrast
            {"type": "swoosh", "color": (255, 255, 0), "alpha": 150,
             "thickness": 22, "curve_type": "arc", "flip": True,
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 5,
             "colors": [(255, 255, 0), (200, 200, 0), (255, 220, 60)],
             "thickness": 0.006, "alpha": 65,
             "angle_range": (55, 80), "seed": 33, "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.28},
    }


@_register("artistic", "weaponized_115",
           "Radioactive marble -- swirling neon green and black, BO2 Pack-a-Punch Element 115")
def _weaponized_115(size: int) -> Dict[str, Any]:
    w115 = skin_utils.generate_weaponized_115(
        size,
        green_peak=(40, 240, 70),
        green_mid=(10, 120, 25),
        dark_color=(3, 5, 2),
        vein_freq=8.0,
        turb_amplitude=12.0,
        octaves=7,
        sharpness=0.22,
        seed=115,
    )
    w115_b = skin_utils.generate_weaponized_115(
        size,
        green_peak=(35, 230, 60),
        green_mid=(8, 100, 20),
        dark_color=(2, 4, 2),
        vein_freq=10.0,
        turb_amplitude=11.0,
        octaves=7,
        sharpness=0.24,
        seed=230,
    )
    w115_accent = skin_utils.generate_weaponized_115(
        size,
        green_peak=(50, 245, 80),
        green_mid=(12, 130, 28),
        dark_color=(3, 6, 3),
        vein_freq=12.0,
        turb_amplitude=13.0,
        octaves=7,
        sharpness=0.20,
        seed=330,
    )
    return {
        "role_spec": {
            "hero": {"color": (15, 130, 30), "pattern": w115, "pattern_opacity": 1.0,
                     "finish": Finish(type="satin", gloss=100, variation=8)},
            "secondary": {"color": (10, 100, 22), "pattern": w115_b, "pattern_opacity": 1.0,
                          "finish": Finish(type="satin", gloss=90, variation=6)},
            "accent": {"color": (20, 160, 40), "pattern": w115_accent, "pattern_opacity": 1.0,
                       "finish": Finish(type="satin", gloss=110, variation=6)},
            "darken": {"color": (5, 35, 10), "finish": Finish(type="matte", gloss=50)},
            "neutral": {"color": (8, 55, 15), "finish": Finish(type="satin", gloss=60)},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.18},
    }


@_register("artistic", "weaponized_115_v2",
           "Weaponized 115 with domain warping -- more fluid/chaotic, less periodic")
def _weaponized_115_v2(size: int) -> Dict[str, Any]:
    w115 = skin_utils.generate_weaponized_115(
        size,
        green_peak=(40, 240, 70),
        green_mid=(10, 120, 25),
        dark_color=(3, 5, 2),
        vein_freq=7.0,
        turb_amplitude=13.0,
        octaves=7,
        sharpness=0.22,
        warp_strength=0.35,
        seed=115,
    )
    w115_b = skin_utils.generate_weaponized_115(
        size,
        green_peak=(35, 230, 60),
        green_mid=(8, 100, 20),
        dark_color=(2, 4, 2),
        vein_freq=9.0,
        turb_amplitude=12.0,
        octaves=7,
        sharpness=0.24,
        warp_strength=0.35,
        seed=230,
    )
    w115_accent = skin_utils.generate_weaponized_115(
        size,
        green_peak=(50, 245, 80),
        green_mid=(12, 130, 28),
        dark_color=(3, 6, 3),
        vein_freq=10.0,
        turb_amplitude=14.0,
        octaves=7,
        sharpness=0.20,
        warp_strength=0.40,
        seed=330,
    )
    return {
        "role_spec": {
            "hero": {"color": (15, 130, 30), "pattern": w115, "pattern_opacity": 1.0,
                     "finish": Finish(type="satin", gloss=100, variation=8)},
            "secondary": {"color": (10, 100, 22), "pattern": w115_b, "pattern_opacity": 1.0,
                          "finish": Finish(type="satin", gloss=90, variation=6)},
            "accent": {"color": (20, 160, 40), "pattern": w115_accent, "pattern_opacity": 1.0,
                       "finish": Finish(type="satin", gloss=110, variation=6)},
            "darken": {"color": (5, 35, 10), "finish": Finish(type="matte", gloss=50)},
            "neutral": {"color": (8, 55, 15), "finish": Finish(type="satin", gloss=60)},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.18},
    }


@_register("artistic", "weaponized_115_v3",
           "Weaponized 115 v3 -- finer/smaller marble blobs, domain warped")
def _weaponized_115_v3(size: int) -> Dict[str, Any]:
    w115 = skin_utils.generate_weaponized_115(
        size,
        green_peak=(40, 240, 70),
        green_mid=(10, 120, 25),
        dark_color=(3, 5, 2),
        vein_freq=14.0,
        turb_amplitude=15.0,
        octaves=7,
        sharpness=0.22,
        warp_strength=0.35,
        seed=115,
    )
    w115_b = skin_utils.generate_weaponized_115(
        size,
        green_peak=(35, 230, 60),
        green_mid=(8, 100, 20),
        dark_color=(2, 4, 2),
        vein_freq=16.0,
        turb_amplitude=14.0,
        octaves=7,
        sharpness=0.24,
        warp_strength=0.35,
        seed=230,
    )
    w115_accent = skin_utils.generate_weaponized_115(
        size,
        green_peak=(50, 245, 80),
        green_mid=(12, 130, 28),
        dark_color=(3, 6, 3),
        vein_freq=18.0,
        turb_amplitude=16.0,
        octaves=7,
        sharpness=0.20,
        warp_strength=0.40,
        seed=330,
    )
    return {
        "role_spec": {
            "hero": {"color": (15, 130, 30), "pattern": w115, "pattern_opacity": 1.0,
                     "finish": Finish(type="satin", gloss=100, variation=8)},
            "secondary": {"color": (10, 100, 22), "pattern": w115_b, "pattern_opacity": 1.0,
                          "finish": Finish(type="satin", gloss=90, variation=6)},
            "accent": {"color": (20, 160, 40), "pattern": w115_accent, "pattern_opacity": 1.0,
                       "finish": Finish(type="satin", gloss=110, variation=6)},
            "darken": {"color": (5, 35, 10), "finish": Finish(type="matte", gloss=50)},
            "neutral": {"color": (8, 55, 15), "finish": Finish(type="satin", gloss=60)},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.18},
    }


@_register("artistic", "weaponized_115_v4",
           "Weaponized 115 v4 -- high contrast neon + dark, Illum glow")
def _weaponized_115_v4(size: int) -> Dict[str, Any]:
    w115 = skin_utils.generate_weaponized_115(
        size,
        green_peak=(25, 255, 45),
        green_mid=(10, 170, 25),
        dark_color=(3, 10, 4),
        vein_freq=14.0,
        turb_amplitude=15.0,
        octaves=7,
        sharpness=0.28,
        warp_strength=0.35,
        threshold=0.48,
        hotspot_strength=0.20,
        seed=115,
    )
    w115_b = skin_utils.generate_weaponized_115(
        size,
        green_peak=(20, 245, 40),
        green_mid=(8, 155, 20),
        dark_color=(2, 8, 3),
        vein_freq=16.0,
        turb_amplitude=14.0,
        octaves=7,
        sharpness=0.30,
        warp_strength=0.35,
        threshold=0.46,
        hotspot_strength=0.18,
        seed=230,
    )
    w115_accent = skin_utils.generate_weaponized_115(
        size,
        green_peak=(30, 255, 55),
        green_mid=(12, 180, 30),
        dark_color=(4, 12, 5),
        vein_freq=18.0,
        turb_amplitude=16.0,
        octaves=7,
        sharpness=0.26,
        warp_strength=0.40,
        threshold=0.45,
        hotspot_strength=0.22,
        seed=330,
    )
    return {
        "role_spec": {
            "hero": {"color": (10, 130, 25), "pattern": w115, "pattern_opacity": 1.0,
                     "finish": Finish(type="satin", gloss=110, variation=8)},
            "secondary": {"color": (8, 110, 20), "pattern": w115_b, "pattern_opacity": 1.0,
                          "finish": Finish(type="satin", gloss=100, variation=6)},
            "accent": {"color": (15, 150, 30), "pattern": w115_accent, "pattern_opacity": 1.0,
                       "finish": Finish(type="satin", gloss=120, variation=6)},
            "darken": {"color": (3, 30, 8), "finish": Finish(type="matte", gloss=50)},
            "neutral": {"color": (5, 50, 12), "finish": Finish(type="satin", gloss=60)},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.15},
        "glow": {"color": (15, 255, 30), "intensity": 0.75, "blur": 8, "edge_mode": True},
    }


@_register("artistic", "weaponized_115_v5",
           "Weaponized 115 v5 -- baked glow, per-pixel finish, radioactive depth")
def _weaponized_115_v5(size: int) -> Dict[str, Any]:
    w115 = skin_utils.generate_weaponized_115(
        size,
        green_peak=(25, 255, 45),
        green_mid=(8, 160, 22),
        dark_color=(2, 8, 3),
        vein_freq=14.0,
        turb_amplitude=15.0,
        octaves=7,
        sharpness=0.28,
        warp_strength=0.35,
        threshold=0.48,
        hotspot_strength=0.25,
        seed=115,
    )
    w115_b = skin_utils.generate_weaponized_115(
        size,
        green_peak=(20, 250, 40),
        green_mid=(6, 145, 18),
        dark_color=(2, 6, 2),
        vein_freq=16.0,
        turb_amplitude=14.0,
        octaves=7,
        sharpness=0.30,
        warp_strength=0.35,
        threshold=0.46,
        hotspot_strength=0.22,
        seed=230,
    )
    w115_accent = skin_utils.generate_weaponized_115(
        size,
        green_peak=(30, 255, 55),
        green_mid=(10, 170, 28),
        dark_color=(3, 10, 4),
        vein_freq=18.0,
        turb_amplitude=16.0,
        octaves=7,
        sharpness=0.26,
        warp_strength=0.40,
        threshold=0.45,
        hotspot_strength=0.28,
        seed=330,
    )
    return {
        "role_spec": {
            "hero": {"color": (10, 130, 25), "pattern": w115, "pattern_opacity": 1.0,
                     "finish": Finish(type="matte", gloss=50, variation=5)},
            "secondary": {"color": (8, 110, 20), "pattern": w115_b, "pattern_opacity": 1.0,
                          "finish": Finish(type="matte", gloss=45, variation=5)},
            "accent": {"color": (15, 150, 30), "pattern": w115_accent, "pattern_opacity": 1.0,
                       "finish": Finish(type="matte", gloss=55, variation=5)},
            "darken": {"color": (2, 20, 5), "finish": Finish(type="matte", gloss=40)},
            "neutral": {"color": (4, 40, 10), "finish": Finish(type="matte", gloss=45)},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.005, "bottom_darken": 0.01},
        "prelight": {"strength": 0.28},
    }


@_register("artistic", "marble_luxe",
           "White marble with gold swoosh + dark lower band + luxury pinstripes")
def _marble_luxe(size: int) -> Dict[str, Any]:
    marble = skin_utils.generate_suminagashi(
        size, colors=[(200, 195, 185), (140, 130, 120), (230, 225, 215), (110, 100, 90)],
        seed=21, num_drops=15, rings_per_drop=20, warp_strength=1.2)
    return {
        "role_spec": {
            "hero": {"color": (238, 234, 228), "finish": Finish(type="gloss", gloss=190, variation=5)},
            "secondary": {"color": (30, 28, 25), "finish": Finish(type="satin", gloss=120)},
            "accent": {"color": (195, 165, 50), "finish": Finish(type="metallic", gloss=215, variation=18)},
            "darken": {"color": (15, 14, 12), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 8), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "overlay", "image": marble, "opacity": 0.75,
             "blend": "normal", "islands": HERO_ISLANDS},
            {"type": "swoosh", "color": (195, 165, 50), "alpha": 180,
             "thickness": 20, "curve_type": "arc",
             "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=215, variation=18)},
            {"type": "slash", "color": (30, 28, 25), "alpha": 140,
             "angle": 25.0, "thickness": 0.25, "position": 0.70,
             "feather": 18, "islands": BODY_SIDES},
            {"type": "slash", "color": (195, 165, 50), "alpha": 200,
             "angle": 78.0, "thickness": 0.006, "position": 0.48,
             "feather": 0, "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 4,
             "colors": [(195, 165, 50), (255, 220, 100)],
             "thickness": 0.005, "alpha": 50,
             "angle_range": (68, 85), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.05, "bottom_darken": 0.08},
        "prelight": {"strength": 0.50},
    }


# ===================================================================
# Category 4: Esports / Clean
# ===================================================================

@_register("esports", "clean_split",
           "Red/black diagonal split with white accent line + geometric blocks")
def _clean_split(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (210, 35, 42), "finish": Finish(type="gloss", gloss=195, variation=3)},
            "secondary": {"color": (18, 18, 22), "finish": Finish(type="satin", gloss=110)},
            "accent": {"color": (245, 245, 248), "finish": Finish(type="gloss", gloss=210)},
            "darken": {"color": (8, 6, 8), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 12), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "blocks", "color": (18, 18, 22), "alpha": 230,
             "color2": (18, 18, 22), "alpha2": 180, "style": "split",
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (245, 245, 248), "alpha": 230,
             "angle": 0.0, "thickness": 0.008, "position": 0.42,
             "feather": 1, "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=210)},
            {"type": "swoosh", "color": (210, 35, 42), "alpha": 140,
             "thickness": 35, "curve_type": "arc", "flip": True,
             "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(245, 245, 248), (210, 35, 42)],
             "thickness": 0.005, "alpha": 55,
             "angle_range": (60, 80), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.14},
        "prelight": {"strength": 0.55},
    }


@_register("esports", "stripe_pro",
           "Blue body with bold yellow stripes + black flanking + swoosh accent")
def _stripe_pro(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (0, 55, 140), "finish": Finish(type="gloss", gloss=195, variation=4)},
            "secondary": {"color": (14, 14, 18), "finish": Finish(type="satin", gloss=100)},
            "accent": {"color": (255, 200, 0), "finish": Finish(type="metallic", gloss=210, variation=15)},
            "darken": {"color": (4, 4, 8), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 8, 12), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "slash", "color": (14, 14, 18), "alpha": 140,
             "angle": 25.0, "thickness": 0.28, "position": 0.68,
             "feather": 20, "islands": BODY_SIDES},
            {"type": "slash", "color": (14, 14, 18), "alpha": 210,
             "angle": 90.0, "thickness": 0.025, "position": 0.47,
             "feather": 2, "islands": HERO_ISLANDS},
            {"type": "slash", "color": (255, 200, 0), "alpha": 240,
             "angle": 90.0, "thickness": 0.045, "position": 0.50,
             "feather": 2, "islands": HERO_ISLANDS,
             "finish": Finish(type="metallic", gloss=210, variation=15)},
            {"type": "slash", "color": (14, 14, 18), "alpha": 210,
             "angle": 90.0, "thickness": 0.025, "position": 0.53,
             "feather": 2, "islands": HERO_ISLANDS},
            {"type": "swoosh", "color": (255, 200, 0), "alpha": 130,
             "thickness": 18, "curve_type": "wave",
             "islands": HERO_ISLANDS},
            {"type": "pinstripes", "count": 4,
             "colors": [(255, 200, 0), (14, 14, 18)],
             "thickness": 0.005, "alpha": 55,
             "angle_range": (55, 78), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.14},
        "prelight": {"strength": 0.55},
    }


@_register("esports", "minimal_block",
           "Dark body with cyan band + angular blocks + clean pinstripes")
def _minimal_block(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (22, 22, 28), "finish": Finish(type="matte", gloss=30, variation=4)},
            "secondary": {"color": (32, 32, 38), "finish": Finish(type="satin", gloss=95)},
            "accent": {"color": (0, 180, 255), "finish": Finish(type="gloss", gloss=215)},
            "darken": {"color": (8, 8, 10), "finish": FINISH_MATTE},
            "neutral": {"color": (10, 10, 14), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "band", "color": (0, 180, 255), "alpha": 210,
             "highlight": (150, 230, 255), "highlight_alpha": 180,
             "band_width": 0.16, "angle": -18.0, "offset_y": -0.05,
             "islands": HERO_ISLANDS,
             "finish": Finish(type="gloss", gloss=215)},
            {"type": "blocks", "color": (0, 180, 255), "alpha": 120,
             "color2": (32, 32, 38), "alpha2": 100, "style": "angular",
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (32, 32, 38), "alpha": 130,
             "angle": 25.0, "thickness": 0.22, "position": 0.70,
             "feather": 18, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 3,
             "colors": [(0, 180, 255), (150, 230, 255)],
             "thickness": 0.006, "alpha": 60,
             "angle_range": (60, 78), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.02, "bottom_darken": 0.04},
        "prelight": {"strength": 0.40},
    }


@_register("esports", "gradient_fade",
           "Purple-to-blue OKLCH gradient with white pinstripes + wave swoosh")
def _gradient_fade(size: int) -> Dict[str, Any]:
    return {
        "role_spec": {
            "hero": {"color": (180, 40, 200), "finish": Finish(type="satin", gloss=140, variation=6)},
            "secondary": {"color": (20, 60, 180), "finish": Finish(type="satin", gloss=130)},
            "accent": {"color": (240, 240, 245), "finish": Finish(type="gloss", gloss=205)},
            "darken": {"color": (10, 8, 16), "finish": FINISH_MATTE},
            "neutral": {"color": (8, 8, 14), "finish": FINISH_MATTE},
        },
        "design_layers": [
            {"type": "swoosh", "color": (240, 240, 245), "alpha": 150,
             "thickness": 22, "curve_type": "wave",
             "islands": HERO_ISLANDS},
            {"type": "slash", "color": (240, 240, 245), "alpha": 190,
             "angle": 90.0, "thickness": 0.005, "position": 0.50,
             "feather": 1, "islands": HERO_ISLANDS},
            {"type": "slash", "color": (10, 8, 16), "alpha": 140,
             "angle": 25.0, "thickness": 0.25, "position": 0.70,
             "feather": 18, "islands": BODY_SIDES},
            {"type": "pinstripes", "count": 4,
             "colors": [(240, 240, 245), (180, 40, 200)],
             "thickness": 0.005, "alpha": 55,
             "angle_range": (60, 80), "islands": HERO_ISLANDS},
        ],
        "hero_gradient": {"top_lighten": 0.10, "bottom_darken": 0.15},
        "prelight": {"strength": 0.50},
        "_oklch_fade": True,
    }


# -----------------------------------------------------------------------
# Psychedelic collection
# -----------------------------------------------------------------------

@_register("psychedelic", "psych_warp_grid",
           "Neon grid lines melting under noise distortion on black void")
def _psych_warp_grid(size: int) -> Dict[str, Any]:
    pat_hero = skin_utils.generate_warp_grid(size, grid_count=22, warp_amount=0.18, seed=900)
    pat_sec = skin_utils.generate_warp_grid(size, grid_count=18, warp_amount=0.22, seed=901)
    pat_acc = skin_utils.generate_warp_grid(size, grid_count=26, warp_amount=0.15, seed=902)
    return {
        "role_spec": {
            "hero":      {"color": (3, 3, 3), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=200, variation=5)},
            "secondary": {"color": (3, 3, 3), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=190)},
            "accent":    {"color": (3, 3, 3), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "darken":    {"color": (2, 2, 2), "finish": FINISH_MATTE},
            "neutral":   {"color": (3, 3, 3), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.15},
    }


@_register("psychedelic", "psych_warp_grid_v2",
           "BOLD neon grid -- thicker lines, stronger glow bloom, brighter nodes")
def _psych_warp_grid_v2(size: int) -> Dict[str, Any]:
    kw = dict(warp_amount=0.20, line_base=0.06, line_var=0.14, glow_strength=0.65)
    pat_hero = skin_utils.generate_warp_grid(size, grid_count=20, seed=900, **kw)
    pat_sec = skin_utils.generate_warp_grid(size, grid_count=16, seed=901, **kw)
    pat_acc = skin_utils.generate_warp_grid(size, grid_count=24, seed=902, **kw)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=200)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.10},
    }


def _warp_grid_bold(size, seed, hue_center=None, hue_spread=300.0):
    """Shared builder for bold warp grid variants."""
    kw = dict(warp_amount=0.22, line_base=0.07, line_var=0.16,
              glow_strength=0.80, hue_center=hue_center, hue_spread=hue_spread)
    return skin_utils.generate_warp_grid(size, grid_count=18, seed=seed, **kw)


@_register("psychedelic", "psych_warp_grid_v3",
           "MAX neon grid -- fattest glow, deepest void, rainbow")
def _psych_warp_grid_v3(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 900)
    pat_sec = _warp_grid_bold(size, 901)
    pat_acc = _warp_grid_bold(size, 902)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_cyan",
           "Neon grid -- cyan/teal mono color scheme")
def _psych_warp_cyan(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 910, hue_center=185.0, hue_spread=50.0)
    pat_sec = _warp_grid_bold(size, 911, hue_center=185.0, hue_spread=50.0)
    pat_acc = _warp_grid_bold(size, 912, hue_center=185.0, hue_spread=50.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_magenta",
           "Neon grid -- hot magenta/pink mono color scheme")
def _psych_warp_magenta(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 920, hue_center=310.0, hue_spread=50.0)
    pat_sec = _warp_grid_bold(size, 921, hue_center=310.0, hue_spread=50.0)
    pat_acc = _warp_grid_bold(size, 922, hue_center=310.0, hue_spread=50.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_fire",
           "Neon grid -- fire red/orange/yellow color scheme")
def _psych_warp_fire(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 930, hue_center=25.0, hue_spread=60.0)
    pat_sec = _warp_grid_bold(size, 931, hue_center=25.0, hue_spread=60.0)
    pat_acc = _warp_grid_bold(size, 932, hue_center=25.0, hue_spread=60.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_toxic",
           "Neon grid -- toxic green/lime radioactive color scheme")
def _psych_warp_toxic(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 940, hue_center=120.0, hue_spread=45.0)
    pat_sec = _warp_grid_bold(size, 941, hue_center=120.0, hue_spread=45.0)
    pat_acc = _warp_grid_bold(size, 942, hue_center=120.0, hue_spread=45.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_ultraviolet",
           "Neon grid -- deep purple/violet blacklight color scheme")
def _psych_warp_ultraviolet(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 950, hue_center=270.0, hue_spread=50.0)
    pat_sec = _warp_grid_bold(size, 951, hue_center=270.0, hue_spread=50.0)
    pat_acc = _warp_grid_bold(size, 952, hue_center=270.0, hue_spread=50.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }


@_register("psychedelic", "psych_warp_gold",
           "Neon grid -- warm amber/gold electric color scheme")
def _psych_warp_gold(size: int) -> Dict[str, Any]:
    pat_hero = _warp_grid_bold(size, 960, hue_center=50.0, hue_spread=40.0)
    pat_sec = _warp_grid_bold(size, 961, hue_center=50.0, hue_spread=40.0)
    pat_acc = _warp_grid_bold(size, 962, hue_center=50.0, hue_spread=40.0)
    return {
        "role_spec": {
            "hero":      {"color": (1, 1, 1), "pattern": pat_hero, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=220, variation=5)},
            "secondary": {"color": (1, 1, 1), "pattern": pat_sec, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=210)},
            "accent":    {"color": (1, 1, 1), "pattern": pat_acc, "pattern_opacity": 1.0,
                          "pattern_blend": "normal", "finish": Finish(type="gloss", gloss=230)},
            "darken":    {"color": (1, 1, 1), "finish": FINISH_MATTE},
            "neutral":   {"color": (1, 1, 1), "finish": FINISH_MATTE},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.01},
        "prelight": {"strength": 0.08},
    }
