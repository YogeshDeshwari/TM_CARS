"""
Tire and wheel customization for TMNF/TMUF Details.dds textures.

Paints tire sidewall brand text (arc-following), accent color bands,
custom spoke patterns, and part-specific tints onto the Details.dds
UV layout.

All internal coordinates reference the 4096x4096 UV atlas and are
scaled automatically for any actual Details.dds resolution.

Usage:
    from tire_customizer import customize_details
    details_img = customize_details(
        base_details_img,
        brand_text="BRIDGESTONE",
        model_text="POTENZA",
        accent_color=(0, 200, 210),
    )
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path
from typing import Tuple, Optional


# -- UV reference coordinates at 4096 scale --------------------------------

_REF = 4096

# Wheel/tire circular island (measured from MINA Aqua + base CH_2026)
_WCX, _WCY = 1161, 506
_TIRE_OUTER_R = 498
_TIRE_INNER_R = 380
_RIM_OUTER_R = 335
_HUB_R = 48
_TEXT_R_INNER = 395
_TEXT_R_OUTER = 475
_ACCENT_BAND_W = 16

# Rectangular UV strips (tire tread and sidewall)
_TREAD_X0, _TREAD_Y0, _TREAD_X1, _TREAD_Y1 = 0, 0, 640, 960
_SWALL_X0, _SWALL_Y0, _SWALL_X1, _SWALL_Y1 = 50, 1430, 1380, 1560


def _s(val, size):
    """Scale a 4096-reference value to actual texture size."""
    return int(round(val * size / _REF))


def _load_tire_font(size_px):
    for p in [
        Path.home() / "Library/Fonts/DejaVuSansCondensed-Bold.ttf",
        Path.home() / "Library/Fonts/DejaVuSans-Bold.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]:
        if p.exists():
            return ImageFont.truetype(str(p), size_px)
    return ImageFont.load_default()


# -- Straight text rendering -----------------------------------------------

def _render_text_strip(text, font_size, color=(255, 255, 255, 255)):
    """Render text as a tight horizontal RGBA image."""
    font = _load_tire_font(font_size)
    dummy = Image.new("RGBA", (1, 1))
    dd = ImageDraw.Draw(dummy)

    bbox_full = dd.textbbox((0, 0), text, font=font)
    tw = bbox_full[2] - bbox_full[0]
    th = bbox_full[3] - bbox_full[1]
    pad = font_size // 3
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox_full[0], pad - bbox_full[1]), text, fill=color, font=font)

    bb = img.getbbox()
    if bb:
        return img.crop(bb)
    return img


# -- Polar warp engine -----------------------------------------------------

def _warp_text_to_arc(
    text_img_arr,
    canvas_size,
    cx, cy,
    r_inner, r_outer,
    angle_start_deg, angle_end_deg,
    flip_radial=False,
):
    """Map a horizontal text image onto a circular arc.

    Angles are in degrees, measured clockwise from 3-o'clock
    (image-coordinate convention).

    flip_radial: if True, bottom-of-text faces the center (for the
    lower arc where letters face outward = away from center).
    """
    h_text, w_text = text_img_arr.shape[:2]
    if w_text == 0 or h_text == 0:
        return np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    r_max = r_outer + 4
    bx0 = max(0, cx - r_max)
    by0 = max(0, cy - r_max)
    bx1 = min(canvas_size, cx + r_max + 1)
    by1 = min(canvas_size, cy + r_max + 1)
    bw, bh = bx1 - bx0, by1 - by0

    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    dx = (xx - cx).astype(np.float64)
    dy = (yy - cy).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.degrees(np.arctan2(dy, dx)) % 360.0

    a_s = angle_start_deg % 360.0
    a_e = angle_end_deg % 360.0

    if a_s <= a_e:
        in_angle = (theta >= a_s) & (theta <= a_e)
        theta_local = theta - a_s
        total_angle = a_e - a_s
    else:
        in_angle = (theta >= a_s) | (theta <= a_e)
        theta_local = (theta - a_s) % 360.0
        total_angle = (a_e - a_s) % 360.0

    if total_angle < 0.01:
        return np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    in_radius = (r >= r_inner) & (r <= r_outer)
    valid = in_angle & in_radius

    u = (theta_local / total_angle * (w_text - 1)).astype(np.int32)
    if flip_radial:
        v = ((r - r_inner) / (r_outer - r_inner) * (h_text - 1)).astype(np.int32)
    else:
        v = ((r_outer - r) / (r_outer - r_inner) * (h_text - 1)).astype(np.int32)

    u = np.clip(u, 0, w_text - 1)
    v = np.clip(v, 0, h_text - 1)

    crop = np.zeros((bh, bw, 4), dtype=np.uint8)
    crop[valid] = text_img_arr[v[valid], u[valid]]

    output = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    output[by0:by1, bx0:bx1] = crop
    return output


# -- Spoke drawing ---------------------------------------------------------

def _draw_spokes(
    draw,
    cx, cy,
    inner_r, outer_r,
    n_spokes, spoke_width_base,
    color,
):
    """Draw tapered spokes as filled polygons."""
    tip_ratio = 0.55
    for i in range(n_spokes):
        angle = 2 * math.pi * i / n_spokes
        half_base = spoke_width_base / 2.0
        half_tip = half_base * tip_ratio

        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)

        pts = [
            (cx + inner_r * dir_x + half_base * perp_x,
             cy + inner_r * dir_y + half_base * perp_y),
            (cx + outer_r * dir_x + half_tip * perp_x,
             cy + outer_r * dir_y + half_tip * perp_y),
            (cx + outer_r * dir_x - half_tip * perp_x,
             cy + outer_r * dir_y - half_tip * perp_y),
            (cx + inner_r * dir_x - half_base * perp_x,
             cy + inner_r * dir_y - half_base * perp_y),
        ]
        draw.polygon([(int(x), int(y)) for x, y in pts], fill=color)


# -- Main entry point ------------------------------------------------------

def customize_details(
    details_img,
    *,
    brand_text="BRIDGESTONE",
    model_text="POTENZA",
    text_color=(220, 220, 225, 240),
    accent_color=(0, 200, 210),
    rubber_color=(18, 18, 22),
    rubber_alpha=5,
    spoke_color=(200, 203, 210),
    spoke_bg_color=(75, 78, 85),
    spoke_count=12,
    spoke_width=28,
    hub_color=(30, 30, 38),
    hub_alpha=35,
    rim_lip_color=(185, 185, 192),
    rim_gap_color=(4, 6, 10),
    caliper_color=None,
    lugnut_color=None,
    tread_color=None,
):
    """Customize wheel/tire regions of a Details.dds image.

    Parameters
    ----------
    details_img : PIL.Image.Image
        Base Details.dds loaded as RGBA.
    brand_text / model_text : str
        Sidewall arc text (upper and lower halves).
    text_color : 4-tuple
        RGBA for the sidewall text.
    accent_color : 3-tuple
        RGB for the thin band on the tire outer shoulder.
    rubber_color : 3-tuple
        RGB for tire rubber (the dark matte surface).
    rubber_alpha : int
        Alpha for rubber (low = shiny, high = matte in TMNF).
    spoke_color : 3-tuple
        RGB for the wheel spokes.
    spoke_bg_color : 3-tuple
        RGB for the area between spokes.
    spoke_count : int
        Number of spokes (10-16 typical).
    spoke_width : int
        Spoke width at hub end, in 4096-reference pixels.
    hub_color : 3-tuple
        RGB for center hub cap.
    hub_alpha : int
        Alpha for hub (higher = more matte).
    rim_lip_color : 3-tuple
        RGB for thin rim lip ring.
    rim_gap_color : 3-tuple
        RGB for dark gap between rim barrel and tire bead.
    caliper_color / lugnut_color : 3-tuple or None
        If set, tint the brake caliper / lug nut.
    tread_color : 3-tuple or None
        If set, override the rectangular tread/sidewall UV strips.
        Defaults to rubber_color.

    Returns
    -------
    PIL.Image.Image
        Customized RGBA Details.dds.
    """
    dw, dh = details_img.size
    sz = max(dw, dh)
    result = details_img.copy()

    cx = _s(_WCX, sz)
    cy = _s(_WCY, sz)
    tire_out = _s(_TIRE_OUTER_R, sz)
    tire_in = _s(_TIRE_INNER_R, sz)
    rim_out = _s(_RIM_OUTER_R, sz)
    hub_r = _s(_HUB_R, sz)
    sw = _s(spoke_width, sz)
    text_ri = _s(_TEXT_R_INNER, sz)
    text_ro = _s(_TEXT_R_OUTER, sz)
    accent_w = _s(_ACCENT_BAND_W, sz)

    # ---- 1. Circular wheel island: build from scratch --------------------

    # Coordinate grids (only the wheel bounding box for speed)
    margin = tire_out + 8
    bx0, by0 = max(0, cx - margin), max(0, cy - margin)
    bx1, by1 = min(dw, cx + margin + 1), min(dh, cy + margin + 1)

    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    dx = (xx - cx).astype(np.float64)
    dy = (yy - cy).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)

    arr = np.array(result)

    # Spoke background fill (between spokes, inside rim)
    rim_fill = (r <= rim_out) & (r > hub_r)
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            rim_fill, spoke_bg_color[ch], arr[by0:by1, bx0:bx1, ch])
    arr[by0:by1, bx0:bx1, 3] = np.where(rim_fill, 3, arr[by0:by1, bx0:bx1, 3])

    # Hub center
    hub_mask = r <= hub_r
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            hub_mask, hub_color[ch], arr[by0:by1, bx0:bx1, ch])
    arr[by0:by1, bx0:bx1, 3] = np.where(hub_mask, hub_alpha, arr[by0:by1, bx0:bx1, 3])

    # Rim gap (dark ring between rim barrel and tire bead)
    gap_mask = (r > rim_out) & (r < tire_in)
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            gap_mask, rim_gap_color[ch], arr[by0:by1, bx0:bx1, ch])
    arr[by0:by1, bx0:bx1, 3] = np.where(gap_mask, 2, arr[by0:by1, bx0:bx1, 3])

    # Rim lip (thin bright ring at the outer edge of the rim)
    lip_w = max(3, _s(8, sz))
    lip_mask = (r >= rim_out - lip_w) & (r <= rim_out)
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            lip_mask, rim_lip_color[ch], arr[by0:by1, bx0:bx1, ch])

    # Tire rubber
    tire_mask = (r >= tire_in) & (r <= tire_out)
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            tire_mask, rubber_color[ch], arr[by0:by1, bx0:bx1, ch])
    arr[by0:by1, bx0:bx1, 3] = np.where(
        tire_mask, rubber_alpha, arr[by0:by1, bx0:bx1, 3])

    # Accent band on outer tire shoulder
    band_inner = tire_out - accent_w
    band_mask = (r >= band_inner) & (r <= tire_out)
    for ch in range(3):
        arr[by0:by1, bx0:bx1, ch] = np.where(
            band_mask, accent_color[ch], arr[by0:by1, bx0:bx1, ch])
    arr[by0:by1, bx0:bx1, 3] = np.where(band_mask, 2, arr[by0:by1, bx0:bx1, 3])

    result = Image.fromarray(arr, "RGBA")

    # ---- 2. Draw spokes on top (as PIL polygons for clean edges) ---------

    spoke_layer = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spoke_layer)
    fill = (*spoke_color, 255)
    _draw_spokes(sd, cx, cy, hub_r + 2, rim_out - 4, spoke_count, sw, fill)
    result = Image.alpha_composite(result, spoke_layer)

    # ---- 3. Arc text on tire sidewall ------------------------------------

    font_sz = max(8, _s(50, sz))
    if brand_text:
        brand_strip = _render_text_strip(brand_text, font_sz, text_color)
        brand_arr = np.array(brand_strip)
        # Upper arc: ~210 deg to ~330 deg (wrapping over the top)
        arc_upper = _warp_text_to_arc(
            brand_arr, max(dw, dh),
            cx, cy, text_ri, text_ro,
            angle_start_deg=210, angle_end_deg=330,
            flip_radial=False,
        )
        arc_layer = Image.fromarray(arc_upper, "RGBA")
        result = Image.alpha_composite(result, arc_layer)

    if model_text:
        model_strip = _render_text_strip(model_text, font_sz, text_color)
        model_arr = np.array(model_strip)
        model_arr = model_arr[:, ::-1, :]  # flip horizontally so text reads L-to-R on bottom
        # Lower arc: ~30 deg to ~150 deg (wrapping under the bottom)
        arc_lower = _warp_text_to_arc(
            model_arr, max(dw, dh),
            cx, cy, text_ri, text_ro,
            angle_start_deg=30, angle_end_deg=150,
            flip_radial=True,
        )
        arc_layer = Image.fromarray(arc_lower, "RGBA")
        result = Image.alpha_composite(result, arc_layer)

    # ---- 4. Rectangular tread and sidewall strips ------------------------

    tc = tread_color or rubber_color
    arr2 = np.array(result)

    tx0, ty0 = _s(_TREAD_X0, sz), _s(_TREAD_Y0, sz)
    tx1, ty1 = _s(_TREAD_X1, sz), _s(_TREAD_Y1, sz)
    arr2[ty0:ty1, tx0:tx1, 0] = tc[0]
    arr2[ty0:ty1, tx0:tx1, 1] = tc[1]
    arr2[ty0:ty1, tx0:tx1, 2] = tc[2]
    arr2[ty0:ty1, tx0:tx1, 3] = rubber_alpha

    sx0, sy0 = _s(_SWALL_X0, sz), _s(_SWALL_Y0, sz)
    sx1, sy1 = _s(_SWALL_X1, sz), _s(_SWALL_Y1, sz)
    arr2[sy0:sy1, sx0:sx1, 0] = tc[0]
    arr2[sy0:sy1, sx0:sx1, 1] = tc[1]
    arr2[sy0:sy1, sx0:sx1, 2] = tc[2]
    arr2[sy0:sy1, sx0:sx1, 3] = rubber_alpha

    result = Image.fromarray(arr2, "RGBA")

    # ---- 5. Optional caliper / lug nut tint ------------------------------

    if caliper_color is not None:
        _tint_region(result, _s(100, sz), _s(1560, sz),
                     _s(500, sz), _s(1820, sz), caliper_color)

    if lugnut_color is not None:
        _tint_circular(result, _s(380, sz), _s(1020, sz),
                       _s(80, sz), lugnut_color)

    print(f"  Tire customized: {spoke_count}-spoke, "
          f"accent=({accent_color[0]},{accent_color[1]},{accent_color[2]}), "
          f"text=\"{brand_text}\" / \"{model_text}\"")
    return result


def _tint_region(img, x0, y0, x1, y1, color):
    """Multiply-tint a rectangular region toward a target color."""
    arr = np.array(img)
    region = arr[y0:y1, x0:x1].astype(np.float64)
    gray = (region[:, :, 0] * 0.299 +
            region[:, :, 1] * 0.587 +
            region[:, :, 2] * 0.114)
    for ch in range(3):
        region[:, :, ch] = gray * (color[ch] / 255.0)
    arr[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    img.paste(Image.fromarray(arr, "RGBA"))


def _tint_circular(img, cx, cy, radius, color):
    """Multiply-tint a circular region toward a target color."""
    arr = np.array(img)
    sz_y, sz_x = arr.shape[:2]
    y0, y1 = max(0, cy - radius), min(sz_y, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(sz_x, cx + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius
    region = arr[y0:y1, x0:x1].astype(np.float64)
    gray = (region[:, :, 0] * 0.299 +
            region[:, :, 1] * 0.587 +
            region[:, :, 2] * 0.114)
    for ch in range(3):
        region[:, :, ch] = np.where(mask, gray * (color[ch] / 255.0), region[:, :, ch])
    arr[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    img.paste(Image.fromarray(arr, "RGBA"))
