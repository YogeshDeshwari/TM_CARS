"""
Shared procedural drawing routines for the gore skin family:
blood splatters, drips, veins, scratches, skulls, raven wings.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw


# -----------------------------------------------------------------------
# Raven-wing ProjShad
# -----------------------------------------------------------------------

def make_raven_projshad(size=512, style="spread"):
    """Build a raven-wing shadow projection image.

    The car shadow will cast a bird silhouette on the ground.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2

    body_w, body_h = int(size * 0.08), int(size * 0.30)
    draw.ellipse([cx - body_w, cy - body_h, cx + body_w, cy + body_h],
                 fill=(255, 255, 255, 220))

    head_r = int(size * 0.055)
    head_cy = cy - body_h + head_r // 2
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 fill=(255, 255, 255, 220))

    beak_len = int(size * 0.05)
    draw.polygon([
        (cx, head_cy - head_r - beak_len),
        (cx - head_r // 3, head_cy - head_r + 2),
        (cx + head_r // 3, head_cy - head_r + 2),
    ], fill=(255, 255, 255, 200))

    def draw_wing(draw, cx, cy, side, size):
        s = 1 if side == "right" else -1
        tip_x = cx + s * int(size * 0.46)
        tip_y = cy - int(size * 0.05)

        pts = [
            (cx + s * int(size * 0.04), cy - int(size * 0.15)),
            (cx + s * int(size * 0.15), cy - int(size * 0.25)),
            (cx + s * int(size * 0.28), cy - int(size * 0.22)),
            (cx + s * int(size * 0.38), cy - int(size * 0.15)),
            (tip_x, tip_y),
            (cx + s * int(size * 0.42), cy + int(size * 0.05)),
            (cx + s * int(size * 0.35), cy + int(size * 0.12)),
            (cx + s * int(size * 0.25), cy + int(size * 0.15)),
            (cx + s * int(size * 0.15), cy + int(size * 0.13)),
            (cx + s * int(size * 0.06), cy + int(size * 0.08)),
        ]
        draw.polygon(pts, fill=(255, 255, 255, 200))

        feather_pts = []
        for i in range(4):
            fx = cx + s * int(size * (0.30 + i * 0.045))
            fy_top = cy + int(size * (0.10 + i * 0.02))
            fy_bot = cy + int(size * (0.18 + i * 0.03))
            draw.line([(fx, fy_top), (fx + s * int(size * 0.04), fy_bot)],
                      fill=(255, 255, 255, 160), width=max(1, size // 200))

    draw_wing(draw, cx, cy, "left", size)
    draw_wing(draw, cx, cy, "right", size)

    tail_pts = [
        (cx, cy + body_h),
        (cx - int(size * 0.06), cy + body_h + int(size * 0.10)),
        (cx, cy + body_h + int(size * 0.08)),
        (cx + int(size * 0.06), cy + body_h + int(size * 0.10)),
    ]
    draw.polygon(tail_pts, fill=(255, 255, 255, 190))

    from scipy.ndimage import gaussian_filter as gf
    arr = np.array(img, dtype=np.float64)
    for ch in range(4):
        arr[:, :, ch] = gf(arr[:, :, ch], sigma=3.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# -----------------------------------------------------------------------
# Blood splatters
# -----------------------------------------------------------------------

def draw_blood_splatters(size, n_splatters, color, seed=42):
    """Generate a blood splatter layer with drip trails."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cr, cg, cb = color[:3]

    for _ in range(n_splatters):
        cx = rng.integers(20, size - 20)
        cy = rng.integers(20, size - 20)
        r = rng.integers(8, 45)
        alpha = rng.integers(140, 230)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(cr, cg, cb, alpha))

        n_drops = rng.integers(3, 10)
        for _ in range(n_drops):
            dx = rng.integers(-r, r)
            dy = rng.integers(-r // 2, r)
            dr = rng.integers(2, max(3, r // 3))
            da = rng.integers(100, 200)
            draw.ellipse([cx + dx - dr, cy + dy - dr,
                          cx + dx + dr, cy + dy + dr],
                         fill=(cr, cg, cb, da))

        if rng.random() < 0.6:
            drip_len = rng.integers(30, 200)
            drip_w = rng.integers(2, max(3, r // 3))
            drip_x = cx + rng.integers(-r // 2, r // 2)
            end_y = min(size - 1, cy + r + drip_len)
            drip_alpha = rng.integers(120, 200)
            for dy in range(0, end_y - cy - r + 1):
                fade = max(40, drip_alpha - dy * drip_alpha // drip_len)
                yy = cy + r + dy
                if yy >= size:
                    break
                draw.line([(drip_x, yy), (drip_x + drip_w, yy)],
                          fill=(cr, cg, cb, fade))

    return img


# -----------------------------------------------------------------------
# Claw scratches
# -----------------------------------------------------------------------

def draw_claw_scratches(size, n_scratches, color, seed=42):
    """Draw parallel claw rake marks."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cr, cg, cb = color[:3]

    for _ in range(n_scratches):
        cx = rng.integers(100, size - 100)
        cy = rng.integers(100, size - 100)
        angle = rng.uniform(-0.6, 0.6)
        n_lines = rng.integers(3, 6)
        length = rng.integers(80, 300)
        spacing = rng.integers(6, 15)

        for i in range(n_lines):
            offset = (i - n_lines // 2) * spacing
            x0 = cx + int(offset * np.cos(angle + np.pi / 2))
            y0 = cy + int(offset * np.sin(angle + np.pi / 2))
            x1 = x0 + int(length * np.cos(angle))
            y1 = y0 + int(length * np.sin(angle))
            w = rng.integers(1, 4)
            alpha = rng.integers(150, 230)
            draw.line([(x0, y0), (x1, y1)], fill=(cr, cg, cb, alpha), width=w)

    return img


# -----------------------------------------------------------------------
# Organic veins / capillaries
# -----------------------------------------------------------------------

def draw_veins(size, n_seeds, color, max_depth=6, seed=42):
    """Draw branching organic vein network."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cr, cg, cb = color[:3]

    def branch(x, y, angle, length, width, depth, alpha):
        if depth <= 0 or length < 5 or width < 1:
            return
        x1 = x + length * np.cos(angle)
        y1 = y + length * np.sin(angle)
        if not (0 <= x1 < size and 0 <= y1 < size):
            return
        draw.line([(int(x), int(y)), (int(x1), int(y1))],
                  fill=(cr, cg, cb, int(alpha)), width=max(1, int(width)))

        wobble = rng.uniform(-0.4, 0.4)
        branch(x1, y1, angle + wobble,
               length * rng.uniform(0.7, 0.9),
               width * 0.8, depth - 1, alpha * 0.85)

        if rng.random() < 0.55:
            fork_angle = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            branch(x1, y1, angle + fork_angle,
                   length * rng.uniform(0.5, 0.7),
                   width * 0.6, depth - 1, alpha * 0.75)

    for _ in range(n_seeds):
        sx = rng.integers(50, size - 50)
        sy = rng.integers(50, size - 50)
        sa = rng.uniform(0, 2 * np.pi)
        sl = rng.uniform(60, 150)
        sw = rng.uniform(3, 6)
        branch(sx, sy, sa, sl, sw, max_depth, 200)

    return img


# -----------------------------------------------------------------------
# Procedural skull
# -----------------------------------------------------------------------

def draw_skull(size, color=(220, 210, 190), outline=(40, 0, 0)):
    """Draw a simple frontal skull at the center of a square image.

    Returns an RGBA image of the given size.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    cr, cg, cb = color
    olr, olg, olb = outline

    skull_w = int(size * 0.38)
    skull_h = int(size * 0.42)
    top = cy - int(size * 0.28)
    draw.ellipse([cx - skull_w, top, cx + skull_w, top + skull_h * 2],
                 fill=(cr, cg, cb, 220), outline=(olr, olg, olb, 200), width=2)

    jaw_w = int(skull_w * 0.75)
    jaw_top = top + int(skull_h * 1.4)
    jaw_h = int(skull_h * 0.55)
    draw.rounded_rectangle(
        [cx - jaw_w, jaw_top, cx + jaw_w, jaw_top + jaw_h],
        radius=int(jaw_w * 0.3),
        fill=(cr, cg, cb, 200), outline=(olr, olg, olb, 180), width=2)

    eye_w = int(skull_w * 0.35)
    eye_h = int(skull_h * 0.30)
    eye_y = top + int(skull_h * 0.55)
    eye_sep = int(skull_w * 0.25)
    for sx in [-1, 1]:
        ex = cx + sx * eye_sep
        draw.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h],
                     fill=(olr, olg, olb, 230))

    nose_y = eye_y + eye_h + int(skull_h * 0.10)
    nose_w = int(skull_w * 0.12)
    nose_h = int(skull_h * 0.18)
    draw.polygon([
        (cx, nose_y),
        (cx - nose_w, nose_y + nose_h),
        (cx + nose_w, nose_y + nose_h),
    ], fill=(olr, olg, olb, 210))

    teeth_y = jaw_top + int(jaw_h * 0.15)
    n_teeth = 6
    teeth_w = jaw_w * 2 // (n_teeth + 1)
    for i in range(n_teeth):
        tx = cx - jaw_w + teeth_w * (i + 1)
        draw.rectangle([tx - teeth_w // 3, teeth_y,
                         tx + teeth_w // 3, teeth_y + int(jaw_h * 0.35)],
                        fill=(cr, cg, cb, 180),
                        outline=(olr, olg, olb, 160), width=1)
    draw.line([(cx - jaw_w + teeth_w // 2, teeth_y),
               (cx + jaw_w - teeth_w // 2, teeth_y)],
              fill=(olr, olg, olb, 180), width=2)

    return img
