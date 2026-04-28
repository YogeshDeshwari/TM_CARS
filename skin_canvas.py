"""
SkinCanvas -- Photoshop-like painting API for TrackMania car skins.

Instead of working through presets and procedural generators, SkinCanvas
gives you direct pixel control over the 2048x2048 texture.  Write a skin
the same way you'd script Photoshop: fill regions, draw shapes, place text,
control material finish per-pixel.

Usage:

    from skin_canvas import SkinCanvas

    c = SkinCanvas()
    c.fill((10, 10, 10))                          # black base
    c.fill_island(3, (205, 150, 25))               # gold nose
    c.contour_lines(3, (10, 10, 10), spacing=17)   # black contour lines on nose
    c.text("AURA", island=7, color=(225, 175, 42)) # text on rear
    c.set_finish_island(3, "metallic")              # metallic nose
    c.save("my_skin")                               # -> out/my_skin.zip
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.ndimage import distance_transform_edt

from car_geometry import CarGeometry, ColorRole
from layer_stack import (
    Finish, FINISH_MATTE, FINISH_SATIN, FINISH_GLOSS,
    FINISH_METALLIC, FINISH_CARBON, FINISH_BRUSHED,
)

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

UV_ATLAS_JSON = Path("assets/uv_atlas/standard_stadium_islands_2048.json")
UV_DIAG_PNG = Path("assets/uv_atlas/diagnostics_2048.png")

_FINISH_MAP = {
    "matte":    FINISH_MATTE,
    "satin":    FINISH_SATIN,
    "gloss":    FINISH_GLOSS,
    "metallic": FINISH_METALLIC,
    "carbon":   FINISH_CARBON,
    "brushed":  FINISH_BRUSHED,
}

# ---------------------------------------------------------------------------
# Layer (internal)
# ---------------------------------------------------------------------------

class _Layer:
    __slots__ = ("image", "mask", "blend", "opacity", "name")

    def __init__(self, image: Image.Image, mask: Optional[Image.Image],
                 blend: str, opacity: float, name: str):
        self.image = image
        self.mask = mask
        self.blend = blend
        self.opacity = opacity
        self.name = name


# ---------------------------------------------------------------------------
# SkinCanvas
# ---------------------------------------------------------------------------

class SkinCanvas:
    """Direct-painting API for a 2048x2048 car skin texture."""

    def __init__(self, size: int = 2048):
        self.size = size
        self._canvas = Image.new("RGBA", (size, size), (0, 0, 0, 255))
        self._finish_map = Image.new("L", (size, size), 0x8E)
        self._layers: List[_Layer] = []
        self._active_layer: Optional[_Layer] = None

        self._geo: Optional[CarGeometry] = None
        self._island_masks: Dict[int, Image.Image] = {}
        self._role_masks: Dict[str, Image.Image] = {}
        self._atlas_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Geometry loading
    # ------------------------------------------------------------------

    def _ensure_geo(self):
        if self._geo is not None:
            return
        if not UV_ATLAS_JSON.exists() or not UV_DIAG_PNG.exists():
            raise FileNotFoundError(
                f"UV atlas not found at {UV_ATLAS_JSON} / {UV_DIAG_PNG}."
            )
        self._geo = CarGeometry.from_json_file(str(UV_ATLAS_JSON))
        with open(UV_ATLAS_JSON) as f:
            self._atlas_data = json.load(f)
        self._build_masks()

    def _build_masks(self):
        from collections import Counter
        diag = np.array(Image.open(UV_DIAG_PNG).convert("RGB"))
        BG = np.array([8, 8, 12])

        for isl in self._atlas_data["islands"]:
            iid = isl["id"]
            x0, y0, x1, y1 = isl["bbox"]
            cx, cy = isl["center"]

            center_col = diag[cy, cx]
            if np.all(np.abs(center_col.astype(int) - BG) < 10):
                region = diag[y0:y1, x0:x1]
                non_bg = ~np.all(np.abs(region.astype(int) - BG) < 10, axis=2)
                if non_bg.any():
                    pixels = region[non_bg][:500]
                    colors = [tuple(c) for c in pixels]
                    center_col = np.array(Counter(colors).most_common(1)[0][0])
                else:
                    continue

            diff = np.abs(diag.astype(int) - center_col.astype(int)).sum(axis=2)
            mask_arr = (diff < 30).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_arr, mode="L")
            if self.size != 2048:
                mask_img = mask_img.resize(
                    (self.size, self.size), Image.Resampling.NEAREST
                )
            self._island_masks[iid] = mask_img

        for role in ColorRole:
            ids = self._geo.get_islands_by_role(role)
            combined = Image.new("L", (self.size, self.size), 0)
            for iid in ids:
                if iid in self._island_masks:
                    combined = ImageChops.lighter(combined, self._island_masks[iid])
            self._role_masks[role.value] = combined

    def island_mask(self, island_id: int) -> Image.Image:
        """Get the L-mode pixel mask for a UV island."""
        self._ensure_geo()
        return self._island_masks.get(
            island_id, Image.new("L", (self.size, self.size), 0)
        )

    def role_mask(self, role: str) -> Image.Image:
        """Get combined mask for a color role (hero/secondary/accent/darken/neutral)."""
        self._ensure_geo()
        return self._role_masks.get(
            role, Image.new("L", (self.size, self.size), 0)
        )

    def islands_mask(self, ids: Sequence[int]) -> Image.Image:
        """Combine masks for multiple island IDs."""
        self._ensure_geo()
        combined = Image.new("L", (self.size, self.size), 0)
        for iid in ids:
            m = self._island_masks.get(iid)
            if m is not None:
                combined = ImageChops.lighter(combined, m)
        return combined

    def island_bbox(self, island_id: int) -> Tuple[int, int, int, int]:
        """Get bounding box (x0, y0, x1, y1) for an island."""
        self._ensure_geo()
        info = self._geo.islands.get(island_id)
        if info is None:
            return (0, 0, self.size, self.size)
        return info.bbox

    # ------------------------------------------------------------------
    # Core painting: direct canvas operations
    # ------------------------------------------------------------------

    def _rgba(self, color: Color) -> Tuple[int, int, int, int]:
        if len(color) == 3:
            return color + (255,)
        return color

    def fill(self, color: Color):
        """Fill the entire canvas with a solid color."""
        self._canvas = Image.new("RGBA", (self.size, self.size), self._rgba(color))

    def fill_rect(self, bbox: Tuple[int, int, int, int], color: Color):
        """Fill a rectangle (x0, y0, x1, y1) with a color."""
        draw = ImageDraw.Draw(self._canvas)
        draw.rectangle(bbox, fill=self._rgba(color))

    def fill_polygon(self, points: List[Tuple[int, int]], color: Color):
        """Fill a polygon defined by a list of (x, y) points."""
        draw = ImageDraw.Draw(self._canvas)
        draw.polygon(points, fill=self._rgba(color))

    def fill_circle(self, center: Tuple[int, int], radius: int, color: Color):
        """Fill a circle."""
        cx, cy = center
        draw = ImageDraw.Draw(self._canvas)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=self._rgba(color),
        )

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int],
                  color: Color, width: int = 1):
        """Draw a line between two points."""
        draw = ImageDraw.Draw(self._canvas)
        draw.line([start, end], fill=self._rgba(color), width=width)

    def draw_polyline(self, points: List[Tuple[int, int]], color: Color,
                      width: int = 1, closed: bool = False):
        """Draw connected line segments."""
        draw = ImageDraw.Draw(self._canvas)
        if closed and len(points) > 2:
            pts = list(points) + [points[0]]
        else:
            pts = list(points)
        draw.line(pts, fill=self._rgba(color), width=width, joint="curve")

    # ------------------------------------------------------------------
    # Island-aware painting
    # ------------------------------------------------------------------

    def fill_island(self, island_id: int, color: Color):
        """Fill a specific UV island with a solid color."""
        mask = self.island_mask(island_id)
        fill = Image.new("RGBA", (self.size, self.size), self._rgba(color))
        self._canvas = Image.composite(fill, self._canvas, mask)

    def fill_islands(self, island_ids: Sequence[int], color: Color):
        """Fill multiple islands with the same color."""
        mask = self.islands_mask(island_ids)
        fill = Image.new("RGBA", (self.size, self.size), self._rgba(color))
        self._canvas = Image.composite(fill, self._canvas, mask)

    def fill_role(self, role: str, color: Color):
        """Fill all islands of a given role (hero/secondary/accent/darken/neutral)."""
        mask = self.role_mask(role)
        fill = Image.new("RGBA", (self.size, self.size), self._rgba(color))
        self._canvas = Image.composite(fill, self._canvas, mask)

    # ------------------------------------------------------------------
    # Patterns and effects
    # ------------------------------------------------------------------

    def contour_lines(self, island_id: int, color: Color,
                      spacing: int = 17, line_width: int = 3):
        """Draw contour-following lines inside an island.

        Uses distance transform so lines follow the panel's natural shape
        (V-shapes on nose, parallel on body, curved on sidepods).

        Parameters
        ----------
        island_id : int
            Which island to draw contours in.
        color : tuple
            RGB/RGBA color for the contour lines.
        spacing : int
            Total period (line_width + gap) in pixels.
        line_width : int
            Width of each contour line in pixels.
        """
        mask = self.island_mask(island_id)
        c = self._rgba(color)
        mask_arr = np.array(mask, dtype=np.float64) / 255.0
        dist = distance_transform_edt(mask_arr > 0.5)
        phase = dist % spacing
        band = (phase < line_width).astype(np.uint8) * 255
        band[mask_arr < 0.5] = 0
        band_mask = Image.fromarray(band, "L")
        fill = Image.new("RGBA", (self.size, self.size), c)
        self._canvas = Image.composite(fill, self._canvas, band_mask)

    def contour_lines_multi(self, island_ids: Sequence[int], color: Color,
                            spacing: int = 17, line_width: int = 3):
        """Draw contour lines on multiple islands."""
        for iid in island_ids:
            self.contour_lines(iid, color, spacing=spacing, line_width=line_width)

    def stripes(self, color: Color, width: int = 10, gap: int = 10,
                angle: float = 0.0, mask: Optional[Image.Image] = None):
        """Draw repeating straight stripes across the canvas or a masked region.

        Parameters
        ----------
        color : tuple
            RGB/RGBA for the stripe color.
        width : int
            Stripe width in pixels.
        gap : int
            Gap between stripes in pixels.
        angle : float
            Rotation angle in degrees (0 = horizontal).
        mask : Image or None
            L-mode mask to restrict stripes to a region.
        """
        c = self._rgba(color)
        sz = self.size
        period = width + gap

        if abs(angle) < 1:
            pat = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pat)
            y = 0
            while y < sz:
                draw.rectangle([0, y, sz, y + width - 1], fill=c)
                y += period
        else:
            diag = int(sz * 1.5)
            pat = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
            draw = ImageDraw.Draw(pat)
            y = 0
            while y < diag:
                draw.rectangle([0, y, diag, y + width - 1], fill=c)
                y += period
            pat = pat.rotate(angle, resample=Image.Resampling.BILINEAR,
                             expand=False, center=(diag // 2, diag // 2))
            ox = (diag - sz) // 2
            pat = pat.crop((ox, ox, ox + sz, ox + sz))

        if mask is not None:
            clipped = pat.copy()
            clipped.putalpha(ImageChops.multiply(clipped.getchannel("A"), mask))
            self._canvas = Image.alpha_composite(self._canvas, clipped)
        else:
            self._canvas = Image.alpha_composite(self._canvas, pat)

    def gradient(self, color_a: Color, color_b: Color,
                 direction: str = "vertical",
                 mask: Optional[Image.Image] = None):
        """Apply a linear gradient.

        Parameters
        ----------
        color_a, color_b : tuple
            Start and end RGB colors.
        direction : str
            "vertical" (top to bottom), "horizontal" (left to right),
            or "diagonal".
        mask : Image or None
            L-mode mask to restrict the gradient.
        """
        sz = self.size
        a = np.array(color_a[:3], dtype=np.float64)
        b = np.array(color_b[:3], dtype=np.float64)

        if direction == "vertical":
            t = np.linspace(0, 1, sz).reshape(sz, 1, 1)
        elif direction == "horizontal":
            t = np.linspace(0, 1, sz).reshape(1, sz, 1)
        else:
            ty = np.linspace(0, 1, sz).reshape(sz, 1)
            tx = np.linspace(0, 1, sz).reshape(1, sz)
            t = ((ty + tx) / 2.0).reshape(sz, sz, 1)

        rgb = (a * (1 - t) + b * t).astype(np.uint8)
        arr = np.zeros((sz, sz, 4), dtype=np.uint8)
        arr[:, :, :3] = rgb
        arr[:, :, 3] = 255
        grad_img = Image.fromarray(arr, "RGBA")

        if mask is not None:
            self._canvas = Image.composite(grad_img, self._canvas, mask)
        else:
            self._canvas = grad_img

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def _load_font(self, size_px: int) -> ImageFont.FreeTypeFont:
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
        for p in candidates:
            if Path(p).exists():
                return ImageFont.truetype(p, size_px)
        return ImageFont.load_default()

    def text(self, content: str, *,
             pos: Optional[Tuple[int, int]] = None,
             island: Optional[int] = None,
             color: Color = (255, 255, 255),
             size_px: int = 72,
             size_frac: Optional[float] = None,
             offset_y: float = 0.5,
             rotation: float = 0,
             anchor: str = "center"):
        """Draw text on the canvas.

        Parameters
        ----------
        content : str
            The text string.
        pos : (x, y) or None
            Absolute pixel position. If None, centers on the given island.
        island : int or None
            Island ID to center the text on (uses island bbox).
        color : tuple
            Text color.
        size_px : int
            Font size in pixels (used when size_frac is None).
        size_frac : float or None
            Font size as a fraction of the island width (overrides size_px).
        offset_y : float
            Vertical offset within the island (0.0 = top, 1.0 = bottom).
        rotation : float
            Degrees to rotate the text.
        anchor : str
            "center", "left", or "right" alignment within the island.
        """
        if island is not None:
            self._ensure_geo()
            x0, y0, x1, y1 = self.island_bbox(island)
            iw, ih = x1 - x0, y1 - y0
            if size_frac is not None:
                size_px = max(12, int(iw * size_frac))

        font = self._load_font(size_px)
        c = self._rgba(color)

        txt_layer = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        bbox = draw.textbbox((0, 0), content, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if pos is not None:
            tx, ty = pos
        elif island is not None:
            x0, y0, x1, y1 = self.island_bbox(island)
            iw, ih = x1 - x0, y1 - y0
            if anchor == "center":
                tx = x0 + (iw - tw) // 2
            elif anchor == "left":
                tx = x0 + int(iw * 0.05)
            else:
                tx = x1 - tw - int(iw * 0.05)
            ty = y0 + int((ih - th) * offset_y)
        else:
            tx = (self.size - tw) // 2
            ty = (self.size - th) // 2

        draw.text((tx, ty), content, fill=c, font=font)

        if abs(rotation) > 0.5:
            txt_layer = txt_layer.rotate(
                rotation, resample=Image.Resampling.BICUBIC,
                center=(tx + tw // 2, ty + th // 2),
            )

        if island is not None:
            mask = self.island_mask(island)
            txt_layer.putalpha(
                ImageChops.multiply(txt_layer.getchannel("A"), mask)
            )

        self._canvas = Image.alpha_composite(self._canvas, txt_layer)

    # ------------------------------------------------------------------
    # Image pasting (logos, stickers, custom artwork)
    # ------------------------------------------------------------------

    def paste(self, image: Union[Image.Image, str, Path], *,
              pos: Optional[Tuple[int, int]] = None,
              island: Optional[int] = None,
              scale: float = 1.0,
              rotation: float = 0,
              opacity: float = 1.0,
              clip_to_island: bool = True):
        """Paste an image (logo, sticker, custom artwork) onto the canvas.

        Parameters
        ----------
        image : Image or path
            RGBA image to paste.
        pos : (x, y) or None
            Top-left pixel position. If None and island given, centers on island.
        island : int or None
            Island to center on / clip to.
        scale : float
            Scale factor (1.0 = original size).
        rotation : float
            Degrees to rotate.
        opacity : float
            0.0 to 1.0.
        clip_to_island : bool
            If True and island is given, clip the pasted image to the island mask.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGBA")
        else:
            image = image.convert("RGBA")

        if abs(scale - 1.0) > 0.01:
            w, h = image.size
            image = image.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )

        if abs(rotation) > 0.5:
            image = image.rotate(
                rotation, expand=True, resample=Image.Resampling.BICUBIC
            )

        if opacity < 1.0:
            a = image.getchannel("A")
            a = a.point(lambda p: int(p * opacity))
            image.putalpha(a)

        layer = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))

        if pos is not None:
            layer.paste(image, pos)
        elif island is not None:
            self._ensure_geo()
            x0, y0, x1, y1 = self.island_bbox(island)
            cx = x0 + (x1 - x0 - image.width) // 2
            cy = y0 + (y1 - y0 - image.height) // 2
            layer.paste(image, (cx, cy))
        else:
            cx = (self.size - image.width) // 2
            cy = (self.size - image.height) // 2
            layer.paste(image, (cx, cy))

        if clip_to_island and island is not None:
            mask = self.island_mask(island)
            layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))

        self._canvas = Image.alpha_composite(self._canvas, layer)

    # ------------------------------------------------------------------
    # Layer system
    # ------------------------------------------------------------------

    def push_layer(self, name: str = "layer", blend: str = "normal",
                   opacity: float = 1.0):
        """Start a new layer. All subsequent painting goes to this layer
        until pop_layer() is called.

        Blend modes: normal, multiply, screen, overlay.
        """
        self._layers.append(_Layer(
            image=self._canvas.copy(),
            mask=None,
            blend=blend,
            opacity=opacity,
            name=name,
        ))
        self._canvas = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))

    def pop_layer(self, mask: Optional[Image.Image] = None):
        """Composite the current layer onto the canvas below it."""
        if not self._layers:
            return
        below_layer = self._layers.pop()
        above = self._canvas

        if below_layer.opacity < 1.0:
            a = above.getchannel("A")
            a = a.point(lambda p: int(p * below_layer.opacity))
            above.putalpha(a)

        if mask is not None:
            above.putalpha(ImageChops.multiply(above.getchannel("A"), mask))

        base = below_layer.image
        blend = below_layer.blend

        if blend == "normal":
            self._canvas = Image.alpha_composite(base, above)
        elif blend == "multiply":
            base_rgb = base.convert("RGB")
            above_rgb = above.convert("RGB")
            result = ImageChops.multiply(base_rgb, above_rgb)
            result_rgba = result.convert("RGBA")
            result_rgba.putalpha(base.getchannel("A"))
            above_mask = above.getchannel("A")
            self._canvas = Image.composite(result_rgba, base, above_mask)
        elif blend == "screen":
            base_rgb = np.array(base.convert("RGB"), dtype=np.float64)
            above_rgb = np.array(above.convert("RGB"), dtype=np.float64)
            screen = 255 - ((255 - base_rgb) * (255 - above_rgb) / 255.0)
            result = Image.fromarray(screen.astype(np.uint8), "RGB").convert("RGBA")
            result.putalpha(base.getchannel("A"))
            above_mask = above.getchannel("A")
            self._canvas = Image.composite(result, base, above_mask)
        elif blend == "overlay":
            base_arr = np.array(base.convert("RGB"), dtype=np.float64)
            above_arr = np.array(above.convert("RGB"), dtype=np.float64)
            low = 2 * base_arr * above_arr / 255.0
            high = 255 - 2 * (255 - base_arr) * (255 - above_arr) / 255.0
            result_arr = np.where(base_arr < 128, low, high)
            result = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
            result.putalpha(base.getchannel("A"))
            above_mask = above.getchannel("A")
            self._canvas = Image.composite(result, base, above_mask)
        else:
            self._canvas = Image.alpha_composite(base, above)

    # ------------------------------------------------------------------
    # Finish / material control
    # ------------------------------------------------------------------

    def set_finish(self, finish: Union[str, int, Finish],
                   mask: Optional[Image.Image] = None):
        """Set material finish for a region.

        Parameters
        ----------
        finish : str, int, or Finish
            - str: "matte", "satin", "gloss", "metallic", "carbon", "brushed"
            - int: raw alpha value (0-255, lower = glossier in TMNF)
            - Finish: a Finish dataclass instance
        mask : Image or None
            L-mode mask. If None, applies to entire canvas.
        """
        if isinstance(finish, str):
            f = _FINISH_MAP.get(finish)
            alpha_val = f.gloss if f else 0x8E
        elif isinstance(finish, int):
            alpha_val = finish
        elif isinstance(finish, Finish):
            alpha_val = finish.gloss
        else:
            alpha_val = 0x8E

        fill = Image.new("L", (self.size, self.size), alpha_val)
        if mask is not None:
            self._finish_map = Image.composite(fill, self._finish_map, mask)
        else:
            self._finish_map = fill

    def set_finish_island(self, island_id: int,
                          finish: Union[str, int, Finish]):
        """Set material finish for a specific island."""
        mask = self.island_mask(island_id)
        self.set_finish(finish, mask=mask)

    def set_finish_islands(self, island_ids: Sequence[int],
                           finish: Union[str, int, Finish]):
        """Set material finish for multiple islands."""
        mask = self.islands_mask(island_ids)
        self.set_finish(finish, mask=mask)

    def set_finish_role(self, role: str, finish: Union[str, int, Finish]):
        """Set material finish for all islands of a role."""
        mask = self.role_mask(role)
        self.set_finish(finish, mask=mask)

    def set_finish_gradient(self, val_a: int, val_b: int,
                            direction: str = "vertical",
                            mask: Optional[Image.Image] = None):
        """Set a gradient finish (per-pixel alpha variation for curvature illusion).

        val_a/val_b are raw alpha values (0-255). Lower = glossier in TMNF.
        """
        sz = self.size
        if direction == "vertical":
            t = np.linspace(0, 1, sz).reshape(sz, 1)
        else:
            t = np.linspace(0, 1, sz).reshape(1, sz)
        vals = (val_a * (1 - t) + val_b * t).astype(np.uint8)
        if direction == "vertical":
            grad = np.broadcast_to(vals, (sz, sz)).copy()
        else:
            grad = np.broadcast_to(vals, (sz, sz)).copy()
        grad_img = Image.fromarray(grad, "L")
        if mask is not None:
            self._finish_map = Image.composite(grad_img, self._finish_map, mask)
        else:
            self._finish_map = grad_img

    # ------------------------------------------------------------------
    # Prelight / AO
    # ------------------------------------------------------------------

    def apply_prelight(self, strength: float = 0.65):
        """Apply synthetic ambient occlusion + top-down lighting to add depth.

        Multiplies a shadow map onto the canvas RGB, preserving colors but
        adding shadow at island edges and a subtle top-down gradient.
        """
        self._ensure_geo()
        sz = self.size
        prelight = Image.new("L", (sz, sz), 255)

        for iid, mask in self._island_masks.items():
            edges = mask.filter(ImageFilter.FIND_EDGES)
            ao = ImageOps.invert(edges)
            ao = ao.filter(ImageFilter.GaussianBlur(sz // 64))
            ao = ao.point(lambda p: int(180 + p * 75 / 255))
            prelight = ImageChops.darker(prelight, ao)

        grad = Image.new("L", (sz, sz), 0)
        for y in range(sz):
            val = int(235 + 20 * (1 - y / sz))
            ImageDraw.Draw(grad).line([(0, y), (sz - 1, y)], fill=val)
        prelight = ImageChops.multiply(prelight, grad)

        diff_rgb = self._canvas.convert("RGB")
        pre_rgb = Image.merge("RGB", (prelight, prelight, prelight))
        mul = ImageChops.multiply(diff_rgb, pre_rgb)

        if strength < 1.0:
            mul = Image.blend(diff_rgb, mul, strength)

        alpha = self._canvas.getchannel("A")
        self._canvas = mul.convert("RGBA")
        self._canvas.putalpha(alpha)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def set_image(self, image: Image.Image):
        """Replace the entire canvas with a pre-built RGBA image."""
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        self._canvas = image.convert("RGBA")

    def compute_adaptive_alpha(self, alpha_bright: int = 0x18,
                                alpha_dark: int = 0x90):
        """Compute luminance-adaptive alpha for the finish map.

        In TMNF, low alpha = vivid color (matte), high alpha = dull color
        (reflective).  This maps bright pattern pixels to low alpha (vivid,
        glowing) and dark background pixels to high alpha (reflective, glossy
        car paint), making patterns pop in-game.
        """
        rgb = np.array(self._canvas.convert("RGB"), dtype=np.float64)
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        lum_norm = np.clip(lum / 255.0, 0.0, 1.0)
        alpha = alpha_dark + (alpha_bright - alpha_dark) * lum_norm
        self._finish_map = Image.fromarray(
            np.clip(alpha, 0, 255).astype(np.uint8), "L"
        )

    def get_diffuse(self) -> Image.Image:
        """Get the final Diffuse image with finish baked into alpha channel."""
        result = self._canvas.copy()
        result.putalpha(self._finish_map)
        return result

    def get_canvas(self) -> Image.Image:
        """Get the raw RGBA canvas (without finish alpha override)."""
        return self._canvas.copy()

    def preview(self, path: Union[str, Path] = "out/_preview.png"):
        """Save a PNG preview of the current canvas."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._canvas.save(p)
        print(f"Preview saved: {p}")

    def save(self, name: str, *, apply_prelight: bool = False,
             prelight_strength: float = 0.65,
             enhance: bool = True,
             adaptive_alpha: bool = True,
             projshad: "Image.Image | None" = None):
        """Save the skin as a game-ready ZIP using ProSkinEngine.

        Parameters
        ----------
        name : str
            Output name (produces out/<name>.zip).
        apply_prelight : bool
            If True, applies ambient occlusion before saving.
        prelight_strength : float
            Strength of the prelight effect (0-1).
        enhance : bool
            If True, applies subtle contrast/sharpness boost.
        adaptive_alpha : bool
            If True, computes luminance-based alpha so bright pattern
            pixels appear vivid and dark areas stay reflective in-game.
        """
        from pro_skin_engine import ProSkinEngine
        from PIL import ImageEnhance, ImageFilter

        if apply_prelight:
            self.apply_prelight(strength=prelight_strength)

        if adaptive_alpha:
            self.compute_adaptive_alpha()

        engine = ProSkinEngine(team_name=name, full_skin=True)
        engine.load_uv_geometry()

        diffuse = self.get_diffuse()

        if enhance:
            rgb = diffuse.convert("RGB")
            rgb = ImageEnhance.Brightness(rgb).enhance(1.08)
            rgb = ImageEnhance.Contrast(rgb).enhance(1.22)
            rgb = ImageEnhance.Color(rgb).enhance(1.18)
            rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2.0, percent=50, threshold=2))
            alpha = diffuse.getchannel("A")
            diffuse = rgb.convert("RGBA")
            diffuse.putalpha(alpha)

        engine.diffuse = diffuse
        if projshad is not None:
            engine.projshad = projshad

        engine._island_finish_alphas = {}
        for iid in engine._island_masks:
            mask = engine._island_masks[iid]
            mask_arr = np.array(mask)
            finish_arr = np.array(self._finish_map)
            masked_vals = finish_arr[mask_arr > 128]
            if len(masked_vals) > 0:
                engine._island_finish_alphas[iid] = int(np.median(masked_vals))
            else:
                engine._island_finish_alphas[iid] = 0x8E

        engine.save()
        print(f"Skin saved: out/{name}.zip")
        return Path(f"out/{name}.zip")
