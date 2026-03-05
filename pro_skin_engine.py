import os
import shutil
import zipfile
import json
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter
import numpy as np
import skin_utils
import random
import math

from layer_stack import SkinLayerStack, Finish, FINISH_MATTE, FINISH_SATIN, FINISH_GLOSS, FINISH_METALLIC, FINISH_CARBON, FINISH_BRUSHED
from tmnf_dds import save_dds_dxt5, save_dds_dxt1, build_dds_dxt5_bytes, build_dds_dxt1_bytes, read_dds_dimensions_from_bytes
from car_geometry import CarGeometry, ColorRole, FinishType, load_stadium_geometry

DEFAULT_BASE_ZIP = Path("CH_all_skins/CH_2026.zip")
UV_ATLAS_JSON = Path("out/uv_atlas/standard_stadium_islands_2048.json")
UV_DIAG_PNG = Path("out/uv_atlas/diagnostics_2048.png")

_FINISH_TYPE_MAP = {
    "matte":    FINISH_MATTE,
    "satin":    FINISH_SATIN,
    "gloss":    FINISH_GLOSS,
    "metallic": FINISH_METALLIC,
    "carbon":   FINISH_CARBON,
    "brushed":  FINISH_BRUSHED,
}


class ProSkinEngine:
    def __init__(self, size=2048, team_name="Team", base_zip=None, full_skin=True):
        """
        Parameters
        ----------
        size : int
            Diffuse texture resolution (usually 2048 for Stadium).
        team_name : str
            Output folder / ZIP name.
        base_zip : path-like or None
            Base car pack ZIP to reskin (default: CH_all_skins/CH_2026.zip).
        full_skin : bool
            If True (default), layers paint the ENTIRE texture -- no
            paint/chassis mask separation.  Set False only for partial
            overlay workflows that need the old mask behavior.
        """
        self.size = size
        self.team_name = team_name
        self.base_zip = Path(base_zip) if base_zip else DEFAULT_BASE_ZIP
        self.assets_dir = Path("assets")
        self.out_dir = Path("out") / team_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._full_skin = full_skin

        self.diffuse = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        self.details = Image.new("RGBA", (size, size), (0, 0, 0, 20))
        self.illum = Image.new("RGBA", (size, size), (0, 0, 0, 255))

        self._encode_finish_in_diffuse_alpha = True
        self._finish_brighten_strength = 0.35
        self._finish_dull_strength = 0.18

        if full_skin:
            # Full-skin mode: paint everywhere, no mask constraints
            self.paint_mask = Image.new("L", (size, size), 255)
            self.chassis_mask = Image.new("L", (size, size), 0)
            self.original_skin = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        else:
            self.paint_mask = self._load_mask("paint_mask.png")
            self.chassis_mask = self._load_mask("chassis_mask.png")
            if Path("base_dds/StadiumCarSkin.png").exists():
                self.original_skin = Image.open("base_dds/StadiumCarSkin.png").resize((size, size)).convert("RGBA")
            else:
                self.original_skin = Image.new("RGBA", (size, size), (50, 50, 50, 255))
            self.diffuse.paste(self.original_skin, (0, 0), self.chassis_mask)

        self.stack = SkinLayerStack(size=size, paint_mask=self.paint_mask)
        self._use_stack = False

        # UV geometry (loaded on demand)
        self._geo: CarGeometry | None = None
        self._island_masks: dict[int, Image.Image] = {}
        self._role_masks: dict[str, Image.Image] = {}
        self._island_finish_alphas: dict[int, int] = {}

        # Deferred post-composition effects
        self._pending_hero_gradient = None   # (top_lighten, bottom_darken)
        self._pending_prelight = None        # strength float
        self._pending_oklch_fade = None      # (color_a, color_b)

    def _load_mask(self, name):
        path = self.assets_dir / "masks" / name
        if path.exists():
            return Image.open(path).resize((self.size, self.size), Image.Resampling.NEAREST).convert("L")
        return Image.new("L", (self.size, self.size), 255)

    # ------------------------------------------------------------------
    # UV GEOMETRY
    # ------------------------------------------------------------------

    def load_uv_geometry(self):
        """Load UV atlas + diagnostics image and build per-island pixel masks.

        After calling this, island_mask(id), role_mask(role), and
        paint_by_role() become available.
        """
        if not UV_ATLAS_JSON.exists() or not UV_DIAG_PNG.exists():
            raise FileNotFoundError(
                f"UV atlas not found at {UV_ATLAS_JSON} / {UV_DIAG_PNG}. "
                "Run tools/export_uv_atlas.py first."
            )
        self._geo = CarGeometry.from_json_file(str(UV_ATLAS_JSON))
        self._build_island_masks()
        self._build_role_masks()
        print(f"UV geometry loaded: {len(self._island_masks)} island masks")

    def _build_island_masks(self):
        """Extract per-island pixel masks from the diagnostics image."""
        diag = np.array(Image.open(UV_DIAG_PNG).convert("RGB"))
        BG = np.array([8, 8, 12])

        with open(UV_ATLAS_JSON) as f:
            atlas = json.load(f)

        for isl in atlas["islands"]:
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

            # Full-image mask: match this island's unique color
            diff = np.abs(diag.astype(int) - center_col.astype(int)).sum(axis=2)
            mask_arr = (diff < 30).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_arr, mode="L")

            if self.size != 2048:
                mask_img = mask_img.resize(
                    (self.size, self.size), Image.Resampling.NEAREST
                )

            self._island_masks[iid] = mask_img
            self._island_finish_alphas[iid] = self._geo.get_finish_alpha_for_island(iid)

    def _build_role_masks(self):
        """Combine island masks by color role."""
        for role in ColorRole:
            ids = self._geo.get_islands_by_role(role)
            combined = Image.new("L", (self.size, self.size), 0)
            for iid in ids:
                if iid in self._island_masks:
                    combined = ImageChops.lighter(combined, self._island_masks[iid])
            self._role_masks[role.value] = combined

    def island_mask(self, island_id: int) -> Image.Image:
        """Get the pixel mask for a specific UV island."""
        return self._island_masks.get(
            island_id, Image.new("L", (self.size, self.size), 0)
        )

    def role_mask(self, role: str) -> Image.Image:
        """Get the combined mask for all islands of a color role.

        Roles: "hero", "secondary", "accent", "darken", "neutral"
        """
        return self._role_masks.get(
            role, Image.new("L", (self.size, self.size), 0)
        )

    def _make_islands_mask(self, island_ids: list) -> Image.Image:
        """Combine masks for a specific set of island IDs."""
        combined = Image.new("L", (self.size, self.size), 0)
        for iid in island_ids:
            m = self._island_masks.get(iid)
            if m is not None:
                combined = ImageChops.lighter(combined, m)
        return combined

    def paint_by_role(self, role_spec: dict, *, clearcoat_sweep=None, fresnel_boost=0):
        """Paint the entire car using role-based color/pattern/finish assignment.

        Parameters
        ----------
        role_spec : dict
            Mapping from role name to a dict with:
              "color": RGB tuple (required)
              "pattern": PIL Image to overlay (optional)
              "pattern_blend": blend mode for pattern (default "normal")
              "pattern_opacity": float 0-1 (default 1.0)
              "pattern_islands": list of island IDs to restrict pattern to
                  (optional -- if omitted, pattern covers all islands in role)
              "finish": Finish object (optional, uses CarGeometry defaults)
        """
        if not self._geo:
            self.load_uv_geometry()

        ROLE_ORDER = ["neutral", "darken", "secondary", "hero", "accent"]

        for role_name in ROLE_ORDER:
            spec = role_spec.get(role_name)
            if spec is None:
                spec = {"color": (20, 20, 22)}

            mask = self.role_mask(role_name)
            color = spec["color"]
            finish = spec.get("finish")
            pattern = spec.get("pattern")
            p_blend = spec.get("pattern_blend", "normal")
            p_opacity = spec.get("pattern_opacity", 1.0)
            p_islands = spec.get("pattern_islands")

            self.add_layer(f"base_{role_name}", color, mask=mask, finish=finish)

            if pattern is not None:
                p_mask = self._make_islands_mask(p_islands) if p_islands else mask
                self.add_layer(
                    f"pattern_{role_name}", pattern,
                    mask=p_mask, blend=p_blend, opacity=p_opacity, finish=finish,
                )

    # ------------------------------------------------------------------
    # Shape primitives (ported from generate_tmnf_skin.py pro styles)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_slash(size, color_rgba, angle_deg=55.0, thickness=0.10,
                    position=0.5, feather=6):
        """Rotated diagonal slash across the texture.

        angle_deg: 0=horizontal, 90=vertical, 45-65=typical racing angle.
        thickness: fraction of image size.
        position: center along perpendicular axis (0-1).
        """
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        rad = math.radians(angle_deg)
        ext = size * 2
        ht = (thickness * size) / 2
        cx = size * position
        cy = size * 0.5
        dx, dy = math.cos(rad), math.sin(rad)
        px, py = -math.sin(rad), math.cos(rad)
        pts = [
            (cx - dx * ext + px * ht, cy - dy * ext + py * ht),
            (cx + dx * ext + px * ht, cy + dy * ext + py * ht),
            (cx + dx * ext - px * ht, cy + dy * ext - py * ht),
            (cx - dx * ext - px * ht, cy - dy * ext - py * ht),
        ]
        d.polygon(pts, fill=color_rgba)
        if feather > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(radius=feather))
        return layer

    @staticmethod
    def _make_swoosh(size, color_rgba, thickness=60, curve_type="wave",
                     flip=False):
        """Organic curve: wave (S-curve), arc, or slash."""
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        pts = []
        if curve_type == "wave":
            for i in range(100):
                t = i / 99.0
                x = t * size
                y = size * 0.5 + math.sin(t * math.pi * 1.5) * size * 0.25
                pts.append((x, y))
        elif curve_type == "arc":
            for i in range(100):
                t = i / 99.0
                a = t * math.pi * 0.5
                x = size * 0.2 + math.cos(a) * size * 0.7
                y = size * 0.8 - math.sin(a) * size * 0.6
                pts.append((x, y))
        else:
            pts = [(0, size * 0.7), (size, size * 0.3)]
        if flip:
            pts = [(size - x, y) for x, y in pts]
        if len(pts) >= 2:
            d.line(pts, fill=color_rgba, width=thickness, joint="curve")
        layer = layer.filter(ImageFilter.GaussianBlur(
            radius=max(2, thickness // 8)))
        return layer

    @staticmethod
    def _make_blocks(size, color1_rgba, color2_rgba, style="angular"):
        """Geometric block polygons for aggressive esports look."""
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if style == "angular":
            d.polygon([(0, size * 0.3), (size * 0.6, 0), (size, 0),
                        (size, size * 0.4), (size * 0.4, size * 0.7),
                        (0, size * 0.5)], fill=color1_rgba)
            d.polygon([(size * 0.7, size), (size, size * 0.6),
                        (size, size)], fill=color2_rgba)
        elif style == "split":
            d.polygon([(0, 0), (size, 0), (size, int(size * 0.4)),
                        (0, int(size * 0.6))], fill=color1_rgba)
        elif style == "corner":
            d.polygon([(0, 0), (int(size * 0.3), 0),
                        (0, int(size * 0.3))], fill=color1_rgba)
            d.polygon([(size, size), (int(size * 0.7), size),
                        (size, int(size * 0.7))], fill=color2_rgba)
        return layer

    @staticmethod
    def _make_diagonal_band(size, color_rgba, highlight_rgba,
                            band_width=0.30, angle_deg=-18.0,
                            offset_x=0.0, offset_y=-0.08):
        """Wide diagonal swoosh band with highlight edge."""
        bw = int(size * (0.55 + band_width))
        bh = int(size * band_width)
        band = Image.new("RGBA", (bw, bh), color_rgba)
        bd = ImageDraw.Draw(band)
        edge_h = max(2, bh // 10)
        bd.rectangle((0, 0, bw, edge_h), fill=highlight_rgba)
        lo_a = max(0, highlight_rgba[3] - 70) if len(highlight_rgba) > 3 else 180
        bd.rectangle((0, bh - edge_h, bw, bh),
                     fill=(highlight_rgba[0], highlight_rgba[1],
                           highlight_rgba[2], lo_a))
        band = band.filter(ImageFilter.GaussianBlur(
            radius=max(2, size // 420)))
        band = band.rotate(angle_deg, expand=True,
                           resample=Image.Resampling.BICUBIC)
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        x = (size - band.size[0]) // 2 + int(size * offset_x)
        y = (size - band.size[1]) // 2 + int(size * offset_y)
        layer.alpha_composite(band, (x, y))
        return layer

    def apply_design_layers(self, layers: list):
        """Apply design overlay layers on top of base role painting.

        Supported layer types:

        hband/vband     -- horizontal/vertical band
        slash           -- rotated diagonal slash (proper angle math)
        swoosh          -- organic curve (wave/arc)
        blocks          -- geometric polygon blocks
        band            -- wide diagonal swoosh band with highlight edge
        overlay         -- pattern image overlay
        pinstripes      -- multiple thin diagonal lines
        """
        if not self._geo:
            self.load_uv_geometry()

        sz = self.size
        for i, layer in enumerate(layers):
            ltype = layer["type"]
            color = layer.get("color", (255, 255, 255))
            finish = layer.get("finish")
            feather = layer.get("feather", 10)
            islands = layer.get("islands")
            opacity = layer.get("opacity", 1.0)
            blend = layer.get("blend", "normal")
            alpha = layer.get("alpha", 255)
            c4 = (color[0], color[1], color[2], alpha)

            if ltype == "hband":
                y_center = int(layer["y"] * sz)
                h = int(layer["h"] * sz)
                mask = Image.new("L", (sz, sz), 0)
                d = ImageDraw.Draw(mask)
                d.rectangle([0, y_center - h // 2, sz, y_center + h // 2],
                            fill=255)
                if feather > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(feather))
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                self.add_layer(f"hband_{i}", color, mask=mask,
                               opacity=opacity, finish=finish)

            elif ltype == "vband":
                x_center = int(layer["x"] * sz)
                w = int(layer["w"] * sz)
                mask = Image.new("L", (sz, sz), 0)
                d = ImageDraw.Draw(mask)
                d.rectangle([x_center - w // 2, 0,
                             x_center + w // 2, sz], fill=255)
                if feather > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(feather))
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                self.add_layer(f"vband_{i}", color, mask=mask,
                               opacity=opacity, finish=finish)

            elif ltype == "slash":
                angle = layer.get("angle", 55.0)
                thick = layer.get("thickness", 0.10)
                pos = layer.get("position", 0.5)
                sl = self._make_slash(sz, c4, angle_deg=angle,
                                      thickness=thick, position=pos,
                                      feather=feather)
                mask = sl.split()[3]
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                rgb = sl.convert("RGB")
                self.add_layer(f"slash_{i}", rgb, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

            elif ltype == "swoosh":
                thick_px = layer.get("thickness", max(40, sz // 26))
                ctype = layer.get("curve_type", "wave")
                flip = layer.get("flip", False)
                sw = self._make_swoosh(sz, c4, thickness=thick_px,
                                       curve_type=ctype, flip=flip)
                mask = sw.split()[3]
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                rgb = sw.convert("RGB")
                self.add_layer(f"swoosh_{i}", rgb, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

            elif ltype == "blocks":
                c2 = layer.get("color2", color)
                a2 = layer.get("alpha2", alpha)
                c2_4 = (c2[0], c2[1], c2[2], a2)
                bstyle = layer.get("style", "angular")
                blk = self._make_blocks(sz, c4, c2_4, style=bstyle)
                mask = blk.split()[3]
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                rgb = blk.convert("RGB")
                self.add_layer(f"blocks_{i}", rgb, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

            elif ltype == "band":
                hl = layer.get("highlight", (255, 255, 255))
                hl_a = layer.get("highlight_alpha", 200)
                hl4 = (hl[0], hl[1], hl[2], hl_a)
                bw = layer.get("band_width", 0.30)
                ang = layer.get("angle", -18.0)
                ox = layer.get("offset_x", 0.0)
                oy = layer.get("offset_y", -0.08)
                bd = self._make_diagonal_band(sz, c4, hl4,
                                              band_width=bw,
                                              angle_deg=ang,
                                              offset_x=ox, offset_y=oy)
                mask = bd.split()[3]
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                rgb = bd.convert("RGB")
                self.add_layer(f"band_{i}", rgb, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

            elif ltype == "pinstripes":
                count = layer.get("count", 4)
                colors_list = layer.get("colors", [color])
                angle_range = layer.get("angle_range", (50, 75))
                pin_thick = layer.get("thickness", 0.008)
                pin_alpha = layer.get("alpha", 80)
                rng = random.Random(layer.get("seed", 42 + i))
                combined = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
                for j in range(count):
                    pc = colors_list[j % len(colors_list)]
                    pa = pin_alpha + rng.randint(-20, 20)
                    pa = max(30, min(200, pa))
                    pc4 = (pc[0], pc[1], pc[2], pa)
                    ang = rng.uniform(angle_range[0], angle_range[1])
                    pos = rng.uniform(0.12, 0.88)
                    ps = self._make_slash(sz, pc4, angle_deg=ang,
                                          thickness=pin_thick,
                                          position=pos, feather=1)
                    combined = Image.alpha_composite(combined, ps)
                mask = combined.split()[3]
                if islands:
                    mask = ImageChops.multiply(
                        mask, self._make_islands_mask(islands))
                rgb = combined.convert("RGB")
                self.add_layer(f"pins_{i}", rgb, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

            elif ltype == "overlay":
                img = layer["image"]
                mask = self._make_islands_mask(islands) if islands else None
                self.add_layer(f"overlay_{i}", img, mask=mask,
                               blend=blend, opacity=opacity, finish=finish)

    # ------------------------------------------------------------------
    # NEW stack-based API
    # ------------------------------------------------------------------

    def add_layer(self, tag, image, *, mask=None, blend="normal", opacity=1.0, finish=None):
        """Add a compositing layer via the new layer stack.

        ``image`` may be a PIL Image or an RGB(A) color tuple -- tuples are
        expanded to a solid-fill image automatically.
        """
        if isinstance(image, tuple):
            if len(image) == 3:
                image = Image.new("RGBA", (self.size, self.size), image + (255,))
            else:
                image = Image.new("RGBA", (self.size, self.size), image)
        self._use_stack = True
        self.stack.add(tag, image, mask=mask, blend=blend, opacity=opacity, finish=finish)

    def flatten_stack(self, clearcoat_sweep=None, fresnel_boost=0):
        """Flatten the layer stack into self.diffuse and self.details.

        Useful if you want to inspect the result before save(), or if you
        need to do additional legacy manipulation after stack composition.
        """
        if not self.stack.layers:
            return

        diffuse, details = self.stack.flatten(
            clearcoat_sweep=clearcoat_sweep,
            fresnel_boost=fresnel_boost,
        )

        # Composite over the chassis base (stack only covers paint area)
        base = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        base.paste(self.original_skin, (0, 0), self.chassis_mask)
        self.diffuse = Image.alpha_composite(base, diffuse)
        self.details = details

    # ------------------------------------------------------------------
    # LEGACY API (unchanged signatures, backward compatible)
    # ------------------------------------------------------------------

    def set_base_material(self, color, material_type="matte"):
        """Sets the base paint color and material finish."""
        base_layer = Image.new("RGBA", (self.size, self.size), color)
        self.diffuse.paste(base_layer, (0, 0), self.paint_mask)

        temp_diffuse = Image.new("RGB", (self.size, self.size), color)
        material_layer = skin_utils.apply_material_finish(temp_diffuse, material_type)
        self.details.paste(material_layer, (0, 0), self.paint_mask)

    def add_pattern(self, pattern_func, color, opacity=1.0, blend_mode="normal"):
        """Generates and applies a pattern to the paintable area."""
        pat_img = pattern_func(self.size, color)

        if opacity < 1.0:
            a = pat_img.getchannel("A")
            a = a.point(lambda p: int(p * opacity))
            pat_img.putalpha(a)

        final_pat = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        final_pat.paste(pat_img, (0, 0), self.paint_mask)

        if blend_mode == "overlay":
            self.diffuse = skin_utils.blend_overlay(self.diffuse, final_pat)
        elif blend_mode == "multiply":
            self.diffuse = skin_utils.blend_multiply(self.diffuse, final_pat)
        else:
            self.diffuse = Image.alpha_composite(self.diffuse, final_pat)

    def _place_sticker_image(self, sticker, pos, scale, rotation, opacity, gloss):
        """Shared placement logic for add_sticker / add_sticker_pro."""
        w, h = sticker.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        sticker = sticker.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if rotation != 0:
            sticker = sticker.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

        px = int(pos[0] * self.size)
        py = int(pos[1] * self.size)
        ox = px - sticker.width // 2
        oy = py - sticker.height // 2

        if opacity < 1.0:
            a = sticker.getchannel("A")
            a = a.point(lambda p: int(p * opacity))
            sticker.putalpha(a)

        sticker_layer = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        sticker_layer.paste(sticker, (ox, oy))
        sticker_layer.putalpha(ImageChops.multiply(sticker_layer.getchannel("A"), self.paint_mask))

        self.diffuse = Image.alpha_composite(self.diffuse, sticker_layer)

        if gloss:
            sticker_gloss_layer = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            sticker_fill = Image.new("RGBA", sticker.size, (0, 0, 0, 220))
            sticker_gloss_layer.paste(sticker_fill, (ox, oy), sticker)
            sticker_gloss_layer.putalpha(
                ImageChops.multiply(sticker_gloss_layer.getchannel("A"), self.paint_mask)
            )
            d_alpha = self.details.getchannel("A")
            s_alpha = sticker_gloss_layer.getchannel("A")
            self.details.putalpha(ImageChops.lighter(d_alpha, s_alpha))

    def add_sticker(self, image_path, pos, scale=1.0, rotation=0, opacity=1.0, gloss=True):
        """Adds a sticker/logo from a file path."""
        if not os.path.exists(image_path):
            print(f"Warning: Sticker not found {image_path}")
            return
        sticker = Image.open(image_path).convert("RGBA")
        self._place_sticker_image(sticker, pos, scale, rotation, opacity, gloss)

    def add_sticker_pro(self, logo_name, pos, scale=1.0, rotation=0, opacity=1.0, gloss=True, color_override=None):
        """Adds a procedural sticker from the generated library."""
        path = Path("assets/generated_logos") / f"{logo_name}.png"
        if not path.exists():
            path = Path("stickers/sponsors") / f"{logo_name}.png"
        if not path.exists():
            path = Path(logo_name)
        if not path.exists():
            print(f"Warning: Sticker {logo_name} not found.")
            return

        sticker = Image.open(path).convert("RGBA")

        if color_override:
            r, g, b, a = sticker.split()
            gray = sticker.convert("L")
            tinted = ImageOps.colorize(gray, "black", color_override)
            tinted.putalpha(a)
            sticker = tinted

        self._place_sticker_image(sticker, pos, scale, rotation, opacity, gloss)

    def add_glow(self, pattern_func, color, intensity=1.0):
        """Adds a glowing pattern to the Illum map."""
        pat_img = pattern_func(self.size, color)

        if intensity < 1.0:
            r, g, b, a = pat_img.split()
            r = r.point(lambda p: int(p * intensity))
            g = g.point(lambda p: int(p * intensity))
            b = b.point(lambda p: int(p * intensity))
            pat_img = Image.merge("RGBA", (r, g, b, a))

        final_pat = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        final_pat.paste(pat_img, (0, 0), self.paint_mask)
        self.illum = ImageChops.screen(self.illum, final_pat)

    # ------------------------------------------------------------------
    # Dirt / finish / output
    # ------------------------------------------------------------------

    def set_dirt_amount(self, amount: float):
        self._dirt_amount = max(0.0, min(1.0, float(amount)))

    def set_finish_encoding(self, enabled: bool, brighten_strength: float = 0.35, dull_strength: float = 0.18):
        self._encode_finish_in_diffuse_alpha = bool(enabled)
        self._finish_brighten_strength = float(brighten_strength)
        self._finish_dull_strength = float(dull_strength)

    def generate_dirty_maps(self):
        print("Generating dirt maps...")
        dirt_amount = getattr(self, "_dirt_amount", 1.0)

        dirt_noise = (
            Image.effect_noise((self.size // 4, self.size // 4), 20)
            .resize((self.size, self.size), Image.Resampling.BICUBIC)
            .convert("L")
        )
        mud_color = (60, 50, 40)
        dirt_layer = ImageOps.colorize(dirt_noise, "black", mud_color)

        dirt_alpha = dirt_noise.point(lambda p: 255 if p > 150 else 0)
        dirt_alpha = dirt_alpha.filter(ImageFilter.GaussianBlur(5))
        if dirt_amount < 1.0:
            dirt_alpha = dirt_alpha.point(lambda p: int(p * dirt_amount))
        dirt_layer.putalpha(dirt_alpha)

        diffuse_dirty = self.diffuse.copy()
        diffuse_dirty.alpha_composite(dirt_layer)
        diffuse_dirty.save(self.out_dir / "DiffuseDirty.png")

        details_dirty = self.details.copy()
        d_alpha = details_dirty.getchannel("A")
        dirt_mask = ImageOps.invert(dirt_alpha)
        new_gloss = ImageChops.multiply(d_alpha, dirt_mask)
        details_dirty.putalpha(new_gloss)
        details_dirty.save(self.out_dir / "DetailsDirty.png")

    def _contrast_punch(self, contrast=1.18, saturation=1.06, gamma=0.95):
        """Subtle contrast/color boost like the pro skins use."""
        from PIL import ImageEnhance
        rgb = self.diffuse.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
        arr = np.array(rgb, dtype=np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        punched = Image.fromarray(arr, "RGB")
        alpha = self.diffuse.getchannel("A")
        self.diffuse = punched.copy()
        self.diffuse.putalpha(alpha)

    def _finalize_finish_channels(self):
        """Set Diffuse alpha per-island using CarGeometry finish values.

        Per TMNF tutorials: black alpha = bright/matte, white = dull/reflective.
        Each car part gets its own finish alpha (high-gloss 0x68, neutral 0x8E,
        matte 0xA8, carbon 0xB0).  Falls back to flat 0x8E when no geometry.
        """
        NEUTRAL = 0x8E

        if self._island_masks:
            finish_map = Image.new("L", (self.size, self.size), NEUTRAL)
            for iid, mask in self._island_masks.items():
                alpha_val = self._island_finish_alphas.get(iid, NEUTRAL)
                island_fill = Image.new("L", (self.size, self.size), alpha_val)
                finish_map = Image.composite(island_fill, finish_map, mask)
            self.diffuse.putalpha(finish_map)
        else:
            pm = self.paint_mask
            neutral = Image.new("L", (self.size, self.size), NEUTRAL)
            da = self.diffuse.getchannel("A")
            new_da = Image.composite(neutral, da, pm)
            self.diffuse.putalpha(new_da)

    def include_audio(self):
        audio_src = self.assets_dir / "audio"
        if not audio_src.exists():
            return
        audio_files = list(audio_src.glob("*.ogg")) + list(audio_src.glob("*.wav"))
        if audio_files:
            print(f"Including audio files: {[f.name for f in audio_files]}")
            import shutil
            for f in audio_files:
                shutil.copy(f, self.out_dir / f.name)

    def apply_oklch_fade(self, color_a, color_b, *, _deferred=False):
        """Apply an OKLCH perceptual gradient across hero areas.

        Smoothly transitions hue/chroma/lightness from color_a to color_b
        across the vertical extent of each hero island.
        """
        if not _deferred and self._use_stack:
            self._pending_oklch_fade = (color_a, color_b)
            return

        if not self._geo or not self._island_masks:
            return

        from palette_lab import gradient_oklch

        sz = self.size
        hero_ids = self._geo.get_islands_by_role("hero")
        steps = 64
        grad_colors = gradient_oklch(color_a, color_b, steps)

        diff_arr = np.array(self.diffuse).copy()

        for iid in hero_ids:
            mask = self._island_masks.get(iid)
            if mask is None:
                continue
            island = self._geo.islands.get(iid)
            if not island:
                continue

            _, y0, _, y1 = island.bbox
            mask_arr = np.array(mask)
            h = max(1, y1 - y0)

            for y in range(sz):
                if mask_arr[y].max() < 128:
                    continue
                t = (y - y0) / h
                t = max(0.0, min(1.0, t))
                idx = min(int(t * (steps - 1)), steps - 1)
                gc = grad_colors[idx]
                row_mask = (mask_arr[y] > 128)
                for c in range(3):
                    diff_arr[y, :, c] = np.where(row_mask, gc[c], diff_arr[y, :, c])

        self.diffuse = Image.fromarray(diff_arr, "RGBA")
        print("  OKLCH fade applied")

    def apply_prelight(self, strength=0.65, *, _deferred=False):
        """Apply a procedural prelight (ambient occlusion + top-down light).

        Multiplies a generated shadow map onto Diffuse RGB, preserving alpha.
        Call before save() -- if the layer stack is in use, the effect is
        automatically deferred until after flatten.
        """
        if not _deferred and self._use_stack:
            self._pending_prelight = strength
            return

        sz = self.size
        prelight = Image.new("L", (sz, sz), 255)

        if self._island_masks:
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

        diff_rgb = self.diffuse.convert("RGB")
        pre_rgb = Image.merge("RGB", (prelight, prelight, prelight))
        mul = ImageChops.multiply(diff_rgb, pre_rgb)

        if strength < 1.0:
            mul = Image.blend(diff_rgb, mul, strength)

        alpha = self.diffuse.getchannel("A")
        self.diffuse = mul.convert("RGBA")
        self.diffuse.putalpha(alpha)
        print(f"  Prelight applied (strength={strength})")

    def apply_hero_gradient(self, top_lighten=0.15, bottom_darken=0.2, *, _deferred=False):
        """Add a subtle vertical gradient within hero islands.

        Makes hero areas lighter at top, darker at bottom for dimension.
        """
        if not _deferred and self._use_stack:
            self._pending_hero_gradient = (top_lighten, bottom_darken)
            return

        if not self._geo or not self._island_masks:
            return

        sz = self.size
        hero_ids = self._geo.get_islands_by_role("hero")
        diff_arr = np.array(self.diffuse).astype(np.float32)

        for iid in hero_ids:
            mask = self._island_masks.get(iid)
            if mask is None:
                continue
            island = self._geo.islands.get(iid)
            if not island:
                continue

            _, y0, _, y1 = island.bbox
            mask_arr = np.array(mask).astype(np.float32) / 255.0

            for y in range(sz):
                if mask_arr[y].max() < 0.01:
                    continue
                t = (y - y0) / max(1, y1 - y0) if y1 > y0 else 0.5
                t = max(0.0, min(1.0, t))
                factor = 1.0 + top_lighten * (1 - t) - bottom_darken * t
                row_mask = mask_arr[y]
                for c in range(3):
                    diff_arr[y, :, c] = (
                        diff_arr[y, :, c] * (1 - row_mask)
                        + np.clip(diff_arr[y, :, c] * factor, 0, 255) * row_mask
                    )

        result = Image.fromarray(diff_arr.astype(np.uint8), "RGBA")
        alpha = self.diffuse.getchannel("A")
        self.diffuse = result
        self.diffuse.putalpha(alpha)
        print("  Hero gradient applied")

    def save(self, clearcoat_sweep=None, fresnel_boost=0):
        """Write all outputs and package into a game-ready ZIP.

        The ZIP is built by cloning the base car pack (which contains the GBX
        mesh files the game needs) and replacing the texture DDS files with our
        generated ones, encoded as proper DXT5/DXT1 with full mipmap chains.
        """
        print(f"Saving skin to {self.out_dir}...")

        if self._use_stack and self.stack.layers:
            self.flatten_stack(clearcoat_sweep=clearcoat_sweep, fresnel_boost=fresnel_boost)

        # Post-composition effects (applied after stack is flattened)
        if self._pending_oklch_fade:
            ca, cb = self._pending_oklch_fade
            self.apply_oklch_fade(ca, cb, _deferred=True)
        if self._pending_hero_gradient:
            tl, bd = self._pending_hero_gradient
            self.apply_hero_gradient(top_lighten=tl, bottom_darken=bd, _deferred=True)
        if self._pending_prelight is not None:
            self.apply_prelight(strength=self._pending_prelight, _deferred=True)

        self._contrast_punch()
        self._finalize_finish_channels()

        # --- PNG previews ---
        self.diffuse.save(self.out_dir / "Diffuse.png")
        self.details.save(self.out_dir / "Details.png")

        has_illum = self.illum.convert("L").getextrema() != (0, 0)
        if has_illum:
            self.illum.save(self.out_dir / "Illum.png")

        icon = self.diffuse.resize((128, 128), Image.Resampling.LANCZOS)
        icon.save(self.out_dir / "Icon.png")

        projshad = self._make_projshad_image()
        projshad.save(self.out_dir / "ProjShad.png")

        self.generate_dirty_maps()
        self.include_audio()

        # --- Resolve target DDS sizes from the base pack ---
        base_sizes = {}
        if self.base_zip.exists():
            with zipfile.ZipFile(self.base_zip, "r") as zin:
                for entry in zin.infolist():
                    if entry.filename.endswith(".dds"):
                        buf = zin.read(entry.filename)
                        w, h = read_dds_dimensions_from_bytes(buf)
                        base_sizes[entry.filename] = (w, h)

        def _match_base_size(img: Image.Image, name: str) -> Image.Image:
            target = base_sizes.get(name)
            if target and (img.width, img.height) != target:
                print(f"  Resizing {name}: {img.width}x{img.height} -> {target[0]}x{target[1]}")
                return img.resize(target, Image.Resampling.LANCZOS)
            return img

        # --- Encode DDS payloads in memory ---
        print("Encoding DDS (DXT5/DXT1 with mipmaps)...")
        dirty_diff = Image.open(self.out_dir / "DiffuseDirty.png").convert("RGBA")
        dirty_det = Image.open(self.out_dir / "DetailsDirty.png").convert("RGBA")

        # Details.dds uses a separate UV space (d* primitives: wheels, rims,
        # structural details). Keep the base pack's original to avoid
        # overwriting wheel textures with our Diffuse-space finish map.
        replacements = {
            "Diffuse.dds":      build_dds_dxt5_bytes(_match_base_size(self.diffuse, "Diffuse.dds")),
            "Icon.dds":         build_dds_dxt5_bytes(_match_base_size(icon, "Icon.dds")),
            "DiffuseDirty.dds": build_dds_dxt5_bytes(_match_base_size(dirty_diff, "DiffuseDirty.dds")),
            "DetailsDirty.dds": build_dds_dxt5_bytes(_match_base_size(dirty_det, "DetailsDirty.dds")),
        }
        if has_illum:
            replacements["Illum.dds"] = build_dds_dxt1_bytes(self.illum)

        # Also write DDS to disk for inspection
        for name, data in replacements.items():
            (self.out_dir / name).write_bytes(data)

        # --- Build game-ready ZIP from base pack ---
        zip_path = self.out_dir.parent / f"{self.team_name}.zip"

        if self.base_zip.exists():
            self._build_reskinned_zip(zip_path, replacements, has_illum)
        else:
            print(f"WARNING: base ZIP not found at {self.base_zip}")
            print("  Writing texture-only ZIP (will NOT work in-game without mesh files).")
            self._write_textures_only_zip(zip_path, replacements)

        print(f"ZIP: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
        print("Done.")

    def _build_reskinned_zip(self, out_zip: Path, replacements: dict, has_illum: bool):
        """Clone the base car pack ZIP, replacing texture DDS files with ours."""
        additions = {}
        with zipfile.ZipFile(self.base_zip, "r") as zin:
            base_names = {i.filename for i in zin.infolist()}

            # If the base pack lacks Illum but we have one, add it
            if has_illum and "Illum.dds" not in base_names:
                additions["Illum.dds"] = replacements.pop("Illum.dds")

            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    name = info.filename
                    if hasattr(info, "is_dir") and info.is_dir():
                        zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                        zi.external_attr = info.external_attr
                        zout.writestr(zi, b"")
                        continue

                    zi = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zi.external_attr = info.external_attr

                    if name in replacements:
                        zout.writestr(zi, replacements[name])
                    else:
                        with zin.open(info, "r") as src, zout.open(zi, "w") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)

                for name, data in additions.items():
                    zi = zipfile.ZipInfo(filename=name)
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zout.writestr(zi, data)

    def _write_textures_only_zip(self, out_zip: Path, replacements: dict):
        """Fallback: write DDS-only ZIP when no base pack is available."""
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
            for name, data in replacements.items():
                zf.writestr(name, data)

    def _make_projshad_image(self) -> Image.Image:
        """Generates a soft shadow blob and returns it (does not save to disk)."""
        shad = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
        d = ImageDraw.Draw(shad)
        d.ellipse([20, 20, 492, 492], fill=(255, 255, 255, 200))
        return shad.filter(ImageFilter.GaussianBlur(30))


# =============================================================================
# AESTHETIC GENERATORS (unchanged)
# =============================================================================

def create_cyber_tech_skin(team_name, colors):
    """CyberTech Style: Matte Black/Dark Base, Neon Circuits, Carbon Accents."""
    engine = ProSkinEngine(team_name=team_name)

    engine.set_base_material(colors["base"], "matte")

    def circuits_pat(size, col):
        return skin_utils.generate_circuit_traces(size, col, density="medium")

    engine.add_pattern(circuits_pat, colors["accent"], opacity=1.0, blend_mode="screen")

    carbon = skin_utils._generate_carbon_pattern(engine.size)
    carbon_colored = skin_utils.colorize_pattern(carbon, (10, 10, 10))

    mask = Image.new("L", (engine.size, engine.size), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([0, engine.size // 2, engine.size, engine.size], fill=255)
    mask = ImageChops.multiply(mask, engine.paint_mask)

    engine.diffuse.paste(carbon_colored, (0, 0), mask)
    carbon_det = skin_utils.apply_material_finish(carbon_colored, "carbon")
    engine.details.paste(carbon_det, (0, 0), mask)

    engine.add_sticker("stickers/sponsors/arc_labs_bold.png", (0.25, 0.25), scale=0.8, rotation=-90)
    engine.add_sticker("stickers/sponsors/hexworks.png", (0.75, 0.75), scale=1.0, rotation=0)

    engine.save()


if __name__ == "__main__":
    cols_cyber = {
        "base": (30, 30, 35),
        "accent": (0, 255, 255),
        "highlight": (255, 0, 255),
    }
    create_cyber_tech_skin("CyberTech_Pro", cols_cyber)
