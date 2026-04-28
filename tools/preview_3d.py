#!/usr/bin/env python3
"""
Software 3D renderer for TMNF/TMUF skin previews.

Renders the StadiumCar OBJ with a skin zip's Diffuse.dds texture
from multiple camera angles and composites into a single preview sheet.

No GPU required -- pure numpy + PIL rasterizer.

Usage:
    python3 tools/preview_3d.py out/my_skin.zip
    python3 tools/preview_3d.py out/my_skin.zip --out-dir out/previews
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OBJ_PATH = ROOT / "models" / "StadiumCar.obj"


def _load_obj(path: Path):
    """Minimal OBJ loader. Returns (vertices, faces, uvs, face_uvs, is_floor).
    is_floor is a bool array per face -- True for ground-plane faces."""
    verts = []
    tex_coords = []
    faces = []
    face_uvs = []
    face_is_floor = []

    current_group = ""
    group_face_start = {}

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] in ("o", "g") and len(parts) >= 2:
                current_group = parts[1].lower()
                group_face_start[current_group] = len(faces)
            elif parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt" and len(parts) >= 3:
                tex_coords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                fv = []
                ft = []
                for p in parts[1:]:
                    indices = p.split("/")
                    fv.append(int(indices[0]) - 1)
                    if len(indices) >= 2 and indices[1]:
                        ft.append(int(indices[1]) - 1)
                if len(fv) == 3:
                    faces.append(fv)
                    if len(ft) == 3:
                        face_uvs.append(ft)
                    face_is_floor.append(False)
                elif len(fv) == 4:
                    faces.append([fv[0], fv[1], fv[2]])
                    faces.append([fv[0], fv[2], fv[3]])
                    if len(ft) == 4:
                        face_uvs.append([ft[0], ft[1], ft[2]])
                        face_uvs.append([ft[0], ft[2], ft[3]])
                    face_is_floor.append(False)
                    face_is_floor.append(False)

    verts_arr = np.array(verts, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int32)
    floor_arr = np.array(face_is_floor, dtype=bool)

    # Auto-detect floor: faces where all 3 vertices have nearly identical Y
    # and the triangle spans a large XZ area (ground plane)
    for i, f in enumerate(faces_arr):
        fv = verts_arr[f]
        y_range = fv[:, 1].max() - fv[:, 1].min()
        xz_area = abs(np.cross(fv[1, [0, 2]] - fv[0, [0, 2]],
                                fv[2, [0, 2]] - fv[0, [0, 2]]))
        if y_range < 0.01 and xz_area > 2.0:
            floor_arr[i] = True

    return (verts_arr, faces_arr,
            np.array(tex_coords, dtype=np.float64) if tex_coords else None,
            np.array(face_uvs, dtype=np.int32) if face_uvs else None,
            floor_arr)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    view = np.eye(4)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(r, eye)
    view[1, 3] = -np.dot(u, eye)
    view[2, 3] = np.dot(f, eye)
    return view


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1
    return proj


def _render_view(
    verts: np.ndarray,
    faces: np.ndarray,
    tex_coords: np.ndarray,
    face_uvs: np.ndarray,
    texture: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    is_floor: Optional[np.ndarray] = None,
    width: int = 640,
    height: int = 480,
    fov: float = 30.0,
    bg_color: Tuple[int, int, int] = (20, 20, 22),
    floor_color: Tuple[int, int, int] = (35, 35, 38),
) -> np.ndarray:
    """Render one view, return HxWx3 uint8 array."""
    up = np.array([0.0, 1.0, 0.0])
    view = _look_at(eye, target, up)
    proj = _perspective(fov, width / height, 0.1, 50.0)
    mvp = proj @ view

    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    clip = (mvp @ verts_h.T).T
    w_clip = clip[:, 3:4]
    w_clip = np.where(np.abs(w_clip) < 1e-8, 1e-8, w_clip)
    ndc = clip[:, :3] / w_clip

    sx = ((ndc[:, 0] + 1) * 0.5 * width).astype(np.float64)
    sy = ((1 - ndc[:, 1]) * 0.5 * height).astype(np.float64)
    sz = ndc[:, 2]

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    zbuf = np.full((height, width), 1.0, dtype=np.float64)

    tex_h, tex_w = texture.shape[:2]

    face_z = sz[faces].mean(axis=1)
    order = np.argsort(-face_z)

    has_uvs = face_uvs is not None and tex_coords is not None

    for fi in order:
        f = faces[fi]
        x0, y0 = sx[f[0]], sy[f[0]]
        x1, y1 = sx[f[1]], sy[f[1]]
        x2, y2 = sx[f[2]], sy[f[2]]

        cross_val = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if cross_val > 0:
            continue

        xmin = max(0, int(min(x0, x1, x2)))
        xmax = min(width - 1, int(max(x0, x1, x2)))
        ymin = max(0, int(min(y0, y1, y2)))
        ymax = min(height - 1, int(max(y0, y1, y2)))
        if xmin >= xmax or ymin >= ymax:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            continue

        face_on_floor = is_floor is not None and fi < len(is_floor) and is_floor[fi]
        use_texture = has_uvs and not face_on_floor

        if use_texture:
            fuv = face_uvs[fi]
            uv0 = tex_coords[fuv[0]]
            uv1 = tex_coords[fuv[1]]
            uv2 = tex_coords[fuv[2]]

        z0, z1, z2 = sz[f[0]], sz[f[1]], sz[f[2]]

        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < zbuf[py, px]:
                        zbuf[py, px] = z
                        if face_on_floor:
                            img[py, px] = floor_color
                        elif use_texture:
                            u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                            v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                            tx = int(u * tex_w) % tex_w
                            ty = int((1 - v) * tex_h) % tex_h
                            img[py, px] = texture[ty, tx]
                        else:
                            img[py, px] = [128, 128, 128]
    return img


CAMERA_ANGLES = {
    "front_right": {
        "eye": np.array([5.0, 2.5, 5.0]),
        "target": np.array([0.0, -0.1, 0.0]),
        "label": "Front 3/4",
    },
    "front_left": {
        "eye": np.array([-5.0, 2.5, 5.0]),
        "target": np.array([0.0, -0.1, 0.0]),
        "label": "Front Left",
    },
    "rear_right": {
        "eye": np.array([5.0, 2.5, -4.5]),
        "target": np.array([0.0, -0.1, 0.0]),
        "label": "Rear 3/4",
    },
    "side": {
        "eye": np.array([7.5, 1.5, 0.3]),
        "target": np.array([0.0, -0.1, 0.0]),
        "label": "Side",
    },
    "top": {
        "eye": np.array([0.3, 9.0, 0.3]),
        "target": np.array([0.0, 0.0, 0.0]),
        "label": "Top",
    },
    "front": {
        "eye": np.array([0.0, 1.5, 7.5]),
        "target": np.array([0.0, -0.1, 0.0]),
        "label": "Front",
    },
}


def render_skin_preview(
    zip_path: Path,
    out_path: Path,
    obj_path: Path = OBJ_PATH,
    tile_w: int = 640,
    tile_h: int = 480,
    angles: Optional[List[str]] = None,
) -> Path:
    """Render a multi-angle 3D preview sheet for a skin zip.
    Returns the output path."""
    if angles is None:
        angles = ["front_right", "rear_right", "side", "front_left", "top", "front"]

    verts, faces, tex_coords, face_uvs, is_floor = _load_obj(obj_path)

    with zipfile.ZipFile(zip_path) as z:
        tex = np.array(Image.open(io.BytesIO(z.read("Diffuse.dds"))).convert("RGB"))

    renders = []
    for angle_name in angles:
        cam = CAMERA_ANGLES[angle_name]
        view_img = _render_view(
            verts, faces, tex_coords, face_uvs, tex,
            cam["eye"], cam["target"],
            is_floor=is_floor,
            width=tile_w, height=tile_h,
        )
        renders.append((cam["label"], view_img))

    cols = 3
    rows = (len(renders) + cols - 1) // cols
    margin = 8
    label_h = 22
    cell_w = tile_w + margin
    cell_h = tile_h + label_h + margin

    sheet_w = cols * cell_w + margin
    sheet_h = rows * cell_h + margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 22))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    skin_name = zip_path.stem
    draw.text((margin, 2), skin_name, fill=(200, 200, 200), font=font)

    for i, (label, view_arr) in enumerate(renders):
        r = i // cols
        c = i % cols
        x = margin + c * cell_w
        y = margin + label_h + r * cell_h

        draw.text((x + 4, y - label_h + 4), label, fill=(170, 170, 175), font=font)
        tile = Image.fromarray(view_arr)
        sheet.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="3D skin preview renderer")
    ap.add_argument("zip_path", help="Skin zip path")
    ap.add_argument("--out-dir", default="out/previews", help="Output directory")
    ap.add_argument("--width", type=int, default=640, help="Tile width")
    ap.add_argument("--height", type=int, default=480, help="Tile height")
    args = ap.parse_args(argv)

    zip_path = Path(args.zip_path)
    if not zip_path.is_absolute():
        zip_path = (ROOT / zip_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    out_path = out_dir / f"{zip_path.stem}_3d.png"
    result = render_skin_preview(zip_path, out_path, tile_w=args.width, tile_h=args.height)
    print(result.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
