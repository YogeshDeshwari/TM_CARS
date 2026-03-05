#!/usr/bin/env python3
"""
Generate skin ZIPs from the preset library.

Usage:
    # Generate all presets
    python generate_preset_batch.py

    # Generate specific presets
    python generate_preset_batch.py gulf_spirit jps_gold stealth_matte

    # List available presets
    python generate_preset_batch.py --list
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from skin_presets import get_preset, list_presets
from pro_skin_engine import ProSkinEngine


def _apply_perpixel_finish(engine: ProSkinEngine, preset: dict, out_dir: str):
    """Replace the Diffuse alpha with per-pixel finish from the hero pattern."""
    hero_pat = preset["role_spec"]["hero"]["pattern"]
    sz = engine.size
    pat = hero_pat.resize((sz, sz), Image.Resampling.LANCZOS) if hero_pat.size != (sz, sz) else hero_pat
    pat_alpha = np.array(pat)[:, :, 3]

    diff_path = engine.out_dir / "Diffuse.png"
    diff_img = Image.open(diff_path).convert("RGBA")
    diff_arr = np.array(diff_img)

    # Only apply per-pixel alpha where there's painted content (not background)
    has_paint = diff_arr[:, :, :3].max(axis=2) > 5
    new_alpha = diff_arr[:, :, 3].copy()
    new_alpha[has_paint] = pat_alpha[has_paint]
    diff_arr[:, :, 3] = new_alpha

    result = Image.fromarray(diff_arr, "RGBA")
    result.save(diff_path)

    # Re-encode the DDS and update the ZIP
    from tmnf_dds import build_dds_dxt5_bytes
    dds_bytes = build_dds_dxt5_bytes(result, mipmaps=True)
    dds_path = engine.out_dir / "Diffuse.dds"
    dds_path.write_bytes(dds_bytes)

    zip_path = Path(out_dir) / f"{engine.team_name}.zip"
    if zip_path.exists():
        import zipfile as zf
        all_files = {}
        with zf.ZipFile(zip_path, "r") as z:
            all_files = {n: z.read(n) for n in z.namelist()}
        all_files["Diffuse.dds"] = dds_bytes
        with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
            for name, data in all_files.items():
                z.writestr(name, data)
    print(f"  Applied per-pixel finish alpha from pattern")


def _apply_illum_glow(engine: ProSkinEngine, preset: dict, glow_cfg: dict):
    """Build Illum map from the hero pattern's green/dark boundary edges."""
    hero_spec = preset["role_spec"].get("hero", {})
    pattern = hero_spec.get("pattern")
    if pattern is None:
        return

    sz = engine.size
    pat = pattern.resize((sz, sz), Image.Resampling.LANCZOS) if pattern.size != (sz, sz) else pattern
    arr = np.array(pat)

    green = arr[:, :, 1].astype(np.float64) / 255.0
    intensity = glow_cfg.get("intensity", 0.6)
    blur_r = glow_cfg.get("blur", 5)
    color = glow_cfg.get("color", (0, 255, 0))
    edge_mode = glow_cfg.get("edge_mode", False)

    if edge_mode:
        green_mask = Image.fromarray((green * 255).astype(np.uint8), "L")
        from PIL import ImageFilter as IF
        dark_mask = Image.fromarray(((1.0 - green) * 255).astype(np.uint8), "L")
        dark_blur = np.array(dark_mask.filter(IF.GaussianBlur(blur_r * 2)), dtype=np.float64) / 255.0
        edge = green * dark_blur
        edge = np.array(Image.fromarray(
            (edge * 255).astype(np.uint8), "L"
        ).filter(IF.GaussianBlur(blur_r)), dtype=np.float64) / 255.0
        glow_arr = edge
    else:
        glow_mask = Image.fromarray((green * 255).astype(np.uint8), "L")
        glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(blur_r))
        glow_arr = np.array(glow_mask).astype(np.float64) / 255.0

    illum_arr = np.zeros((sz, sz, 4), dtype=np.uint8)
    for ch in range(3):
        illum_arr[:, :, ch] = np.clip(glow_arr * color[ch] * intensity, 0, 255).astype(np.uint8)
    illum_arr[:, :, 3] = 255

    illum_img = Image.fromarray(illum_arr, "RGBA")
    from PIL import ImageChops
    engine.illum = ImageChops.screen(engine.illum, illum_img)
    mode = "edge" if edge_mode else "full"
    print(f"  Applied Illum glow ({mode}): color={color}, intensity={intensity}, blur={blur_r}")


def generate_one(preset_name: str, out_dir: str = "out") -> Path:
    """Generate a single skin from a preset name. Returns the ZIP path."""
    t0 = time.time()
    preset = get_preset(preset_name, size=2048)
    meta = preset["meta"]
    print(f"\n{'='*60}")
    print(f"  {meta['name']}  ({meta['category']})")
    print(f"  {meta['description']}")
    print(f"{'='*60}")

    team = meta["name"]
    engine = ProSkinEngine(team_name=team, full_skin=True)
    engine.load_uv_geometry()

    engine.paint_by_role(preset["role_spec"])

    dl = preset.get("design_layers", [])
    if dl:
        engine.apply_design_layers(dl)

    if preset.get("_oklch_fade"):
        hero_col = preset["role_spec"]["hero"]["color"]
        sec_col = preset["role_spec"]["secondary"]["color"]
        engine.apply_oklch_fade(hero_col, sec_col)

    hg = preset.get("hero_gradient", {})
    if hg:
        engine.apply_hero_gradient(**hg)

    pl = preset.get("prelight", {})
    if pl:
        engine.apply_prelight(**pl)

    glow_cfg = preset.get("glow")
    if glow_cfg:
        _apply_illum_glow(engine, preset, glow_cfg)

    engine.save()

    # Post-process: if the pattern has per-pixel finish alpha, apply it
    # to the saved Diffuse.dds (the engine's _finalize_finish_channels
    # overwrites with per-island values, but we want per-pixel for depth)
    hero_pat = preset["role_spec"].get("hero", {}).get("pattern")
    if hero_pat is not None:
        pat_alpha = np.array(hero_pat.convert("RGBA"))[:, :, 3]
        # Only apply if the pattern actually varies its alpha (not all 255)
        if pat_alpha.min() < 200 and pat_alpha.max() > pat_alpha.min() + 20:
            _apply_perpixel_finish(engine, preset, out_dir)

    elapsed = time.time() - t0
    zip_path = Path(out_dir) / f"{team}.zip"
    print(f"  Done in {elapsed:.1f}s -> {zip_path}")
    return zip_path


def main():
    args = sys.argv[1:]

    if "--list" in args:
        presets = list_presets()
        print(f"\n{'Name':<22s} {'Category':<18s} Description")
        print("-" * 78)
        for p in presets:
            print(f"{p['name']:<22s} {p['category']:<18s} {p['description'][:60]}")
        print(f"\n{len(presets)} presets available.")
        return

    if args:
        names = [a for a in args if not a.startswith("-")]
    else:
        names = [p["name"] for p in list_presets()]

    print(f"Generating {len(names)} skin(s)...")
    zips = []
    for name in names:
        try:
            zp = generate_one(name)
            zips.append(zp)
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Generated {len(zips)}/{len(names)} skins.")
    for z in zips:
        print(f"  {z}")


if __name__ == "__main__":
    main()
