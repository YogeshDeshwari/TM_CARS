#!/usr/bin/env python3
"""
Generate calibration zips + previews for "hands-on car" spatial awareness.

This answers:
  - Which visible areas are driven by Diffuse.dds vs Details.dds?
  - Where are mirrored/overlapped regions?

Outputs:
  - out/calibration/UV_DEBUG.zip
  - out/calibration/UV_DEBUG_ALL.zip (also overwrites Details/Dirty with labeled debug patterns)
  - out/calibration/DIFFUSE_PROBE.zip (loud diffuse-only probe texture)
  - out/calibration/DETAILS_PROBE.zip (loud details-only probe texture)
  - preview sheets under out/calibration/previews/

Usage:
  python3 tools/make_calibration_packs.py --base-zip CH_all_skins/CH_2026.zip --profile profiles/<sha>.json
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _font(px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", px)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", px)
        except Exception:
            return ImageFont.load_default()


def _probe_texture(size: int, *, label: str, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    step = max(16, size // 24)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                d.rectangle((x, y, x + step - 1, y + step - 1), fill=fg)
    # crosshair
    d.line((0, size // 2, size, size // 2), fill=(255, 255, 255), width=max(1, size // 512))
    d.line((size // 2, 0, size // 2, size), fill=(255, 255, 255), width=max(1, size // 512))
    # label
    f = _font(max(18, size // 18))
    d.text((size // 2, int(size * 0.08)), label, fill=(255, 255, 255), anchor="mm", font=f)
    d.text((size // 2, int(size * 0.92)), label, fill=(255, 255, 255), anchor="mm", font=f)
    return img.convert("RGBA")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-zip", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out-dir", default="out/calibration")
    ap.add_argument("--previews", default="out/calibration/previews")
    args = ap.parse_args(argv)

    base_zip = (ROOT / str(args.base_zip)).resolve()
    prof = (ROOT / str(args.profile)).resolve()
    out_dir = (ROOT / str(args.out_dir)).resolve()
    prev_dir = (ROOT / str(args.previews)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    # Determine sizes from base zip.
    with zipfile.ZipFile(base_zip, "r") as z:
        dh = z.open("Diffuse.dds").read(128)
        diffuse = Image.open(io.BytesIO(z.read("Diffuse.dds"))).convert("RGBA")
        size = diffuse.size[0]
        has_details = "Details.dds" in z.namelist()
        details_size = None
        if has_details:
            details = Image.open(io.BytesIO(z.read("Details.dds"))).convert("RGBA")
            details_size = details.size[0]

    # 1) UV debug zips (existing generator feature).
    _run(
        [
            sys.executable,
            "generate_tmnf_skin.py",
            "--name",
            "UV_DEBUG",
            "--base-zip",
            str(base_zip),
            "--base-profile",
            str(prof),
            "--out",
            str(out_dir),
            "--uv-debug",
        ]
    )
    _run(
        [
            sys.executable,
            "generate_tmnf_skin.py",
            "--name",
            "UV_DEBUG_ALL",
            "--base-zip",
            str(base_zip),
            "--base-profile",
            str(prof),
            "--out",
            str(out_dir),
            "--uv-debug",
            "--uv-debug-all",
        ]
    )

    # 2) Diffuse/Details probe zips (loud checkers + labels).
    # We rebuild a new zip from base, replacing only one texture at a time.
    def build_probe_zip(out_name: str, *, replace: dict[str, Image.Image]) -> Path:
        out_path = out_dir / f"{out_name}.zip"
        with zipfile.ZipFile(base_zip, "r") as zin, zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                name = info.filename
                if name in replace:
                    bio = io.BytesIO()
                    replace[name].save(bio, format="PNG")
                    # Store probe images as DDS? Not worth it; but TMNF needs DDS.
                    # So instead we call generator to write DDS; this probe zip is for inspection via preview tool,
                    # and can also be used in-game if you convert probes to DDS later.
                    zout.writestr(name.replace(".dds", ".png"), bio.getvalue())
                else:
                    if hasattr(info, "is_dir") and info.is_dir():
                        zout.writestr(zipfile.ZipInfo(name), b"")
                    else:
                        zout.writestr(info, zin.read(name))
        return out_path

    # These probes are for *understanding*; for in-game use you still rely on UV_DEBUG zips.
    diffuse_probe = _probe_texture(size, label="DIFFUSE PROBE", fg=(0, 230, 255), bg=(20, 0, 40))
    details_probe = _probe_texture(int(details_size or size), label="DETAILS PROBE", fg=(255, 40, 170), bg=(0, 18, 40))
    build_probe_zip("DIFFUSE_PROBE_AS_PNG", replace={"Diffuse.dds": diffuse_probe})
    if has_details:
        build_probe_zip("DETAILS_PROBE_AS_PNG", replace={"Details.dds": details_probe})

    # 3) Previews for the real in-game calibration zips.
    for zp in [
        out_dir / "UV_DEBUG.zip",
        out_dir / "UV_DEBUG_ALL.zip",
    ]:
        _run([sys.executable, "tools/preview_skin_zip.py", str(zp), "--out-dir", str(prev_dir), "--tile", "360"])

    print("Calibration zips:")
    print(out_dir / "UV_DEBUG.zip")
    print(out_dir / "UV_DEBUG_ALL.zip")
    print("Previews:")
    print(prev_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

