#!/usr/bin/env python3
"""
Run a "night gaming" palette batch for CH_2026.

Single-command usage:
  python3 tools/run_night_batch.py

What it does:
  - Generates 20 zips total:
      - 10x pro_fusion_fade (AcidFade-style)
      - 10x pro_fusion_inkblot (NeonInk-style)
  - Validates each zip against a base profile
  - Writes preview contact sheets for each zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Palette:
    slug: str
    base: str
    accent: str
    stripe: str


PALETTES: list[Palette] = [
    Palette("Obsidian_CyanMagenta", "#050613", "#00E5FF", "#FF1FB8"),
    Palette("Navy_IceViolet", "#05071C", "#7C00FF", "#D9F7FF"),
    Palette("Black_CyanWhite", "#04040A", "#00FFFF", "#FFFFFF"),
    Palette("Indigo_TealFuchsia", "#0B0D1E", "#34EDF3", "#F715AB"),
    Palette("DeepBlue_ElectricBlueHotPink", "#08102B", "#00C8FF", "#FF2AA6"),
    Palette("PurpleCore_IceCyan", "#12061A", "#EA00D9", "#7AFBFF"),
    Palette("StealthViolet_NeonCyan", "#060611", "#00FFD5", "#9B30FF"),
    Palette("Midnight_CyanRazorWhite", "#071034", "#0ABDC6", "#F4F6FF"),
    Palette("BluePurpleSplit", "#090B18", "#3B4BFF", "#FF00FE"),
    Palette("CrimsonTech_Cyan", "#08060B", "#FF2244", "#00E5FF"),
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-zip", default="CH_all_skins/CH_2026.zip")
    ap.add_argument(
        "--profile",
        default="profiles/227d422b2c68c62ee0947bd19fb7cd5e7c972a3c96a8bae2fdc3d8f678a38dcc.json",
    )
    ap.add_argument("--out", default="CH_all_skins")
    ap.add_argument("--previews", default="out/previews/night")
    ap.add_argument("--report", default="out/reports/night_batch_report.json", help="Write a JSON report (args, outputs, timings, status).")
    ap.add_argument("--resume", action="store_true", help="Skip generation for outputs that already exist and validate OK; skip preview if it already exists.")
    ap.add_argument("--tag", default="CH")
    ap.add_argument("--team-name", default="Cavern Hunters")
    ap.add_argument("--seed-base", type=int, default=262000, help="Seed offset; each palette uses seed_base+i.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    base_zip = Path(args.base_zip)
    profile = Path(args.profile)
    out_dir = Path(args.out)
    prev_dir = Path(args.previews)
    prev_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    inspire_fade = "examples/Incoming-Winter-Public-SKIN_by_MINA_TM.zip"
    inspire_ink = "examples/Deep-Galaxy-SKIN_by_MINA_TM.zip"

    # Strong but safe "sharper/darker" grading (tuned to avoid banding/posterization).
    fade_grade = dict(contrast="1.38", color="1.12", gamma="0.88", vignette="120")
    ink_grade = dict(contrast="1.34", color="1.10", gamma="0.90", vignette="110")

    made: list[Path] = []
    report: dict[str, Any] = {
        "version": 1,
        "args": vars(args),
        "generated": [],
        "validated_ok": [],
        "previews": [],
        "errors": [],
    }

    def _run_capture(cmd: list[str]) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return int(p.returncode), str(p.stdout or "")

    def _validate_one(zip_path: Path) -> bool:
        rc, out = _run_capture(
            [
                sys.executable,
                "tools/validate_skin_zip.py",
                str(zip_path),
                "--profile",
                str(profile),
                "--strict",
            ]
        )
        if rc == 0 and "OK" in out:
            return True
        report["errors"].append({"step": "validate", "zip": str(zip_path), "returncode": rc, "output": out[-4000:]})
        return False

    def gen_one(name: str, *, style: str, pal: Palette, seed: int, grade: dict[str, str], inspire_zip: str) -> Path:
        out_zip = out_dir / f"{name}.zip"
        cmd = [
            sys.executable,
            "generate_tmnf_skin.py",
            "--name",
            name,
            "--base-zip",
            str(base_zip),
            "--base-profile",
            str(profile),
            "--out",
            str(out_dir),
            "--team-name",
            str(args.team_name),
            "--tag",
            str(args.tag),
            "--style",
            style,
            "--base-color",
            f"{pal.base}ff",
            "--accent-color",
            f"{pal.accent}ff",
            "--stripe-color",
            f"{pal.stripe}ff",
            "--wheel-color",
            pal.accent,
            "--seed",
            str(seed),
            "--finish-alpha",
            "auto",
            "--finish-neutral",
            "150",
            "--finish-invert",
            "--grade-contrast",
            grade["contrast"],
            "--grade-color",
            grade["color"],
            "--grade-gamma",
            grade["gamma"],
            "--vignette-strength",
            grade["vignette"],
            "--sanitize",
            "--proj-wings",
            "--inspire-zip",
            inspire_zip,
            "--inspire-source",
            "diffuse",
            "--inspire-strength",
            "0.48",
        ]
        if args.dry_run:
            print(" ".join(cmd))
            return out_zip
        if bool(args.resume) and out_zip.exists():
            if _validate_one(out_zip):
                report["generated"].append({"name": name, "zip": str(out_zip), "status": "skipped_existing_valid"})
                return out_zip
        try:
            _run(cmd)
            report["generated"].append({"name": name, "zip": str(out_zip), "status": "generated"})
        except subprocess.CalledProcessError as e:
            report["errors"].append({"step": "generate", "name": name, "zip": str(out_zip), "returncode": int(getattr(e, "returncode", 1) or 1)})
            raise
        return out_zip

    # Generate
    for i, pal in enumerate(PALETTES, start=1):
        seed = int(args.seed_base) + i
        fade_name = f"CH_2026_AcidFade_{pal.slug}_N{i:02d}"
        ink_name = f"CH_2026_NeonInk_{pal.slug}_N{i:02d}"
        made.append(gen_one(fade_name, style="pro_fusion_fade", pal=pal, seed=seed, grade=fade_grade, inspire_zip=inspire_fade))
        made.append(gen_one(ink_name, style="pro_fusion_inkblot", pal=pal, seed=seed + 1000, grade=ink_grade, inspire_zip=inspire_ink))

    if args.dry_run:
        return 0

    # Validate (per-zip so resume mode can be reliable and we get better reporting)
    ok: list[Path] = []
    for p in made:
        if _validate_one(p):
            ok.append(p)
            report["validated_ok"].append(str(p))

    # Previews
    for p in ok:
        out_png = prev_dir / (p.stem + "_sheet.png")
        if bool(args.resume) and out_png.exists():
            report["previews"].append({"zip": str(p), "preview": str(out_png), "status": "skipped_existing"})
            continue
        try:
            _run([sys.executable, "tools/preview_skin_zip.py", str(p), "--out-dir", str(prev_dir), "--tile", "360"])
            report["previews"].append({"zip": str(p), "preview": str(out_png), "status": "written"})
        except subprocess.CalledProcessError as e:
            report["errors"].append({"step": "preview", "zip": str(p), "returncode": int(getattr(e, "returncode", 1) or 1)})
            raise

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Generated: {len(made)} zips")
    print(f"Validated OK: {len(ok)} zips")
    print(f"Previews: {prev_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

