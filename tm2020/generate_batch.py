#!/usr/bin/env python3
"""
CLI for generating TM2020 skin packs.

Usage:
    python3 -m tm2020.generate_batch <preset_name> [preset_name2 ...]
    python3 -m tm2020.generate_batch --list
    python3 -m tm2020.generate_batch --all

Output lands in out_tm2020/<preset_name>.zip
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure parent dir on path for skin_utils imports
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tm2020.presets import get_preset, list_presets
from tm2020.engine import TM2020SkinEngine

OUT_DIR = _root / "out_tm2020"


def main() -> None:
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print("TM2020 Skin Generator")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python3 -m tm2020.generate_batch <preset> [preset2 ...]")
        print("  python3 -m tm2020.generate_batch --list")
        print("  python3 -m tm2020.generate_batch --all")
        print()
        print(f"Output directory: {OUT_DIR}")
        return

    if "--list" in args:
        presets = list_presets()
        print(f"Available TM2020 presets ({len(presets)}):")
        print("-" * 50)
        for name, desc in presets:
            print(f"  {name:<30s} {desc}")
        return

    if "--all" in args:
        names = [name for name, _ in list_presets()]
    else:
        names = args

    total_start = time.time()
    generated = []

    for name in names:
        print(f"\n{'=' * 50}")
        print(f"Generating: {name}")
        print(f"{'=' * 50}")

        t0 = time.time()
        try:
            spec = get_preset(name)
            engine = TM2020SkinEngine(spec)
            zip_path = engine.generate(OUT_DIR)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")
            generated.append((name, zip_path))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    total = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"Generated {len(generated)}/{len(names)} skins in {total:.1f}s")
    for name, path in generated:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
