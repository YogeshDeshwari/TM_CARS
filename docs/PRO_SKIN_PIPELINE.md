## Pro skin pipeline (TMNF/TMUF Stadium) — how we work in this repo

This is the “battle-tested” workflow we used to produce the recent **high-contrast pro skins** (Fusion Inkblot + Fusion Fade), validate them, preview them, and package them for sharing.

### Goals
- **Game-ready zips**: always compatible with the target Stadium pack.
- **Aesthetics first**: vibrant, contrasty, sharp “night gaming” looks.
- **No-yellow palettes**: avoid yellow hues in all generated palettes.
- **No logos / no sidepod branding** by default (team text only where needed).

### Core constraints (TMNF/TMUF specifics)
- **Diffuse alpha is finish/spec**, not transparency.
- Match the donor pack:
  - Diffuse resolution (often 2048) and format (often **DXT5**)
  - Details resolution (often 4096 in “pro” packs)
  - Mipmaps matter for in-game filtering and distance readability

### Dataset we used for reference
- `examples/` contains known-good Stadium packs (e.g. MINA / KACKY).
- Those packs establish practical “pro” signals:
  - `Diffuse.dds`: 2048 DXT5 + mipmaps
  - `Details.dds`: 4096 DXT5 + mipmaps
  - `ProjShad.dds`: 512 DXT1 + mipmaps
  - Dirt maps: 1024 DXT3/DXT5 + mipmaps (varies)

### Pipeline overview

#### 1) Choose a base zip (compatibility anchor)
We reskin a working Stadium mod zip (keeps `.Gbx` + pack-specific extras):
- Example base: `CH_all_skins/CH_2026.zip`

#### 2) Profile the base zip (one-time per pack)
Create a JSON profile capturing sizes/formats/mipmaps:
- Tool: `tools/profile_base_zip.py`
- Output: `profiles/<sha256>.json`

This lets the generator automatically:
- match base Diffuse dimensions
- respect format/mipmap conventions
- apply pack-specific “recommended” defaults

#### 3) Generate skins (procedural design system)
Primary generator:
- `generate_tmnf_skin.py` with `--base-zip` + `--base-profile`

Key design levers we used:
- **styles**: `pro_fusion_inkblot`, `pro_fusion_fade`
- **palette**: curated + auto (no-yellow enforced in palette sampler)
- **inspiration**: `--inspire-zip` + palette-mapping (composition guide only)
- **finish**: `--finish-alpha auto` (+ optional `--finish-invert`)
- **finish design** (added): `--finish-design {off|edges|sweep}` + `--finish-design-strength` (premium matte/gloss choreography)
- **final grade** (added): `--grade-contrast`, `--grade-color`, `--grade-gamma`, `--vignette-strength`
- **DXT sharpness** (added): `--dxt-sharpen {auto|on|off}` (+ strength/radius/percent/threshold)
- **spatial awareness tooling** (optional): `tools/export_uv_atlas.py` + `tools/make_calibration_packs.py`

#### 4) Sanitize output (banding + watermark best-effort)
Sanitization is applied in generation (`--sanitize` default true):
- deband/dither in low-gradient areas
- conservative watermark suppression for common template regions

#### 5) Validate zips (fast “will this work?” gate)
Tool:
- `tools/validate_skin_zip.py`

Checks include:
- required filenames present
- DDS header sanity (size/format/mips)
- optional comparison to base/profile

#### 6) Preview sheets (fast aesthetic QA)
Tool:
- `tools/preview_skin_zip.py`

Writes a PNG sheet showing key textures + wing crop:
- `out/previews/..._sheet.png`

#### 7) Batch generation (single-command runs)
We created a single-command batch runner:
- `tools/run_night_batch.py`

It generates a “night gaming palette” batch:
- 10 palettes × 2 styles (AcidFade + NeonInk) = 20 skins
- validates each
- emits preview sheets to `out/previews/night/`
- supports `--resume` and writes a JSON report (default: `out/reports/night_batch_report.json`)

#### 8) Package for sharing (no __MACOSX, Discord-friendly)
Finder-created zips often add `__MACOSX/` and `.DS_Store`. We avoid that by packing only the per-skin `.zip` files (no directories).

Tools:
- **Split into <100MB packs**: `tools/pack_ch_all_skins_jan.py`
- **One big pack** (may exceed Discord limits): `tools/pack_ch_all_skins_single.py`
- **Clean an existing zip** (strip `__MACOSX/` + `.DS_Store`): `tools/clean_zip_macos.py`

Outputs we used:
- `CH_Jan_pack_01.zip`, `CH_Jan_pack_02.zip`
- or `CH_Jan_Skins_01.zip`, `CH_Jan_Skins_02.zip`

