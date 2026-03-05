## Technical deep dive: current issues + where we can improve

This is a practical checklist of issues observed during iteration, plus concrete improvements (code + workflow).

### A) Tooling / reliability issues

- **Occasional process aborts**
  - Symptom: commands sometimes fail with "Aborted" even when short.
  - Mitigation we adopted:
    - prefer single long-running batch scripts (`tools/run_night_batch.py`) vs many separate invocations
    - avoid massive chained shell commands
  - Improvement idea:
    - add a "resume" mode to batch scripts (skip already-generated outputs)
    - write progress JSON (what succeeded/failed) for crash recovery
  - Status:
    - implemented in `tools/run_night_batch.py` via `--resume` + `--report`

- **`__MACOSX/` folder in zips**
  - Symptom: Finder-created zips include `__MACOSX/` and `.DS_Store`, sometimes causing import issues.
  - Fix we implemented:
    - pack only the per-skin `.zip` files (no directories) using ZIP_STORED
    - tools: `tools/pack_ch_all_skins_jan.py`, `tools/pack_ch_all_skins_single.py`
  - Improvement idea:
    - add a "zip cleaner" tool that removes `__MACOSX/` entries from an existing zip (no re-zip required)
  - Status:
    - implemented: `tools/clean_zip_macos.py`

### B) Rendering / game-compat issues

- **Finish/spec channel ambiguity (pack-dependent)**
  - Symptom: "inverted" looking finish depending on environment/mod pack.
  - Current controls:
    - `--finish-alpha auto|neutral|opaque`
    - `--finish-neutral`
    - `--finish-invert`
  - Improvement idea:
    - per-pack "finish profile" heuristics (auto-detect inversion by sampling base alpha patterns + known lighting expectations)
    - optional per-island finish masks (matte top, glossy sides, etc.)

- **DXT compression eats thin details**
  - Symptom: ultra-thin lines/halftones collapse or get muddy after DXT5.
  - Current mitigations:
    - micro-hatch / controlled noise
    - final contrast punch
  - Improvement idea:
    - "DXT-aware" thickness rules (minimum pixel widths at 2048)
    - optional pre-sharpen pass on edge masks before compression
    - consistent dithering strategy for gradients
  - Status:
    - implemented (safe default): `generate_tmnf_skin.py` `--dxt-sharpen auto` (+ knobs)

### C) Design system limitations (aesthetic ceiling)

- **Need stronger UV-awareness**
  - Symptom: a global design can land in awkward places across parts.
  - Current:
    - standard Stadium island detection using known template Diffuse
    - wing rects hard-coded for standard Stadium
  - Improvement idea:
    - maintain a stable "part map" for Stadium (sidepods, fenders, mudguards, roof, nose)
    - allow per-part rules:
      - "busy" only on hero panels
      - keep roof/cockpit calmer
      - enforce clean separators at part boundaries
  - New tooling (spatial awareness):
    - `tools/export_uv_atlas.py`: export UV island atlas (PNG + JSON)
    - `tools/make_calibration_packs.py`: generates `UV_DEBUG` / `UV_DEBUG_ALL` zips + previews

- **Material choreography (matte/gloss placement)**
  - Symptom: great colors still look "flat" without deliberate finish design.
  - Current:
    - auto finish map driven by RGB
    - new grading knobs: `--grade-*` + `--vignette-strength`
  - Improvement idea:
    - explicit finish layers:
      - glossy pinstripes
      - matte base zones
      - "clearcoat sweep" aligned with band direction
  - Status:
    - implemented (opt-in): `generate_tmnf_skin.py` `--finish-design {edges|sweep}`

### D) Mudguard color harmonization (planned / not implemented)

Goal:
- Ensure the 4 mudguard/tyre-guard regions read as a **cohesive intentional color** (default on), while still letting you override.

Why this is hard to “just do”:
- You need reliable part identification (mudguards/arches) across packs, which is a UV/mesh mapping problem.

Proposed solution:
- Use the spatial-awareness tooling (`tools/export_uv_atlas.py` + calibration zips) to build a per-pack part map,
  then apply a luma-preserving tint inside those masks.

