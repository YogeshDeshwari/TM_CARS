# TMNF / TMUF Skinning Notes (Working Research Log)

This repo aims to generate **original** TrackMania (TMNF/TMUF) skins procedurally, inspired by professional skinning practices.

This document is a **living research log**: it captures what we’ve learned so far so future work can continue without losing context.

## Big picture (what we’re building)

We’re building a workflow that reliably outputs **game-ready Stadium skin zips** while staying true to TMNF/TMUF rendering rules:
- **Diffuse alpha is finish/spec**, not transparency (most common Stadium packs).
- Skins should not accidentally “fall back to defaults” for dirt/illum/shadow due to missing files.
- We keep a strong separation between:
  - **what must be correct** (files/formats/channels/packaging)
  - **what is artistic** (palettes, patterns, decals, composition)

### Decision tree (pick your path)

- **If you want a Stadium skin that “just works”**:
  - Use `generate_tmnf_skin.py` with `--base-zip` (recommended).
- **If you want better 3D shading**:
  - Provide a baked prelight and use `--prelight` + `--prelight-strength`.
- **If you want custom car shadow/projection**:
  - Use `--proj-logo` or `--proj-image` (ProjShad hygiene is handled automatically).
- **If you have complex glow (neon/diodes/exhaust heat)**:
  - Bake an Illum and pass `--illum-image` (packs as DXT1 by default).
- **If the pack’s finish convention looks “inverted”**:
  - Try `--finish-invert` (applies to `--finish-alpha auto`).

### Non-goals (for now)

- Full model authoring/import rules (pivots/lights/vertex budgets) are useful, but separate from our day-to-day “generate great Stadium skins” loop.

## Refined process (what we do in this repo)

This section is the **current “best known” workflow** distilled from the sources below + our local dataset inspection.

### Goal A: generate a TMNF/TMUF Stadium skin zip (recommended)

We usually generate a new zip by **starting from a known-good donor/base zip** (a working Stadium mod pack zip)
and replacing textures inside it. This is the most compatible approach because it keeps the required model files and
any pack-specific extras.

- **Inputs**
  - a working base Stadium zip (contains model `.Gbx` files + textures)
  - a style/palette/seed and optional decals/logos
- **Outputs (minimum)**
  - `Diffuse.dds` (generated) — **alpha = finish/spec**, not transparency
  - `Icon.dds` (generated when present in base zip; otherwise left alone)
- **Outputs (strongly recommended / enforced by our tools)**
  - `DiffuseDirty.dds` + `DetailsDirty.dds` (added if missing): Stadium dirt overlay maps
    - default we emit is “no dirt”: alpha=0 (black)
  - `Illum.dds` (added if missing): night illumination map
    - default we emit is “no illum”: RGB=0 (black), written as DXT1

In code, we enforce the “added if missing” behavior for:
- `generate_tmnf_skin.py` reskin mode (`--base-zip ...`)
- `generate_pro_suite.py` and `generate_pro_suite_v2.py` packagers

### Workflow checklist (use this every time)

- **Start from a donor zip** (`--base-zip`):
  - it already contains the correct `.Gbx` model files and pack-specific extras.
- **Generate Diffuse + finish alpha correctly**:
  - default: `--finish-alpha auto --finish-neutral 0x8E`
  - if it looks “wrong” in-game: rerun with `--finish-invert`
- **Always keep prelight as a top-level shading pass** (when used):
  - `--prelight <file>` and tune with `--prelight-strength 0.35..1.0`
- **Ensure aux textures exist** (our tooling does this automatically):
  - `DiffuseDirty.dds`, `DetailsDirty.dds`, `Illum.dds`
- **For projection shadow**:
  - use `--proj-logo` or `--proj-image`
  - we enforce: white background, horizontal flip, white border, DXT1 preference
- **Validate quickly**:
  - open the output zip in-game; check garage lighting + a darker environment (tunnel/garage) to see finish behavior

### Troubleshooting (fast fixes)

- **Colors look washed out / too reflective**:
  - try `--finish-alpha neutral --finish-neutral 0x8E` (baseline), or `--finish-invert` (if auto seems inverted)
- **Colors look too flat / “no car paint”**:
  - reduce matte areas by using a higher neutral (`--finish-neutral 160`) or disable inversion
- **Prelight makes everything too dark**:
  - reduce `--prelight-strength` (try 0.45) or ensure your prelight is near-white in non-shadow regions
- **Glow/neon looks wrong at night**:
  - supply a baked illum with `--illum-image ...` (keeps alpha as mask if present)
- **Dirt overlay looks mismatched**:
  - ensure Dirty maps exist; our tools add them (defaults to “no dirt” alpha=0)
- **Custom ProjShad has artifacts at edges / mirrored text**:
  - use `--proj-image` and let the tool finalize it (mirror + white border)
- **Text appears mirrored on the opposite side**:
  - UV mirroring is normal; rely on `--sidepod-branding-mirror` options or avoid mirrored islands for text-heavy decals

### Goal B: generate standalone textures (fallback / prototyping)

If you don’t have a base zip, we can still generate textures and previews, but you’ll need to package them correctly
for the game yourself. This is best for rapid iteration, not final distribution.

### Compression + size heuristics (practical defaults)

From local dataset inspection + tutorials:
- **Diffuse.dds**: usually DXT5; common sizes 1024 or 2048 (match the base zip you reskin)
- **Details.dds**: often 2048–4096 in “pro” packs (quality signal); alpha affects gloss/finish for details primitives
- **Dirty maps**: 512–1024 recommended; DXT5; alpha is “dirt amount”
- **Illum.dds** (TMNF/TMUF): alpha unused; DXT1 is ideal; 512–1024 usually enough
- **ProjShad.dds**: commonly 512 (Stadium); often DXT1 (no alpha); used by shadow projection plane

### Finish/alpha rule of thumb (critical)

- Treat Diffuse alpha as **material finish**:
  - black-ish = brighter / more matte
  - white-ish = duller / more reflective
- Neutral baseline often used in tutorials: **`0x8E`** (i.e. `8E8E8E`)

Our generator exposes:
- `--finish-alpha {auto|neutral|opaque}`
- `--finish-neutral 0x8E`

### Mirroring rule of thumb (important for text)

Many Stadium UV islands are mirrored. The classic manual rule (useful for procedural stamping):
- **graphics**: mirror once (commonly vertical flip) to map to the other side
- **text**: mirror + flip so it reads correctly on the mirrored side

We implement some of this automatically via “branding mirror” options (sidepod text) and by avoiding placements on
unreliable islands unless explicitly chosen.

### ProjShad rule of thumb (car shadow projection)

From a TMX tutorial (see below):
- **White background**; darker pixels = the visible projection
- **Flip horizontally** so any text doesn’t appear backward when projected
- **Keep a white border** at the edges to avoid projection artifacts
- Prefer **DXT1 (no alpha) + mipmaps** and **512×512** for Stadium

## Core concept: Alpha is not transparency

In TMNF/TMUF mod packs, **Diffuse alpha is usually a material/finish channel**, not transparency.

From a common community tutorial (pasted in chat), the classic mental model is:
- **Black alpha**: highest perceived brightness, minimal/no reflection
- **White alpha**: duller colors, highest reflection/gloss

This implies:
- You can make decals/logos pop by making them **more matte (lower alpha)** on a **glossier (higher alpha)** body.
- If you make whites/glow accents too glossy (high alpha), they can look **grey/dull**.

In this repo:
- `generate_tmnf_skin.py` now supports `--finish-alpha {opaque|neutral|auto}` and `--finish-neutral` (default `0x8E`) to approximate this workflow.

## Alpha workflows (community tutorials) + how to reconcile “contradictions”

Sources:
- ManiaPark “car paint” tutorial (pasted earlier): black alpha = bright/matte, white alpha = dull/shiny.
- User-pasted “[TUTO] alpha” (2009): describes “darker alpha = less bright, lighter alpha = more bright”.

### Why the tutorials appear to contradict

Most of the time, the Diffuse/Details alpha is best understood as a **material/finish control** (reflection/shine),
not literal opacity. Once reflection is involved, *perceived brightness* becomes environment-dependent:
- In a bright environment, higher reflection can make panels appear “brighter”.
- In a dark garage/tunnel, high reflection can make areas look darker (reflecting dark surroundings), while matte areas
  keep their painted color more faithfully.

So “alpha makes brightness” is often a shorthand for “alpha changes how much environment lighting/reflection mixes into the paint”.

### Practical workflow from the [TUTO] alpha excerpt

- Work in grayscale for the alpha channel.
- A fast starting point is to copy a flattened version of your paint into alpha (then adjust brightness/contrast).
- Keep a *separate alpha workflow* (often a simplified B/W version) so you can tune material behavior without changing paint.
- Avoid extreme full-black/full-white alpha over large areas; it can flatten perceived form (too matte) or over-reflect (too mirror-like).
- Glass generally isn’t affected the same way; focus this workflow on `Diffuse.dds` / `Details.dds` (not Illum/Icon).

### Implementation note (this repo)

Because packs/tutorials vary in how they describe the effect, we support inverting the generated finish map when needed:
- see `generate_tmnf_skin.py --finish-invert`

## “How to skin” (ManiaPark 2D tutorial) — layer stack + workflow

Source: ManiaPark thread “**[MP Tutorial - 2D] How to skin [EN] [FR] [RU] [DE]**” (user pasted excerpt, 2011).

This tutorial is written for **CanyonCar**, but the workflow is broadly applicable: a clean PSD structure, prelight as multiply,
and an “alpha edit” workflow that preserves crisp edges.

### PSD layer stack (recommended structure)

The template’s suggested structure (key idea: separate concerns):
- **Wireframe**: orientation only; must be hidden for export; blending often Multiply/Screen/Overlay
- **Prelight**: baked shading; blending = **Multiply**
- **Alpha edit**: grayscale “material/finish paint” used to build the DDS alpha channel; blending = Normal
- **Text/Stickers**: decals/text layers; blending = Normal
- **Paint**: base paint colors; blending = Normal
- **Background layer**: solid background set as background/locked; important for DDS export hygiene

This aligns with our generator model:
- we treat prelight as a multiply over RGB (`--prelight`, `--prelight-strength`)
- we treat Diffuse alpha as finish/material (not transparency)

### Mirroring / text placement

The tutorial demonstrates the classic mirrored-UV issue: for the other side of the car, **rotate text 180°** and reposition
so it reads correctly. (Exact transforms depend on the template/model UV layout.)

### Alpha editing workflow (Photoshop “smart object” trick)

Key idea: build alpha in a dedicated group using duplicates of your paint/text groups:
- make **Alpha edit** visible
- duplicate **Text/Stickers** and relevant **Paint** layers into Alpha edit
- convert groups into **Smart Objects**
- apply **Color Overlay** with chosen gray values (e.g. base light gray + slightly darker gray for decals)
  - preserves outlines/FX cleanly in the alpha output
- copy-merged the alpha edit view and paste into the actual **DDS alpha channel**

Practical tips captured in the thread:
- don’t use overly dark alpha values everywhere (can look worse)
- keep alpha work separated from paint work for speed and control

### Export + packing notes (CanyonCar vs StadiumCar)

The tutorial’s example uses CanyonCar naming (`SkinDiffuse.dds`) and CanyonCar folders; TMNF/TMUF Stadium typically uses:
- Stadium skins/mod packs: `Diffuse.dds`, `Icon.dds`, plus optional `Details.dds`, `ProjShad.dds`, dirty maps, illum maps

In this repo, we generate Stadium-compatible zips by reskinning a known-good Stadium base zip.

## Using Photoshop’s 3D to skin (fast preview workflow)

Source: ManiaPark thread “**[Tutorial - 2D] Using Photoshop's 3D to skin**” (user pasted excerpt, 2010).

This is a **manual workflow** that helps artists preview skins on the 3D model inside Photoshop, reducing “go in game / tweak / re-export” cycles.
It does not replace in-game testing for final export, especially for alpha/finish behavior.

### Requirements / constraints (from the tutorial)

- Photoshop **CS4 Extended or newer** (3D features).
- You need the **prelight**.
- You need the model as a `*.3ds` file.
- Important modeling constraint for Photoshop 3D skinning:
  - the `*.3ds` must be “tweaked/stuck” so that (excluding glass) the model is grouped into **three elements**:
    - `sBody`
    - `dBody`
    - glass
  - if the model has “too many components”, you won’t be able to skin details properly in Photoshop.
- Performance note: Photoshop 3D can be heavy on low-end hardware.

### Core workflow

- Import `*.3ds` into Photoshop via **3D → New Layer from 3D file**.
- In the 3D panel, locate the mesh/material whose **Diffuse/Details** texture you want to edit.
- Load the **prelight** as the texture.
- Double click the texture name to open a `*.psb` tab; paint there like a PSD.
- Switching back to the main PSD updates the 3D preview automatically as PS saves the PSB.

### Important limitations / how it maps to our repo

- You still need to finalize + test the skin in TrackMania:
  - DDS export settings
  - alpha channel / finish behavior
  - any pack-specific expectations
- In this repo, the equivalent “fast iteration” tools are:
  - PNG previews (`--preview-png`)
  - `--prelight` applied procedurally for a more 3D-looking diffuse
  - reskinning a known-good base zip to avoid missing required files

### Related link from the thread

- `https://www.moddingway.com/news/479.html` (mentioned as “nice tuto” for similar skinning workflows)

## Dataset findings (our local examples)

From analyzing example zips (`examples/` + `examples/not_car/`):
- Many “pro” Stadium skins use:
  - `Diffuse.dds`: 2048×2048, typically DXT5
  - `Details.dds`: often **4096×4096**, typically DXT5 (strong quality signal)
  - `ProjShad.dds`: commonly 512×512, often DXT1
  - `DiffuseDirty.dds` / `DetailsDirty.dds`: commonly 1024×1024, often DXT3
- “not_car” total conversion packs often include:
  - `Illum.dds` (glow/emission map)
  - audio (`Engine*.ogg`, `horn.wav`)
  - sometimes non-square textures for custom models

Generated report:
- `out/skin_reports/REPORT.md` (local; not meant to be committed)

## Model ↔ texture semantics (CarPark documentation)

A highly useful reference thread explains how TMUF/TM2 models are structured and what textures/channels do:
- CarPark thread: `https://www.trackmania-carpark.com/forum/viewtopic.php?f=14&t=24143`

Key takeaways relevant to skin generation:
- **Primitives** correspond to texture sets:
  - `s*` uses `Diffuse.dds` (SkinDiffuse); **alpha affects reflection/gloss**
  - `d*` uses `Details.dds` (DetailsDiffuse); **alpha affects gloss**
  - `g*` uses `Details.dds` for glass-like parts (special rules; alpha may behave differently)
- **Dirty textures**:
  - `DiffuseDirty.dds` / `DetailsDirty.dds`: **alpha affects dirt opacity**
- **Illumination**:
  - `Illum.dds` / `DetailsIllum.dds`: emission/glow map
  - (thread notes alpha can be used as a “threshold/intensity” control for different emission behaviors)
- **Projection shadow**:
  - `ProjShad.dds` / `FakeShad.dds` is tied to the `Projshad` dummy plane and simulates the car shadow on the road.

## TMU model upload requirements (CarPark “TMU requirements” thread)

Another useful CarPark thread lists “must have” files/parts for uploads:
- `https://www.trackmania-carpark.com/forum/viewtopic.php?f=14&t=24143` (general model structure)
- “Models TMU requirements - English” (user-provided link in chat; page shows required files/parts)

Practical generator implications:
- When targeting **skin zips** (not custom models), we focus on:
  - `Diffuse.dds` (+ correct alpha semantics)
  - `Icon.dds`
  - optionally `Details.dds`, `Dirty` maps, `ProjShad.dds`, `Illum.dds` if present in the base zip
- When targeting **full model packs**, the requirements expand to include `.Gbx` and additional textures.

## Dirty + Illum textures (why they matter, and channel semantics)

Source: user-pasted excerpt from a CarPark admin/tutorial post:
“**DiffuseDirty, DetailsDirty and Illum: why you need them and how they work**” (2015).

### `DiffuseDirty.dds` and `DetailsDirty.dds` (Stadium-only dirt overlay)

- **Where used**: Stadium environment only (dirt blocks).
- **Why required**: If missing from your zip, the game falls back to the default Stadium dirty textures, which can look wrong/mismatched relative to the player’s vehicle pack. The tutorial explicitly says **they MUST be in the zip**.
- **RGB meaning**: the dirt texture that will be applied.
- **Alpha meaning** (mask for *how much* dirt replaces the base texture):
  - **White alpha**: replace 100% with dirt texture (full dirt)
  - **Black alpha**: replace 0% with dirt texture (no dirt)
- **Export settings**:
  - DDS with alpha, commonly **DXT5**, with **mipmaps** (“interpolated alpha” in common tooling).
- **Recommended size**:
  - Usually **≤ 1024×1024**; **512×512** can be fine (and can look smoother due to blur).
- **“Disable dirt but still comply” trick**:
  - If you don’t want dirt overlay, set a **full black alpha** texture. This still overrides the default Stadium dirties, but results in no dirt being applied.

### `Illum.dds` (night illumination for Details)

- **Where used**: at night for all environments **except Stadium** (per the tutorial).
- **Why required**: If missing, the game uses the environment’s default illum (e.g. CoastCar illum on Coast, etc.), which can mismatch a custom `Details.dds`.
- **RGB meaning**: illuminated version of `Details.dds` (anything mapped to Details can be lit).
- **Alpha meaning**:
  - Tutorial states: **alpha is not used in TM Forever** (but is used in TM2).
- **Export settings**:
  - Can be **DXT1** (no alpha) to save size; DXT5 also works but is wasted weight.
- **Recommended size**:
  - Usually **≤ 1024×1024**, unless extremely small illuminated details need more resolution.
- **“Disable illum but still comply” trick**:
  - Use a **full black** illum (RGB=0). This overrides defaults but lights nothing.

## Illum for complex geometries (baked glow workflow)

Source: user-pasted tutorial describing Illum authoring with **MentalRay** in **3ds Max** (conceptually similar in Maya).

### When this matters

This workflow is relevant when you’re making a **full vehicle pack / custom model** (or highly detailed Details UVs) and want:
- diode lights / neon tubes
- glow from heated metal (exhaust)
- controlled intensities that match geometry rather than hand-paint guesswork

### Key constraints (authoring hygiene)

- Remove glass parts from the scene for the bake (they create messy lighting artifacts).
- Ensure light-emitting parts you want to illuminate **do not overlap in UVs**.
- Keep emissive parts easy to paint in UV space (even “tucked away” islands).

### Baking approach (high-level)

- Add a Skylight but set its intensity low and keep it **off** during the final illum render (the tutorial uses it as a setup aid).
- Use **MentalRay** (Production renderer).
- Use neutral Arch & Design materials:
  - diffuse = white
  - reflectivity = 0
- Paint an **emissive texture** for the parts that should glow:
  - headlight lamps: very bright (near-white)
  - rear diodes: bright colored dots
  - hot exhaust: **much darker** than lamps, with gradient falloff
- Drive **Self Illumination (Glow)** from that painted texture, then **Render To Texture** (same concept as prelight).

### Export/packing notes (TMNF/TMUF)

- In TMNF/TMUF, `Illum.dds` alpha is generally unused; **DXT1** is preferred.
- In game, Illum affects **Details-mapped** parts (anything using `Details.dds` UVs).

### Note: conflicting community descriptions of Diffuse/Details alpha “brightness”

We have at least one pasted note claiming: “darker alpha = less bright, whiter alpha = more bright.”
This conflicts with another widely-circulated TMNF skin tutorial (also pasted earlier) describing:
“black alpha = brightest/matte, white alpha = dullest/shiniest.”

In this repo, we treat Diffuse alpha as a **finish/spec tradeoff** (not transparency), and our `--finish-alpha auto`
generates an alpha map from RGB for readability. If we see in-game evidence that a specific car pack uses an inverted convention,
we should add a switch (or auto-detect) and document the pack.

## Prelight (baked shading) tutorial notes

Source: user-pasted excerpt from “**[Tutorial - 3D] Prelight**” (2009).

### What “prelight” is (in practice)

Prelight here is a **baked lighting/shadow map** for the car body UVs (typically `sBody`). It’s authored in a DCC tool
(3ds Max in the tutorial) using “Render To Texture”, producing an image that encodes how the body should be shaded.

For skinning, it’s commonly applied as a **multiply** onto the Diffuse RGB:
- white / near-white = no change
- darker = darker shading (shadows)

### How to bake it (3ds Max workflow in the tutorial)

- Select `sBody` and apply a **pure white material** to it (ambient/diffuse/specular all 255).
- Add a **skylight** above the body.
- Optional: add a **grey plane below** the car to reduce unrealistic “light from below”.
- Enable **Light Tracer** (if needed).
- Use **Rendering → Render To Texture**:
  - set **padding = 8** (helps avoid seams; “prelight jut out from wireframe”)
  - Add **CompleteMap**
  - Save to **TGA** (`sBodyCompleteMap.tga`) with Target Map Slot = **Diffuse Color**
  - Choose a size (tutorial suggests **2048** for quality; test with 256)
  - Do not save the rendered frame window output; use the exported `sBodyCompleteMap.tga`
- Convert/export to DDS as **DXT1** (no alpha) per the tutorial.

### Implementation impact (this repo)

- Our `generate_tmnf_skin.py --prelight` applies prelight as a **multiply on RGB** while preserving the base Diffuse alpha
  (finish/spec). This matches the above usage.
- If the prelight file contains an alpha channel (e.g., from TGA workflows), we should treat that alpha as a **mask**
  that limits where the multiply applies, to avoid affecting unused UV areas.

### Community consensus (why you should keep the shadows)

Source: ManiaPark thread “**Prelight shadows when skinning...**” (user pasted excerpt, 2012).

- Yes, it matters: keeping prelight shadows makes the result look more realistic/3D.
- The common workflow is: put prelight **over all layers** in **Multiply** mode.

### “Where do shadows/details go?” (wireframe + UV map guidance)

Source: ManiaPark thread “**Making car details and rim textures**” (user pasted excerpt, 2008).

Key takeaways:
- **Shadows**: typically authored via **Render To Texture** (prelight bake).
- **Cracks/body lines/dirt**: often painted by hand in GIMP/Photoshop on the texture.
- **Placement**: “where to put shadows/details” is determined by the **UV map / wireframe**:
  - you paint aligned to the template/wireframe; then applying the prelight gives believable shading automatically.
- Prelight is usually composited with **Multiply** plus a chosen opacity/strength (don’t be afraid to reduce intensity).

How this maps to our repo:
- Use `--prelight` (and tune with `--prelight-strength`) to apply baked shading.
- Use `--uv-debug` to generate an island map and understand where rims/tyres/body islands live in a given base pack.

## ProjShad.dds / car shadow projection (TMX tutorial)

Source: user-pasted excerpt from “**CAR SHADOW TUTORIAL Part 1, 2 and 3**” (TMX / ManiaExchange forum, 2007).

### What ProjShad is (practical)

`ProjShad.dds` is the **projection texture** used for the car shadow under the vehicle. The game projects it under the car,
and it appears in the rotating car preview if included in the skin zip.

### Key semantics + best practices

- **Format**: save as **DXT1 (no alpha)** + **mipmaps** (tutorial recommendation)
- **Color meaning**: tutorial states **white (`#FFFFFF`) is transparent** (i.e., “lets all the light through”)
  - practical rule of thumb: keep a **white background**, draw your shadow/tyre marks/logo in darker tones
- **Horizontal flip**: flip the final image **horizontally** (left↔right) so text isn’t backward when projected.
- **Edge glitch avoidance**: ensure the graphic is **bordered by white**; if dark pixels run to the edge,
  projection artifacts can appear.

### Sizes / aspect ratio notes

- Suggested sizes:
  - Stadium: **512×512**
  - other environments: **256×256**
- The tutorial notes the projection is stretched to fit the car’s bounding box; it suggests working at an approx **1:1.5**
  “car-like” ratio (e.g. 340×512 or 512×768), then squashing to a square for what-you-see-is-what-you-get.
- It also notes non-square sizes can work, but the game may pick the closest mip level and stretch/squish.

### Packaging

- Put `ProjShad.dds` in the same zip as:
  - `Diffuse.dds`
  - `Icon.dds`
- The filename must be exactly **`ProjShad.dds`** or the game will use the default shadow.

## Car paint tutorial (ManiaPark): practical workflow + Alpha “finish” rules

Source: `https://maniapark.com/threadshow/548` (user pasted full text).

### File basics / export settings

- **Default Diffuse template size**: 1024×1024 (classic Stadium template).
- **Export format**:
  - `Diffuse.dds`: **DXT5 / interpolated alpha**, with mipmaps (per Photoshop NVIDIA DDS plugin UI)
  - `Icon.dds`: same compression settings; tutorial uses 128×128 PSD for the icon
- **Packaging**: zip the DDS files (ZIP, not RAR).

### Key rule: Diffuse alpha is “finish”, not transparency

The tutorial’s model matches common TMNF practice:
- **Black alpha**: color as bright as possible, **no reflection**
- **White alpha**: color duller, **max reflection**

It recommends using **neutral alpha = `8E8E8E`** for the “whole car baseline”, then overriding per layer/element:
- base paint: slightly darker than `8E8E8E` for “bright + still reflective”
- graphics/text: can use black alpha for “non-reflective but bright / almost glowing”

This aligns with our implementation defaults:
- `--finish-neutral 0x8E` (default) and `--finish-alpha auto|neutral|opaque`

### Practical compositing/layout workflow (useful for procedural generation)

- **Keep parts as separate layers** while designing; don’t flatten/merge too early:
  - you need “per-element control” for alpha finishing (matte decals on glossy body, etc.)
- **Mirroring tips** (because many Stadium UV islands mirror):
  - graphics/images: design one side, then duplicate + flip vertically to map to the other side
  - text/wording: duplicate + flip vertically **and horizontally** to read correctly on the mirrored side

### Icon workflow tip (manual)

The tutorial notes that TMN/TMNF can generate a higher-quality `Icon.dds` by:
- loading the car in the in-game paint editor
- “Save As” to a temp zip
- copying the generated `Icon.dds` back into the real skin zip (do not overwrite Diffuse)

We generate icons procedurally in `generate_tmnf_skin.py`, but this is a useful fallback when matching legacy behavior.

## Sources queue (to ingest into rules)

The following links are known-good references, but we still need to extract their concrete rules into this doc.

Note: in this environment, web search is currently returning generic summaries rather than the actual forum post content for some of these pages. If you paste the key excerpts (or the specific sections you care about), we can turn them into precise, testable constraints.

### CarPark / community tutorials

- **Import rules (basic)**: `https://www.trackmania-carpark.com/forum/viewtopic.php?f=5&t=4205`
- **CarPark tutorials index**: `https://www.trackmania-carpark.com/tutorial.php`
- **TM-Community TMN “basic”**: `https://tm-community.com/index.php?page=tuto&id=58`
- **TMN “must have” tutorial (Trackmania official forum)**: `https://www.trackmania.com/fr/forum/viewtopic.php?t=13869`
- **UVW Mapping (textures)**: `https://waylon-art.com/uvw_tutorial/uvwtut_01.html`
- **ProjShad.dds thread**: `https://www.trackmania-carpark.com/forum/viewtopic.php?t=1638`
- **Prelight topic**: `https://www.trackmania-carpark.com/forum/viewtopic.php?f=5&t=7549`
- **Prelight topic 2**: `https://www.trackmania-carpark.com/forum/viewtopic.php?f=14&t=10909`

### What we expect to extract from the remaining links (so we can refine further)

- **Import rules / TMN “must have”**: model packing requirements (object naming, pivots, lights, vertex limits)
  - impacts our *future* “full model pack” tooling, not just skins
- **UVW mapping tutorial**: best practices for templates + UV islands
  - impacts “real UV zone masks” and better targeted decal placement
- **ProjShad.dds thread**: exact expectations for `ProjShad.dds` (channels + best size/format)
  - impacts `--proj-image` / `--proj-logo`
- **Prelight topics**: how to author/encode prelight and how to combine it with Diffuse/Details
  - impacts `--prelight` behavior and our default blend mode/strength

### Pivot rotation tutorial (3ds Max)

- **How to turn the pivots with 3Ds max (AVI)**: (need the exact URL / filename)
  - If you can share the link (or a short summary of what it demonstrates), we’ll document the required pivot orientations in one place.

## Where we are in code (current state)

- `generate_tmnf_skin.py`
  - added `pro_mixmatch` livery style (mix of multiple pattern families)
  - added `--palette auto_vibrant` (seed-driven vibrant palette sampler)
  - added TMNF-style finish alpha generation via `--finish-alpha`
  - fixed a bug in base-pack glow recolor loop (indentation)
- `palette_lab.py`: generates high-contrast palettes (complementary / split-complementary / triadic)
- `style_compiler.py`: inspiration-based recipe system (novel outputs, not copies)
- `tmnf_dds.py`: extracted DDS helpers for reuse (reduces coupling)

