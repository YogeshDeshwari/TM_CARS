# TM_CARS -- Procedural Skin Generator for TMNF/TMUF

Generates complete `.zip` skin packs for TrackMania Nations Forever / United Forever,
ready to drop into the game's `Skins/Vehicles/StadiumCar/` folder.

## Quick Start

```bash
pip install -r requirements.txt
python3 generate_creative_skins.py
```

Output lands in `out/<skin_name>.zip`.

## How It Works

1. A **pattern generator** function produces a 2048x2048 RGBA image (the body paint).
2. `ProSkinEngine` composites it onto the car's UV layout, generates dirt maps, encodes DDS textures with mipmaps, and packages everything into a game-ready ZIP.
3. Optionally, `tire_customizer` paints custom wheel spokes, sidewall brand text, and accent-colored tire shoulders onto `Details.dds`.

## Creating a Skin

In `generate_creative_skins.py`, call `build_skin()`:

```python
from generate_creative_skins import build_skin

# Minimal -- just a pattern
build_skin("my_skin", my_pattern_function)

# With custom tires
build_skin("my_skin", my_pattern_function, tire_config=dict(
    brand_text="BRIDGESTONE",
    model_text="POTENZA",
    accent_color=(0, 200, 210),
    spoke_count=12,
))
```

### tire_config options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `brand_text` | `"BRIDGESTONE"` | Upper sidewall arc text |
| `model_text` | `"POTENZA"` | Lower sidewall arc text |
| `text_color` | `(220, 220, 225, 240)` | RGBA for sidewall text |
| `accent_color` | `(0, 200, 210)` | RGB for tire shoulder band |
| `rubber_color` | `(18, 18, 22)` | RGB for tire rubber |
| `spoke_color` | `(200, 203, 210)` | RGB for wheel spokes |
| `spoke_bg_color` | `(75, 78, 85)` | RGB between spokes |
| `spoke_count` | `12` | Number of spokes |
| `caliper_color` | `None` | Optional brake caliper tint |
| `lugnut_color` | `None` | Optional lug nut tint |

## Project Structure

```
generate_creative_skins.py   -- Entry point and pattern generators
pro_skin_engine.py           -- Core engine (UV compositing, DDS, ZIP)
tire_customizer.py           -- Tire/wheel customization for Details.dds
car_geometry.py              -- UV island classification and roles
layer_stack.py               -- Layer compositing with blend modes
skin_utils.py                -- Procedural pattern generators and helpers
tmnf_dds.py                  -- DDS encoding (DXT1/DXT5 with mipmaps)
palette_lab.py               -- OKLCH color utilities
assets/
  base_car/CH_2026.zip       -- Base car pack (meshes + default textures)
  masks/                     -- Paint and chassis masks
  uv_atlas/                  -- UV island atlas (JSON + diagnostic PNG)
models/
  StadiumCar.obj             -- Reference mesh for UV polygon extraction
stickers/                    -- Sticker/logo images for pattern use
tools/                       -- Validation and preview utilities
out/                         -- Generated skins (gitignored)
```

## Dependencies

- Python 3.9+
- Pillow, NumPy, SciPy (see `requirements.txt`)
- .NET SDK (only if using `tools/remove_guards/` for mudguard removal)
