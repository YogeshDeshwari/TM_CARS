# Custom Design Patterns and Wheel Customization

How we create custom procedural skin designs and customize wheels in this repo.

---

## Part 1: Custom Procedural Designs (Case Study: Weaponized 115)

### The Problem

We wanted to recreate the Call of Duty Black Ops 2 "Weaponized 115" camo look:
organic flowing regions of bright neon green and deep black, like a radioactive
mineral with swirling veins. Geometric patterns (circuits, hexagons, etc.) looked
nothing like it. We needed a fundamentally different approach.

### The Solution: Turbulent Marble Noise

The algorithm is classic **marble noise**: `sin(coordinate + amplitude * turbulence)`.
This produces flowing, organic vein-like patterns that look like natural stone or
mineral formations.

#### Building Blocks

**1) Fractal Brownian Motion (fBM)** -- `_fbm_noise()` in `skin_utils.py`

Generates smooth random fields by summing interpolated noise at increasing
frequencies with decreasing amplitude. Each octave doubles the frequency and
halves the contribution. Result: smooth but detailed random terrain.

```
result = 0
amplitude = 1.0
for each octave:
    noise = random grid at this frequency, upscaled via bicubic interpolation
    result += noise * amplitude
    amplitude *= 0.5
normalize to [0, 1]
```

**2) Turbulence Field** -- `_turbulence_field()` in `skin_utils.py`

Like fBM but sums `abs(noise - 0.5)` instead of raw noise. This creates sharper
ridges and more dramatic features -- the "chaos" that makes marble look natural.

**3) Marble Veins** -- the core of `generate_weaponized_115()`

Apply `sin()` to coordinate grids distorted by the turbulence field. Multiple
sine waves at different angles are blended for isotropy (so the pattern does not
have a single dominant direction):

```
marble_a = sin(y_grid + ascent * x_grid + amplitude * turbulence_1)     # primary
marble_b = sin(x_grid*0.8 + y_grid*0.4 + amplitude*0.75 * turbulence_2) # secondary
marble_c = sin(x_grid*0.3 - y_grid*0.9 + amplitude*0.6 * turbulence_3)  # tertiary

raw = 0.40 * marble_a + 0.35 * marble_b + 0.25 * marble_c
```

#### Domain Warping (the key to fluid/chaotic patterns)

Domain warping distorts the coordinate grids BEFORE the sine calculation, making
the veins flow and twist in unexpected ways. We apply two levels:

```python
if warp_strength > 0:
    warp_x = _fbm_noise(size, 5, seed+500)
    warp_y = _fbm_noise(size, 5, seed+600)
    warp_scale = warp_strength * 2 * pi * vein_freq
    x_grid += (warp_x - 0.5) * warp_scale
    y_grid += (warp_y - 0.5) * warp_scale

    # Second level (warp-on-warp) for extra chaos
    warp_x2 = _fbm_noise(size, 4, seed+700)
    warp_y2 = _fbm_noise(size, 4, seed+800)
    x_grid += (warp_x2 - 0.5) * warp_scale * 0.5
    y_grid += (warp_y2 - 0.5) * warp_scale * 0.5
```

`warp_strength=0` gives clean sine veins. `warp_strength=0.35` gives the fluid
Weaponized 115 look. Higher values get increasingly chaotic.

#### Shaping and Color Mapping

The raw marble value is shaped to control the green/dark ratio:

```
raw_01 = sqrt(0.5 * (raw + 1))        # compress darks
marble_val = clip((raw_01 - 0.62) / sharpness, 0, 1)
```

- `threshold` (0.62) controls the overall dark/green balance
- `sharpness` (0.20-0.28) controls boundary steepness

Color mapping uses a three-stop gradient: `dark_color -> green_mid -> green_peak`
with the crossover at marble_val = 0.4.

#### Post-Processing

1. **Green glow bloom**: Gaussian blur at green/dark boundaries adds subtle glow
2. **Bright hotspots**: fBM-modulated highlights in green regions for depth
3. **Surface grit**: Fine random noise (+/-8) for texture

### Key Parameters for Custom Designs

| Parameter | What it controls | Weaponized 115 values |
|-----------|-----------------|----------------------|
| `vein_freq` | Number of veins (higher = more, smaller blobs) | 14.0 (hero), 16.0 (secondary), 18.0 (accent) |
| `turb_amplitude` | Chaos level of vein distortion | 14.0-16.0 |
| `octaves` | Detail levels in noise | 7 |
| `sharpness` | Boundary hardness (lower = sharper) | 0.20-0.24 |
| `warp_strength` | Fluid/organic feel (0=off, 0.3+=fluid) | 0.35-0.40 |
| `green_peak` | Brightest color | (40, 240, 70) |
| `green_mid` | Mid-tone color | (10, 120, 25) |
| `dark_color` | Dark base color | (3, 5, 2) |

### How to Create a New Custom Design

1. **Choose your colors**: Pick `peak`, `mid`, and `dark` RGB values for your
   two-tone look. The algorithm does dark-to-bright, so think of it as
   "shadow color" and "highlight color."

2. **Tune vein_freq**: Start at 4.0 (large blobs) and increase. 14-18 gives
   fine detail. Each UV role (hero/secondary/accent) should use slightly
   different frequencies so they do not look identical.

3. **Tune turb_amplitude**: 10-16 is the sweet spot. Lower = calmer veins,
   higher = more turbulent.

4. **Enable domain warping**: `warp_strength=0.35` is a good starting point.
   This is what makes it look organic vs geometric.

5. **Adjust sharpness**: 0.20 = very sharp boundaries, 0.30 = softer gradients.

6. **Use different seeds**: Each seed produces a completely different pattern.
   Use different seeds for hero, secondary, and accent roles.

7. **Register the preset** in `skin_presets.py`:
```python
@_register("artistic", "my_custom_skin", "Description here")
def _my_custom(size: int) -> Dict[str, Any]:
    pattern = skin_utils.generate_weaponized_115(
        size,
        green_peak=(R, G, B),    # your peak color
        green_mid=(R, G, B),     # your mid color
        dark_color=(R, G, B),    # your dark color
        vein_freq=14.0,
        turb_amplitude=15.0,
        octaves=7,
        sharpness=0.22,
        warp_strength=0.35,
        seed=115,
    )
    return {
        "role_spec": {
            "hero":      {"color": (R,G,B), "pattern": pattern, "pattern_opacity": 1.0, ...},
            "secondary": {"color": (R,G,B), "pattern": pattern_b, "pattern_opacity": 1.0, ...},
            "accent":    {"color": (R,G,B), "pattern": pattern_c, "pattern_opacity": 1.0, ...},
            "darken":    {"color": (R,G,B), ...},
            "neutral":   {"color": (R,G,B), ...},
        },
        "design_layers": [],
        "hero_gradient": {"top_lighten": 0.01, "bottom_darken": 0.02},
        "prelight": {"strength": 0.18},
    }
```

8. **Generate**: `python run_skins_v2.py` with your preset name, or call
   `ProSkinEngine` directly.

### Color Scheme Ideas (Not Just Green)

The marble noise generator works with ANY two-tone color scheme. Just change
the RGB values. Despite the parameter names saying "green", they accept any color:

- **Weaponized (original)**: dark=(3,5,2), mid=(10,120,25), peak=(40,240,70) -- green/black
- **Lava/Magma**: dark=(10,2,0), mid=(180,40,5), peak=(255,120,20) -- red-orange/black
- **Frost/Ice**: dark=(2,5,10), mid=(40,120,180), peak=(100,200,255) -- blue/dark
- **Toxic Purple**: dark=(5,2,8), mid=(80,20,140), peak=(180,60,255) -- purple/black
- **Gold Vein**: dark=(5,4,2), mid=(120,90,20), peak=(240,200,60) -- gold/black

---

## Part 2: Wheel Customization (Stickers, Colors)

### Details.dds UV Layout

Wheels and tires are textured via `Details.dds` (not `Diffuse.dds`). The
Details.dds UV layout at 4096x4096 contains these wheel-related regions:

```
+------------------+-------------------------------+---------+
|                  |                               |         |
|  TIRE TREAD      |     WHEEL FACE CIRCLE         | struct  |
|  (0,0)-(640,960) |  center (1161,506) r=490      | parts   |
|  dark rectangle  |  rim + spokes visible          | (not    |
|                  |  bbox: (669,22)-(1648,1000)    | wheel)  |
|                  |                               |         |
+------------------+------+--------+--------+------+---------+
|  BRAKE DISC      | HUB  | HUB    |        |               |
|  (0,1000)-(380)  | CAPS | CAPS   | fender |               |
|  circle          |      |        | arc    |               |
+------------------+------+--------+--------+               |
|  TIRE SIDEWALL BAR                        |               |
|  (50,1430)-(1380,1560)                    |               |
|  horizontal rectangle                     |               |
+-------------------------------------------+---------------+
```

### Which Regions to Paint

For smiley/sticker coverage of the full wheel:

| Region | Coordinates (4096) | What it is |
|--------|-------------------|------------|
| Wheel face circle | center (1161,506), radius 490 | The rim you see from the side |
| Tire tread | rectangle (0,0)-(640,960) | Circumferential rubber surface (road contact) |
| Tire sidewall | rectangle (50,1430)-(1380,1560) | Flat side of the tire (visible from side) |

DO NOT paint on the right column (structural parts like headlights, brackets),
the brake disc, hub caps, or the fender arc -- those are not wheels.

### How to Apply Stickers to Wheels

#### Step 1: Build the wheel mask

```python
from PIL import Image, ImageDraw

def build_wheel_mask(W):
    """W = texture width (e.g. 4096)."""
    mask = Image.new("L", (W, W), 0)
    draw = ImageDraw.Draw(mask)

    # Wheel face circle
    cx, cy, r = 1161, 506, 490
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

    # Tire tread
    draw.rectangle([0, 0, 640, 960], fill=255)

    # Tire sidewall
    draw.rectangle([50, 1430, 1380, 1560], fill=255)

    return mask
```

#### Step 2: Fill with base color, then tile stickers

```python
from PIL import ImageChops

# Yellow base fill (masked to wheel regions)
fill = Image.new("RGBA", (W, H), (230, 220, 40, 255))
alpha = ImageChops.multiply(fill.getchannel("A"), wheel_mask)
fill.putalpha(alpha)
details_img = Image.alpha_composite(details_img, fill)

# Tile sticker pattern (masked to wheel regions)
pattern = tile_stickers(sticker_img, tile_size=110, canvas_size=(W, H))
alpha2 = ImageChops.multiply(pattern.getchannel("A"), wheel_mask)
pattern.putalpha(alpha2)
details_img = Image.alpha_composite(details_img, pattern)
```

#### Step 3: Encode and package

```python
from tmnf_dds import build_dds_dxt5_bytes

dds_bytes = build_dds_dxt5_bytes(details_img, mipmaps=True)
# Write into the skin zip as "Details.dds"
```

### Sticker Tiling Function

```python
import random
from PIL import Image

def tile_stickers(sticker, tile_size, canvas_size, seed=42):
    sw, sh = sticker.size
    sc = tile_size / float(max(sw, sh))
    st = sticker.resize((int(sw*sc), int(sh*sc)), Image.LANCZOS)
    step = int(max(st.size) * 0.85)

    rng = random.Random(seed)
    canvas = Image.new("RGBA", canvas_size, (0,0,0,0))
    for y in range(-step, canvas_size[1]+step, step):
        for x in range(-step, canvas_size[0]+step, step):
            angle = rng.uniform(-15, 15)
            tile = st.rotate(angle, expand=True, resample=Image.BICUBIC)
            canvas.alpha_composite(tile, (x, y))
    return canvas
```

### Sticker Preparation (Background Removal)

Sticker images often have white/light backgrounds that must be removed:

```python
def prepare_sticker(path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    brightness = (r + g + b) / 3
    bg = (brightness > 200) & (np.abs(r-g) < 30) & (np.abs(g-b) < 30)
    arr[bg, 3] = 0   # make background transparent
    img = Image.fromarray(arr, "RGBA")
    bb = img.getchannel("A").getbbox()
    return img.crop(bb) if bb else img
```

### Tile Size Guide

| Tile size (px at 4096) | Effect |
|----------------------|--------|
| 50-65 | Small, dense sticker-bomb (lots of tiny stickers) |
| 80-110 | Medium, balanced (clearly recognizable, good coverage) |
| 120-160 | Large, bold (fewer stickers, each very visible) |

### How We Discovered the UV Regions

1. **FruitBomb diff method**: compared `CH_2026.zip` base Details.dds with a skin
   that had stickers (FruitBomb) to find exactly which pixels changed. This gave
   the precise wheel face circle.

2. **GolfMaster color detection**: the GolfMaster F1 skin has green-colored tire
   rubber, so detecting green pixels revealed tire tread and sidewall positions.

3. **Visual inspection**: saved an auto-contrasted crop of the top-left 2000x2000
   region of Details.dds and visually identified each UV island.

Key lesson: never use auto-detection (like color filtering) without restricting
to known regions -- other car parts may share colors and pollute the mask.

### Rim Color vs Tire Color

- **Rim**: textured by both `Diffuse.dds` (via `sXXWheel` mesh objects) and the
  wheel face circle on `Details.dds`
- **Tire rubber**: textured ONLY via `Details.dds` (tire tread + sidewall regions)
- **Brake disc**: separate circle on `Details.dds`, usually left as default
