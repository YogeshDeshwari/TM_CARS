# TMNF Skin System -- Deep Dive

Analysis of `examples/not_car/` skins that replace the car with entirely different
objects (bob-omb, flintstone car, minecart, van, steve-in-a-cart), all playable in-game.

## The "Skin" System Is a Full Vehicle Modding System

Everything visual and audible about the car is replaceable. The only thing that
stays fixed is the invisible physics/collision hitbox.

---

## 1. 3D Mesh (Gbx Files)

Two mesh files control the car's 3D model:

| File | Purpose |
|------|---------|
| `MainBody.Solid.Gbx` | Low-detail mesh (shown at distance) |
| `MainBodyHigh.Solid.Gbx` | High-detail mesh (shown up close) |

Both can be completely custom geometry. Sizes across skins:

| Skin | MainBody | MainBodyHigh | Notes |
|------|----------|-------------|-------|
| Standard car (CH_2026) | 79,812 B | 1,126,504 B | 14:1 LOD ratio |
| No-mudguard (CHwarColor) | 91,750 B | 1,108,992 B | Mudguard polys removed |
| bob-omb | 123,452 B | 259,594 B | Spherical bomb shape |
| flintstone | 96,029 B | 671,109 B | Flintstones car |
| minecart | 1,869 B | 1,869 B | Box shape, identical LODs |
| pedo-van | 58,205 B | 2,585,205 B | Ford Transit, 2.3x standard |
| steve-in-a-cart | 25,510 B | 25,510 B | Minecraft character, identical LODs |

**Key findings:**
- You can use the same file for both LODs (minecart, steve do this).
- Mesh complexity ranges from 1.8 KB (box) to 2.6 MB (detailed van).
- Filename casing varies (`MainBody` vs `Mainbody` vs `mainbody`) -- game is case-insensitive.

---

## 2. Texture Files

### Required vs Optional

| Texture | Purpose | Required? |
|---------|---------|-----------|
| `Diffuse.dds` | Base color + material alpha | YES |
| `Details.dds` | Secondary UV (wheels, structural parts) | YES |
| `Icon.dds` | Menu thumbnail | YES |
| `DiffuseDirty.dds` | Dirty variant of Diffuse (after crashes) | No -- game handles missing |
| `DetailsDirty.dds` | Dirty variant of Details | No -- game handles missing |
| `ProjShad.dds` | Shadow projected onto ground | No -- some skins omit it |
| `Illum.dds` | Emissive/glow map | No -- not in standard skins |

### Dimensions Are Flexible

There is no fixed resolution. All of these work in-game:

| Skin | Diffuse | Details | ProjShad | Illum |
|------|---------|---------|----------|-------|
| Standard | 2048x2048 | 4096x4096 | 512x512 | -- |
| bob-omb | 2048x2048 | 2048x2048 | 1024x1024 | -- |
| flintstone | 1024x1024 | 1024x1024 | 512x512 | 1024x1024 |
| minecart | 1024x1024 | 1024x1024 | -- | 1024x1024 |
| pedo-van | 1024x1024 | **4096x2048** | 512x512 | **4096x2048** |
| steve | 1024x1024 | 1024x1024 | 256x256 | 1024x1024 |

Non-square textures work (4096x2048 on the van).

### DDS Formats

Multiple compression formats are supported:

| Format | Alpha | Used by |
|--------|-------|---------|
| DXT1 | 1-bit (on/off) | ProjShad, Illum, Icon |
| DXT3 | Explicit 4-bit | flintstone (all textures) |
| DXT5 | Interpolated 8-bit | Most skins (standard format) |

---

## 3. Illum.dds -- The Emissive Layer

Present in 4 of 5 custom skins but **absent from the standard car template**.

- Controls which parts of the model **glow/emit light**.
- Uses the same UV space as Diffuse.dds.
- Bright pixels = glow, black pixels = no glow.
- DXT1 works fine for this (no alpha needed, just RGB intensity).
- Can be any resolution (1024x1024 in most skins, 4096x2048 on the van).

**Implications for our skins:** We can add Illum.dds to make neon elements
actually glow in-game (circuit traces, accent lines, etc.). The engine already
handles it -- we just never included it.

---

## 4. Audio Files

Up to 7 audio slots, all optional:

| Slot | Purpose | Format |
|------|---------|--------|
| `EngineFast.ogg` | Engine at high RPM | .ogg or .wav |
| `EngineIdle.ogg` | Engine idle | .ogg or .wav |
| `EngineLow.ogg` | Engine at low RPM | .ogg or .wav |
| `EngineMid.ogg` | Engine at mid RPM | .ogg or .wav |
| `EngineRev.ogg` | Engine rev | .ogg or .wav |
| `EngineRev2.ogg` | Engine rev variant | .ogg or .wav |
| `horn.wav` | Horn sound | .wav |

The standard car skin includes none of these -- it uses default game sounds.
Custom skins can replace any or all of them.

---

## 5. Other Files

| File | Purpose |
|------|---------|
| `readme.txt` / `readme_carpark.txt` | Author credits, ignored by game |

---

## Implications for Our Pipeline

### No-Mudguard Builds
The mesh defines visibility. When mudguard geometry is removed, surfaces that were
hidden become visible. The texture is identical -- the difference is purely which
UV-mapped faces the renderer draws. Fixing the visual requires understanding which
UV regions the no-mudguard mesh exposes differently.

### Illum.dds for Glow Effects
We should generate an Illum.dds alongside Diffuse.dds. For skins like weaponized_115,
the green circuit traces could actually glow. Implementation:
1. Build an illum image from the design layers (circuit traces, accent slashes)
2. Encode as DXT1 (no alpha needed)
3. Add to the zip replacements

### Texture Size Freedom
We don't need to match the base pack's texture sizes exactly. Smaller textures
(1024x1024) work fine and produce smaller zips. Larger textures give more detail.

### Missing Textures Are OK
We can safely omit DiffuseDirty, DetailsDirty, or ProjShad if needed.
The game gracefully handles missing optional textures.
