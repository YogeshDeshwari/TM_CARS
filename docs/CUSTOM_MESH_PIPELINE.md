# Custom Mesh Pipeline -- TMNF Car Geometry

How to create, modify, and export custom car meshes for TrackMania Nations Forever.

---

## How TMNF Car Meshes Work

The car is built from **named mesh objects** inside a `.Solid.Gbx` file. The game
identifies each part by its name and assigns physics, rendering, and UV behavior
accordingly.

### Object Naming Convention

**Body:**

| Object | Description | Texture Layer |
|--------|-------------|---------------|
| `sBody` | Paintable body exterior | Diffuse.dds |
| `dBody` | Non-paintable details (grille, lights, rubber trim) | Details.dds |
| `gBody` | Transparent glass (windows, light covers) | Details.dds |

**Wheels (XX = FL, FR, RL, RR):**

| Object | Description | Texture Layer |
|--------|-------------|---------------|
| `sXXWheel` | Paintable rim | Diffuse.dds |
| `dXXWheel` | Non-paintable tire | Details.dds |

**Mudguards (XX = FL, FR only):**

| Object | Description | Texture Layer |
|--------|-------------|---------------|
| `sXXGuard` | Paintable fender/mudguard | Diffuse.dds |
| `dXXGuard` | Non-paintable fender detail | Details.dds |

**Suspension (optional, XX = FL, FR, RL, RR):**

| Object | Description |
|--------|-------------|
| `dXXHub` | Wheel hub/kingpin (required for other suspension parts to work) |
| `dXXArmTop` | Upper control arm |
| `dXXArmBot` | Lower control arm |
| `dXXArmDir` | Steering rod |
| `dXXSusp` | Spring/shock absorber |
| `dXXCardan` | Rear driveshaft (RL, RR only) |

**Other:**

| Object | Description |
|--------|-------------|
| `pPilHead` | Driver head (wobbles in-game) |
| `ProjShad` | Shadow projection cone (uses ProjShad.dds) |
| `LightFProj` | Headlight projection cone (uses CarLights.dds) |
| `LightFL1/2/3` | Front-left light helpers (flare positions) |
| `LightFR1/2/3` | Front-right light helpers (flare positions) |
| `LightRL` | Rear-left light helper |
| `LightRR` | Rear-right light helper |

### UV Mapping Rules

- `s`-prefixed objects (sBody, sXXWheel, sXXGuard) use **Diffuse.dds** UV space
- `d`-prefixed objects (dBody, dXXWheel, dXXGuard) use **Details.dds** UV space
- `gBody` uses **Details.dds** UV space (map to empty area)
- `sBody` UVs should NOT overlap
- `dBody` UVs CAN overlap
- `dXXWheel` objects can all share the same UV region

### Scale

Models must be at **0.1% of real-world size**:
- 2800mm wheelbase = 2.8mm in model
- 660mm wheel diameter = 0.660mm in model

### Vertex Limits

- `MainBodyHigh.Solid.Gbx` (high detail): ~100,000 vertices max
- `MainBody.Solid.Gbx` (low detail): ~3,600 vertices max

---

## Pipeline: Modifying the Standard Car

### Tools Required

1. **Blender** (3.x+)
2. **Blender Gbx Tools** -- import Solid.Gbx into Blender
   - Install: Blender Preferences > Extensions > Add remote repository:
     `https://blender.gbx.tools/index.json`
   - Then search "Blender Gbx Tools" and install
   - GitHub: https://github.com/search?q=blender+gbx+tools
3. **3ds2gbxml** -- convert .3ds back to Solid.Gbx
   - GitHub: https://github.com/GreffMASTER/3ds2gbxml
   - Python-based, converts .3ds to GBXML visual and surface model files

### Workflow

```
Standard Solid.Gbx
       |
       v
[Blender Gbx Tools]  -- import into Blender
       |
       v
  Edit in Blender     -- delete/modify objects, adjust UVs
       |
       v
  Export as .3ds       -- Blender File > Export > 3DS
       |
       v
   [3ds2gbxml]         -- convert to GBXML/Solid.Gbx
       |
       v
 Custom Solid.Gbx      -- use in skin zip
```

### Fallback: TMNF Built-in Importer

If 3ds2gbxml doesn't work for a specific case, TMNF has a built-in converter:
1. Open TMNF launcher
2. Help > Custom Data > Data Importer > Car Geometry
3. Browse to your .3ds file
4. It creates a .Solid.Gbx in the same folder

No error reporting -- you only find out if something is wrong when you load it in-game.

---

## No-Mudguard Build

### What to Remove

Delete these objects from the mesh:
- `sFLGuard` (front-left mudguard, paintable)
- `dFLGuard` (front-left mudguard, detail)
- `sFRGuard` (front-right mudguard, paintable)
- `dFRGuard` (front-right mudguard, detail)

### Why the Current No-Mudguard Mesh Looks Wrong

The `sBody` mesh has faces that were designed to sit behind the mudguards, never
meant to be seen. When the guard objects are removed, these faces become visible
with UV mapping that was intended to be occluded. This creates:

- Bright color bleed from adjacent UV regions
- Flat/stretched textures on newly exposed inner surfaces
- Unexpected pattern placement on wheel wells

### Proper Fix (Blender)

1. Import the standard mesh
2. Delete guard objects
3. Select the `sBody` faces that were behind the guards
4. Re-UV-map them to a controlled region (dark/neutral area, or match adjacent panels)
5. Optionally add geometry to close gaps where guards connected to the body
6. Export both LOD versions

---

## File Reference

### Texture Files in a Skin Zip

| File | Required | Format | Purpose |
|------|----------|--------|---------|
| `Diffuse.dds` | Yes | DXT5 | Body color + material alpha (s-objects) |
| `Details.dds` | Yes | DXT5 | Wheels, interior, structural (d-objects) |
| `Icon.dds` | Yes | DXT5 | Menu thumbnail |
| `DiffuseDirty.dds` | No | DXT5 | Dirty body variant |
| `DetailsDirty.dds` | No | DXT5 | Dirty details variant |
| `ProjShad.dds` | No | DXT1 | Ground shadow |
| `Illum.dds` | No | DXT1 | Emissive/glow map |
| `MainBody.Solid.Gbx` | Yes | -- | Low-detail mesh |
| `MainBodyHigh.Solid.Gbx` | Yes | -- | High-detail mesh |
| `EngineFast.ogg` | No | OGG/WAV | Engine high RPM |
| `EngineIdle.ogg` | No | OGG/WAV | Engine idle |
| `EngineLow.ogg` | No | OGG/WAV | Engine low RPM |
| `EngineMid.ogg` | No | OGG/WAV | Engine mid RPM |
| `EngineRev.ogg` | No | OGG/WAV | Engine rev |
| `EngineRev2.ogg` | No | OGG/WAV | Engine rev variant |
| `Horn.wav` | No | WAV | Horn sound |

### Texture Size

No fixed requirement. Any power-of-two dimension works. Non-square is fine
(e.g., 4096x2048). Common sizes: 1024x1024, 2048x2048, 4096x4096.

### DDS Formats

DXT1 (1-bit alpha), DXT3 (explicit alpha), and DXT5 (interpolated alpha) all work.
DXT5 is standard for Diffuse/Details. DXT1 is fine for ProjShad/Illum.
