# No-Mudguard Skin Build Process

## How It Works

Mudguards are **3D geometry** baked into the `.Gbx` mesh files, not part of the texture.
Removing them requires modified mesh files with the guard objects stripped out.

## Two Approaches (and Why One is Better)

### Approach 1: CHwarColor.zip Swap (OLD -- causes UV artifacts)

The original approach used `examples/no_mudguard/CHwarColor.zip`, a third-party skin
with mudguards already removed. Its Gbx files were swapped into our skins.

**Problem**: CHwarColor.zip is a *different model variant*. Its `sBody` geometry differs
from the standard CH_2026 model:

| File | Standard (CH_2026) | CHwarColor |
|------|-------------------|------------|
| MainBody.Solid.Gbx | 79,812 bytes | 91,750 bytes |
| MainBodyHigh.Solid.Gbx | 1,126,504 bytes | 1,108,992 bytes |

This causes two failures:
1. **Exposed `sBody` faces** behind where mudguards were have UVs pointing to arbitrary
   Diffuse.dds regions, creating visual artifacts (wrong colors/patterns leaking through)
2. **Wheel spoke UV islands** (`sXXWheel`) may land at different Diffuse positions,
   breaking any per-island texture painting

### Approach 2: `remove_guards` Tool (CORRECT -- preserves UV mapping)

Use `tools/remove_guards` to surgically strip guard objects from the **standard CH_2026
Gbx files**. This preserves all remaining objects (`sBody`, `sXXWheel`, `dXXWheel`,
`dBody`, etc.) with their **exact original UV mapping**.

Benefits:
- Wheel spoke UVs (`sXXWheel`) stay in the same Diffuse positions as the standard model
- Any UV island mask built from the standard model works without changes
- The only visible difference is the missing mudguards
- Exposed `sBody` faces show the body pattern (their UVs point to body panel areas)

## The `remove_guards` Tool

Located at `tools/remove_guards/`. Built with C# / GBX.NET.

### What it removes

```csharp
var guardNames = new HashSet<string> {
    "dFRGuard", "dFLGuard",   // Details-mapped front guard geometry
    "sFRGuard", "sFLGuard",   // Diffuse-mapped front guard geometry
    "sRLHub",   "sRRHub"      // Rear hub caps
};
```

### Running it

```bash
cd tools/remove_guards
dotnet run -- <input.Solid.Gbx> <output.Solid.Gbx>
```

### Which Gbx files have guards

| File | Guard objects | Notes |
|------|--------------|-------|
| MainBody.Solid.Gbx (low-poly LOD) | 0 | Only has dBody, sBody, ProjShad, lights, wheels |
| MainBodyHigh.Solid.Gbx (high-poly) | 6 | dFRGuard, dFLGuard, sFRGuard, sFLGuard, sRLHub, sRRHub |

Only `MainBodyHigh.Solid.Gbx` needs processing. The low-poly LOD has no guard objects,
but running the tool on it is harmless (it saves an identical copy).

## Step-by-Step: Building a No-Mudguard Skin

### 1. Extract standard Gbx files

```python
import zipfile

with zipfile.ZipFile("CH_all_skins/CH_2026.zip", "r") as z:
    z.extract("MainBody.Solid.Gbx", "/tmp/nomud_gbx/")
    z.extract("MainBodyHigh.Solid.Gbx", "/tmp/nomud_gbx/")
```

### 2. Run remove_guards

```bash
cd tools/remove_guards
dotnet run -- /tmp/nomud_gbx/MainBodyHigh.Solid.Gbx /tmp/nomud_gbx/MainBodyHigh_NoGuards.Solid.Gbx
dotnet run -- /tmp/nomud_gbx/MainBody.Solid.Gbx /tmp/nomud_gbx/MainBody_NoGuards.Solid.Gbx
```

Expected output for MainBodyHigh:
```
Original children: 35
  ...
  dFRGuard [REMOVE]
  dFLGuard [REMOVE]
  sFRGuard [REMOVE]
  sFLGuard [REMOVE]
  sRLHub [REMOVE]
  sRRHub [REMOVE]
  ...
Filtered children: 29
Removed: 6 guard objects
```

### 3. Build the no-mudguard skin zip

Take all texture files from the source skin, replace the Gbx files with the no-guard
versions:

```python
import zipfile, os

source_skin = "out/CH_MySkin.zip"
output_skin = "out/CH_MySkin_nomud.zip"

with zipfile.ZipFile(source_skin, "r") as z_src, \
     zipfile.ZipFile(output_skin, "w", zipfile.ZIP_DEFLATED) as z_out:

    for info in z_src.infolist():
        if info.filename == "MainBody.Solid.Gbx":
            z_out.write("/tmp/nomud_gbx/MainBody_NoGuards.Solid.Gbx", "MainBody.Solid.Gbx")
        elif info.filename == "MainBodyHigh.Solid.Gbx":
            z_out.write("/tmp/nomud_gbx/MainBodyHigh_NoGuards.Solid.Gbx", "MainBodyHigh.Solid.Gbx")
        else:
            z_out.writestr(info.filename, z_src.read(info.filename))
```

## TMNF Mesh Object Naming Convention

Understanding which objects map to which texture:

| Prefix | Texture file | Examples |
|--------|-------------|----------|
| `s*` | Diffuse.dds | sBody, sFLWheel, sFRGuard |
| `d*` | Details.dds | dBody, dFLWheel, dFRGuard |
| `g*` | (glass/special) | gBody |

When guard objects are removed, `sBody` faces that were occluded behind them become
visible. Since those faces were never intended to be seen, their UVs point to whatever
body panel area happens to be nearby in UV space. With the standard model's Gbx
(approach 2), these faces show the body pattern -- which looks acceptable. With a
different model variant (approach 1), the UV mismatch is unpredictable.
