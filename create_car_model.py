#!/usr/bin/env python3
"""
TMNF/TMUF Car Model Creation & Editing Toolkit

This script provides tools for working with TrackMania Nations Forever car models.
It can extract meshes from GBX files and create new car skins from scratch.

WORKFLOW:
=========
1. GBX -> OBJ (extract existing model to editable format)
2. Edit in Blender (modify geometry, UV mapping, textures)
3. OBJ -> 3DS in Blender (export)
4. 3DS -> GBX via TMNF's built-in importer (in-game: Help -> Custom data -> Car geometry)
5. Create textures (Diffuse.dds, Details.dds, etc.) with proper UV mapping
6. Package as ZIP for use in game

REQUIRED TOOLS:
===============
- Python 3.8+
- .NET 9 SDK (for GBX.NET)
- Blender (for 3D editing)
- GIMP/Photoshop with DDS plugin (for textures)
- TMNF/TMUF game (for 3DS -> GBX conversion)

CAR MODEL STRUCTURE (from ugghost.com tutorial):
================================================
Required mesh objects (names matter!):
    sBody    - paintable body (uses Diffuse.dds)
    dBody    - non-paintable details like interior, lights (uses Details.dds)
    gBody    - transparent glass
    
Wheel objects (required):
    dFLWheel, sFLWheel - Front Left (d=tire/detail, s=rim/skin)
    dFRWheel, sFRWheel - Front Right
    dRLWheel, sRLWheel - Rear Left  
    dRRWheel, sRRWheel - Rear Right

Optional wheel parts:
    dxxHub      - kingpin/hub (xx = FL,FR,RL,RR)
    dxxArmTop   - upper control arm
    dxxArmBot   - lower control arm
    dxxArmDir   - steering rod
    dxxSusp     - spring/shock
    sxxGuard/dxxGuard - mudguards (only FL/FR)
    dxxCardan   - rear driveshafts (only RL/RR)

Other:
    pPilHead   - driver's head (bobbles)
    ProjShad   - cone mesh for shadow projection
    LightFProj - cone for headlight projection

TEXTURE FILES:
==============
Required:
    Diffuse.dds  - main skin texture (sBody, sxxWheel use this)
    Details.dds  - detail texture (dBody, dxxWheel, gBody use this)
    Icon.dds     - car selection icon

Optional:
    DetailsDirty.dds  - dirty version of Details
    DiffuseDirty.dds  - dirty version of Diffuse
    Illum.dds         - illumination map
    ProjShad.dds      - shadow projection texture
    Horn.wav/ogg      - horn sound
    Engine*.ogg       - engine sounds

SCALE:
======
- Model should be 0.1% of real size
- 2800mm wheelbase = 2.8mm in model
- 660mm wheel diameter = 0.66mm in model
- Max bounds: 6mm x 3mm x 2.5mm (L x W x H)
"""

import os
import sys
import zipfile
import struct
import io
import subprocess
import shutil
from pathlib import Path

# Check for required dependencies
try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Required: pip install Pillow numpy")
    sys.exit(1)


def extract_gbx_to_obj(gbx_path: str, output_dir: str = None) -> str:
    """
    Extract a .Solid.Gbx file to OBJ format using GBX.NET.
    
    Args:
        gbx_path: Path to the GBX file
        output_dir: Optional output directory (default: same as input)
    
    Returns:
        Path to the generated OBJ file
    """
    gbx_path = Path(gbx_path).resolve()
    if not gbx_path.exists():
        raise FileNotFoundError(f"GBX file not found: {gbx_path}")
    
    # Check for GBX.NET tool
    gbx_net_path = Path("/tmp/gbx-net/Samples/Beginner/SolidExtract/SolidExtract.csproj")
    if not gbx_net_path.exists():
        raise RuntimeError("GBX.NET not found. Please clone https://github.com/BigBang1112/gbx-net.git to /tmp/gbx-net")
    
    # Copy liblzo2 if needed
    lzo_src = Path("/opt/homebrew/lib/liblzo2.dylib")
    lzo_dst = gbx_net_path.parent / "bin/Debug/net9.0/liblzo2.dylib"
    if lzo_src.exists() and not lzo_dst.exists():
        lzo_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(lzo_src, lzo_dst)
    
    # Run extraction
    env = os.environ.copy()
    env["DOTNET_ROOT"] = "/opt/homebrew/opt/dotnet/libexec"
    
    result = subprocess.run(
        ["dotnet", "run", "--project", str(gbx_net_path), "--no-build", "--", str(gbx_path)],
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"GBX extraction failed: {result.stderr}")
    
    obj_path = gbx_path.with_suffix(".Gbx.obj")
    if not obj_path.exists():
        raise RuntimeError(f"OBJ file not created: {obj_path}")
    
    # Move to output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        new_obj_path = output_dir / obj_path.name
        shutil.move(obj_path, new_obj_path)
        mtl_path = gbx_path.with_suffix(".Gbx.mtl")
        if mtl_path.exists():
            shutil.move(mtl_path, output_dir / mtl_path.name)
        obj_path = new_obj_path
    
    return str(obj_path)


def extract_car_from_zip(zip_path: str, output_dir: str) -> dict:
    """
    Extract all car files from a zip and convert GBX to OBJ.
    
    Args:
        zip_path: Path to the car zip file
        output_dir: Output directory for extracted files
    
    Returns:
        Dictionary of extracted file paths
    """
    zip_path = Path(zip_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = {}
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            # Extract file
            data = z.read(name)
            out_path = output_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            extracted[name.lower()] = str(out_path)
            
            # Convert GBX to OBJ
            if name.lower().endswith('.solid.gbx'):
                try:
                    out_path.chmod(0o644)  # Ensure readable
                    obj_path = extract_gbx_to_obj(str(out_path), str(output_dir))
                    extracted[name.lower().replace('.solid.gbx', '.obj')] = obj_path
                except Exception as e:
                    print(f"Warning: Could not extract {name} to OBJ: {e}")
    
    return extracted


def build_dds_rgba8(img: Image.Image) -> bytes:
    """Build a DDS file in RGBA8 format from a PIL Image."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    w, h = img.size
    header = bytearray(128)
    header[0:4] = b'DDS '
    header[4:8] = struct.pack('<I', 124)  # header size
    header[8:12] = struct.pack('<I', 0x1 | 0x2 | 0x4 | 0x1000 | 0x8)  # flags
    header[12:16] = struct.pack('<I', h)
    header[16:20] = struct.pack('<I', w)
    header[20:24] = struct.pack('<I', w * 4)  # pitch
    header[76:80] = struct.pack('<I', 32)  # pixel format size
    header[80:84] = struct.pack('<I', 0x41)  # RGBA
    header[88:92] = struct.pack('<I', 32)  # bits per pixel
    header[92:96] = struct.pack('<I', 0x00FF0000)  # R mask
    header[96:100] = struct.pack('<I', 0x0000FF00)  # G mask
    header[100:104] = struct.pack('<I', 0x000000FF)  # B mask
    header[104:108] = struct.pack('<I', 0xFF000000)  # A mask
    header[108:112] = struct.pack('<I', 0x1000)  # caps
    
    # Convert to BGRA (DDS format)
    arr = np.array(img)[::-1]  # Flip vertically
    bgra = arr[:, :, [2, 1, 0, 3]].tobytes()
    
    return bytes(header) + bgra


def create_blank_texture(width: int, height: int, color=(128, 128, 128, 255)) -> Image.Image:
    """Create a blank texture with specified color."""
    img = Image.new('RGBA', (width, height), color)
    return img


def create_icon_from_diffuse(diffuse_path: str, size=(128, 128)) -> Image.Image:
    """Create an icon from the diffuse texture."""
    try:
        img = Image.open(diffuse_path).convert('RGBA')
    except:
        img = create_blank_texture(size[0], size[1])
    
    # Resize to icon size
    img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def create_car_zip(
    output_path: str,
    gbx_low: str = None,
    gbx_high: str = None,
    diffuse: Image.Image = None,
    details: Image.Image = None,
    icon: Image.Image = None,
    extra_files: dict = None
):
    """
    Create a car zip file with all required components.
    
    Args:
        output_path: Path for the output zip
        gbx_low: Path to MainBody.Solid.Gbx
        gbx_high: Path to MainBodyHigh.Solid.Gbx
        diffuse: Diffuse texture as PIL Image
        details: Details texture as PIL Image
        icon: Icon texture as PIL Image
        extra_files: Dict of {filename: bytes} for additional files
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default textures if not provided
    if diffuse is None:
        diffuse = create_blank_texture(2048, 2048)
    if details is None:
        details = create_blank_texture(4096, 4096)
    if icon is None:
        icon = create_blank_texture(128, 128)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Add model files
        if gbx_low:
            z.write(gbx_low, 'MainBody.Solid.Gbx')
        if gbx_high:
            z.write(gbx_high, 'MainBodyHigh.Solid.Gbx')
        elif gbx_low:
            # Use low poly as high poly if no high provided
            z.write(gbx_low, 'MainBodyHigh.Solid.Gbx')
        
        # Add textures
        z.writestr('Diffuse.dds', build_dds_rgba8(diffuse))
        z.writestr('Details.dds', build_dds_rgba8(details))
        z.writestr('Icon.dds', build_dds_rgba8(icon))
        
        # Add extra files
        if extra_files:
            for name, data in extra_files.items():
                if isinstance(data, bytes):
                    z.writestr(name, data)
                elif isinstance(data, Image.Image):
                    z.writestr(name, build_dds_rgba8(data))
                elif isinstance(data, str) and os.path.exists(data):
                    z.write(data, name)
    
    print(f"Created car zip: {output_path}")
    return str(output_path)


def analyze_obj_groups(obj_path: str) -> dict:
    """
    Analyze an OBJ file and list its groups/objects.
    
    Returns dict with group names and their vertex/face counts.
    """
    groups = {}
    current_group = "default"
    vertex_count = 0
    face_count = 0
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('g '):
                if current_group in groups:
                    groups[current_group]['faces'] = face_count
                current_group = line[2:].strip()
                groups[current_group] = {'vertices': 0, 'faces': 0}
                face_count = 0
            elif line.startswith('v '):
                vertex_count += 1
            elif line.startswith('f '):
                face_count += 1
    
    if current_group in groups:
        groups[current_group]['faces'] = face_count
    
    return {
        'total_vertices': vertex_count,
        'groups': groups
    }


def print_workflow_guide():
    """Print the complete workflow guide."""
    print("""
================================================================================
         TMNF/TMUF Car Model Creation - Complete Workflow Guide
================================================================================

OVERVIEW:
---------
Creating a car skin from scratch involves these steps:
1. Create or modify a 3D model (OBJ format)
2. Set up proper UV mapping for textures
3. Convert to GBX format using TMNF's built-in importer
4. Create textures that match the UV layout
5. Package as a ZIP file

STEP-BY-STEP GUIDE:
-------------------

STEP 1: Get a Base Model
------------------------
Option A: Extract existing model:
    python create_car_model.py extract examples/some_car.zip out/extracted/

Option B: Create new model in Blender:
    - Start with a car template
    - Model must have specific object names (sBody, dBody, etc.)
    - Scale: 0.1% of real size (2800mm wheelbase = 2.8mm in model)

STEP 2: Edit in Blender
-----------------------
    - Import the OBJ file
    - Modify geometry as needed
    - Create UV unwrap for each material group:
        * sBody -> maps to Diffuse.dds
        * dBody -> maps to Details.dds
        * Wheels use both textures
    - Export as .3DS file (File -> Export -> 3D Studio)

STEP 3: Convert 3DS to GBX
--------------------------
    In TrackMania Forever/United:
    1. Launch the game
    2. Go to: Help -> Custom data -> Data importer -> Car geometry
    3. Browse to your .3ds file
    4. Click Open
    5. A .Solid.Gbx file will be created in the same folder

STEP 4: Create Textures
-----------------------
    Create these DDS files matching your UV layout:
    
    - Diffuse.dds (2048x2048 recommended)
      * Contains the main car body skin
      * Alpha channel controls specularity/material
      
    - Details.dds (4096x4096 recommended)
      * Interior, tires, lights, glass areas
      * Alpha channel for transparency on glass
      
    - Icon.dds (128x128 or 64x64)
      * Car selection preview icon

STEP 5: Package as ZIP
----------------------
    python create_car_model.py create output.zip \\
        --gbx-low model/MainBody.Solid.Gbx \\
        --gbx-high model/MainBodyHigh.Solid.Gbx \\
        --diffuse textures/Diffuse.dds \\
        --details textures/Details.dds

STEP 6: Install and Test
------------------------
    Copy the ZIP to:
    Documents/TrackMania/Skins/Vehicles/StadiumCar/
    
    Launch the game and select your car!

TIPS:
-----
- Always back up original files
- Test frequently - small changes, quick iterations
- Use Debug UVs to visualize mapping
- The alpha channel in Diffuse.dds controls material properties
- Standard stadium car uses alpha=113 for the body

================================================================================
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TMNF/TMUF Car Model Creation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show full workflow guide
  python create_car_model.py guide
  
  # Extract a car zip to editable formats
  python create_car_model.py extract examples/car.zip out/extracted/
  
  # Analyze an OBJ file
  python create_car_model.py analyze models/StadiumCar.obj
  
  # Create a new car zip
  python create_car_model.py create out/MyCar.zip --gbx-low model/MainBody.Solid.Gbx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Guide command
    guide_parser = subparsers.add_parser('guide', help='Show complete workflow guide')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract car zip to editable formats')
    extract_parser.add_argument('zip_path', help='Path to car zip file')
    extract_parser.add_argument('output_dir', help='Output directory')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze OBJ file structure')
    analyze_parser.add_argument('obj_path', help='Path to OBJ file')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new car zip')
    create_parser.add_argument('output', help='Output zip path')
    create_parser.add_argument('--gbx-low', help='MainBody.Solid.Gbx path')
    create_parser.add_argument('--gbx-high', help='MainBodyHigh.Solid.Gbx path')
    create_parser.add_argument('--diffuse', help='Diffuse.dds or PNG path')
    create_parser.add_argument('--details', help='Details.dds or PNG path')
    create_parser.add_argument('--icon', help='Icon.dds or PNG path')
    
    args = parser.parse_args()
    
    if args.command == 'guide':
        print_workflow_guide()
    
    elif args.command == 'extract':
        print(f"Extracting {args.zip_path} to {args.output_dir}...")
        files = extract_car_from_zip(args.zip_path, args.output_dir)
        print("\nExtracted files:")
        for name, path in sorted(files.items()):
            print(f"  {name}: {path}")
    
    elif args.command == 'analyze':
        print(f"Analyzing {args.obj_path}...")
        info = analyze_obj_groups(args.obj_path)
        print(f"\nTotal vertices: {info['total_vertices']}")
        print("\nGroups:")
        for name, data in info['groups'].items():
            print(f"  {name}: {data['faces']} faces")
    
    elif args.command == 'create':
        # Load textures if provided
        diffuse = None
        details = None
        icon = None
        
        if args.diffuse:
            diffuse = Image.open(args.diffuse).convert('RGBA')
        if args.details:
            details = Image.open(args.details).convert('RGBA')
        if args.icon:
            icon = Image.open(args.icon).convert('RGBA')
        
        create_car_zip(
            args.output,
            gbx_low=args.gbx_low,
            gbx_high=args.gbx_high,
            diffuse=diffuse,
            details=details,
            icon=icon
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()



