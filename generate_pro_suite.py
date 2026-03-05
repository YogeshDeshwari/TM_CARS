import argparse
import os
import zipfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import skin_styles
import skin_utils

# Default output directory
OUT_DIR = Path("out/pro_suite")

def create_pilot_texture():
    """Creates a dark grey pilot texture to ensure clean look."""
    img = Image.new("RGB", (512, 512), (10, 10, 10))
    # Add some seam lines so it looks like a suit
    d = ImageDraw.Draw(img)
    d.line([(256, 0), (256, 512)], fill=(30,30,30), width=4) # Zipper
    return img

def build_dds(img: Image.Image) -> bytes:
    """
    Simple DDS saver. 
    For a robust tool we might want mipmaps, but for now we use 
    PIL's save if available or a simple header if not.
    Actually, standard PIL doesn't save DDS well. 
    We will rely on the user having 'generate_tmnf_skin.py' logic 
    or just save as PNG if for preview, but for the game we need DDS.
    
    Workaround: We will import build_dds_dxt5_bytes from tmnf_dds
    (extracted from generate_tmnf_skin) to avoid coupling to the giant CLI file.
    """
    try:
        from tmnf_dds import build_dds_dxt1_bytes, build_dds_dxt5_bytes
        # Default to DXT5 (alpha + mipmaps) for Diffuse/Details/Icon/Pilot.
        # For maps where alpha is unused (e.g., Illum in TMNF), callers can use build_dds_dxt1_bytes directly.
        return build_dds_dxt5_bytes(img, mipmaps=True)
    except ImportError:
        print("Warning: generate_tmnf_skin not found, cannot build DDS.")
        return b""

def build_dds_dxt1(img: Image.Image) -> bytes:
    """Build a DXT1 DDS (no alpha). Useful for Illum.dds in TMNF/TMUF."""
    try:
        from tmnf_dds import build_dds_dxt1_bytes
        rgb = img.convert("RGB")
        return build_dds_dxt1_bytes(rgb, mipmaps=True)
    except ImportError:
        return b""

def _default_dirty_rgba(size: int = 1024) -> Image.Image:
    """
    Default 'no dirt' Dirty texture:
    - RGB: black (unused if alpha is black)
    - A: 0 (full black => no dirt overlay)
    """
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))

def _default_illum_rgb(size: int = 1024) -> Image.Image:
    """Default 'no illum' texture (RGB=0)."""
    return Image.new("RGB", (size, size), (0, 0, 0))

def package_skin(name: str, textures: dict, base_zip: Path, out_path: Path):
    """Packages the generated textures into a game-ready ZIP."""
    
    if not base_zip.exists():
        print(f"Error: Base zip {base_zip} not found.")
        return

    with zipfile.ZipFile(base_zip, 'r') as zin, zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        base_names = set(zin.namelist())
        # Copy everything from base EXCEPT the textures we are replacing
        for item in zin.infolist():
            if item.filename not in ["Diffuse.dds", "Details.dds", "Icon.dds", "StadiumCarPilot.dds"]:
                zout.writestr(item, zin.read(item.filename))
        
        # Write our new textures
        if "Diffuse" in textures:
            zout.writestr("Diffuse.dds", build_dds(textures["Diffuse"]))
        if "Details" in textures:
            zout.writestr("Details.dds", build_dds(textures["Details"]))
        if "Icon" in textures:
            zout.writestr("Icon.dds", build_dds(textures["Icon"]))
            
        # Write Pilot
        pilot = create_pilot_texture()
        zout.writestr("StadiumCarPilot.dds", build_dds(pilot))

        # Ensure Stadium aux textures exist for consistency with community upload rules:
        # - Dirty maps: Stadium-only dirt overlay; if missing, game falls back to default Stadium dirties.
        # - Illum: night illumination (not used on Stadium per some docs), but if missing game falls back to env default.
        if "DiffuseDirty.dds" not in base_names:
            zout.writestr("DiffuseDirty.dds", build_dds(_default_dirty_rgba(1024)))
        if "DetailsDirty.dds" not in base_names:
            zout.writestr("DetailsDirty.dds", build_dds(_default_dirty_rgba(1024)))
        if "Illum.dds" not in base_names:
            zout.writestr("Illum.dds", build_dds_dxt1(_default_illum_rgb(1024)))
        
    print(f"Generated: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a suite of 5 Pro Skins.")
    parser.add_argument("--team", required=True, help="Team Name")
    parser.add_argument("--base-zip", required=True, help="Path to a donor zip (for model files)")
    parser.add_argument("--colors", nargs=3, help="Hex codes: Base Accent Highlight (e.g. #000000 #FFD700 #00FFFF)")
    
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parse Colors
    if args.colors:
        cols = {
            "base": skin_utils.hex_to_rgb(args.colors[0]),
            "accent": skin_utils.hex_to_rgb(args.colors[1]),
            "highlight": skin_utils.hex_to_rgb(args.colors[2]),
            "secondary": (40, 40, 40) # Default dark grey
        }
    else:
        # Default Eror Black/Gold
        cols = {
            "base": (5, 5, 5),
            "accent": (218, 165, 32),
            "highlight": (255, 223, 100),
            "secondary": (30, 30, 30)
        }

    # Define the Suite
    styles = {
        "TechLuxe": skin_styles.TechLuxeStyle(),
        "KintsugiRoyal": skin_styles.KintsugiRoyalStyle(),
        "RacingSport": skin_styles.RacingSportStyle(),
        "StealthOps": skin_styles.StealthOpsStyle(),
        "Heritage": skin_styles.HeritageStyle(),
    }
    
    for style_name, generator in styles.items():
        print(f"Designing {style_name}...")
        
        # Generate Textures
        textures = generator.generate(args.team, cols)
        
        # Create Icon (Simple placeholder using Diffuse)
        # Resize Diffuse to 128x128 for icon
        icon = textures["Diffuse"].resize((128, 128), Image.Resampling.LANCZOS)
        textures["Icon"] = icon
        
        # Output Filename
        safe_team = args.team.replace(" ", "_")
        filename = f"{safe_team}_Pro_{style_name}.zip"
        out_path = OUT_DIR / filename
        
        # Package
        package_skin(style_name, textures, Path(args.base_zip), out_path)

        # Save Preview PNG
        textures["Diffuse"].save(OUT_DIR / f"{safe_team}_{style_name}_Preview.png")

if __name__ == "__main__":
    main()
