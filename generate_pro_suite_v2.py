import argparse
import zipfile
from pathlib import Path
from PIL import Image, ImageOps, ImageChops, ImageDraw
import skin_utils # Still useful for hex_to_rgb
import shutil

# =============================================================================
# THE DIGITAL BODYSHOP COMPOSITOR (Integrated)
# =============================================================================

class DigitalBodyshop:
    def __init__(self, size=2048):
        self.size = size
        self.assets_dir = Path("assets")
        
    def load_asset(self, category, name):
        """Loads a PNG asset."""
        path = self.assets_dir / category / f"{name}.png"
        if not path.exists():
            # Fallback generation if asset missing (for robustness during dev)
            print(f"Warning: Asset {name} missing, generating fallback.")
            if category == "materials":
                return Image.new("RGB", (self.size, self.size), (20,20,20))
            else:
                return Image.new("RGBA", (self.size, self.size), (0,0,0,0))
                
        return Image.open(path).resize((self.size, self.size), Image.Resampling.LANCZOS)

    def compose(self, base_material_name, pattern_name, colors, zone_mask_type="sidepods"):
        """
        The core "Photoshop-like" workflow.
        """
        # 1. Base Layer
        base = self.load_asset("materials", base_material_name)
        if base_material_name == "matte_black":
             # Tint it to the specific team base color
             base = ImageOps.colorize(base.convert("L"), "black", colors["base"])
        elif base_material_name == "carbon_fiber":
             # Carbon is usually grey, tint it slightly with base
             carbon_tint = ImageOps.colorize(base.convert("L"), "black", (60,60,60))
             base = Image.blend(base.convert("RGB"), carbon_tint, 0.5)

        # 2. Pattern Layer
        if pattern_name:
            pattern_mask = self.load_asset("patterns", pattern_name).convert("L")
            pattern_fill = Image.new("RGBA", base.size, colors["accent"])
            
            # Zone Masking
            zone_mask = Image.new("L", base.size, 0)
            d = ImageDraw.Draw(zone_mask)
            
            if zone_mask_type == "sidepods":
                # Bottom half + Fenders
                d.rectangle([0, self.size//2, self.size, self.size], fill=255)
            elif zone_mask_type == "all":
                d.rectangle([0, 0, self.size, self.size], fill=255)
            elif zone_mask_type == "stripes":
                # Center stripe zone
                center = self.size // 2
                w = self.size // 6
                d.rectangle([center-w, 0, center+w, self.size], fill=255)
            
            final_mask = ImageChops.multiply(pattern_mask, zone_mask)
            
            diffuse = base.convert("RGBA")
            diffuse.paste(pattern_fill, (0,0), final_mask)
        else:
            diffuse = base.convert("RGBA")
            final_mask = Image.new("L", base.size, 0) # No pattern gloss

        # 3. Details Map (Material Definition)
        # Base Material Roughness
        if base_material_name == "matte_black":
            base_gloss = 20 # Matte
        elif base_material_name == "carbon_fiber":
            base_gloss = 140 # Semi-Gloss Carbon
        else:
            base_gloss = 100
            
        details = Image.new("RGBA", base.size, (0,0,0, base_gloss))
        
        # Pattern Gloss (Metallic)
        shiny_fill = Image.new("RGBA", base.size, (0,0,0, 220)) # Metallic
        details.paste(shiny_fill, (0,0), final_mask)
        
        return {"Diffuse": diffuse, "Details": details}

# =============================================================================
# SUITE GENERATOR
# =============================================================================

OUT_DIR = Path("out/pro_suite_v2")

def build_dds(img: Image.Image) -> bytes:
    try:
        from tmnf_dds import build_dds_dxt5_bytes
        return build_dds_dxt5_bytes(img, mipmaps=True)
    except ImportError:
        return b""

def build_dds_dxt1(img: Image.Image) -> bytes:
    """Build a DXT1 DDS (no alpha). Useful for Illum.dds in TMNF/TMUF."""
    try:
        from tmnf_dds import build_dds_dxt1_bytes
        return build_dds_dxt1_bytes(img.convert("RGB"), mipmaps=True)
    except ImportError:
        return b""

def _default_dirty_rgba(size: int = 1024) -> Image.Image:
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))

def _default_illum_rgb(size: int = 1024) -> Image.Image:
    return Image.new("RGB", (size, size), (0, 0, 0))

def create_pilot_texture():
    return Image.new("RGB", (512, 512), (10, 10, 10)) # Black Suit

def package_skin(name: str, textures: dict, base_zip: Path, out_path: Path):
    if not base_zip.exists(): return

    with zipfile.ZipFile(base_zip, 'r') as zin, zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        base_names = set(zin.namelist())
        for item in zin.infolist():
            if item.filename not in ["Diffuse.dds", "Details.dds", "Icon.dds", "StadiumCarPilot.dds"]:
                zout.writestr(item, zin.read(item.filename))
        
        if "Diffuse" in textures: zout.writestr("Diffuse.dds", build_dds(textures["Diffuse"]))
        if "Details" in textures: zout.writestr("Details.dds", build_dds(textures["Details"]))
        if "Icon" in textures: zout.writestr("Icon.dds", build_dds(textures["Icon"]))
        zout.writestr("StadiumCarPilot.dds", build_dds(create_pilot_texture()))

        # Ensure Stadium aux textures exist (Dirty + Illum) even if donor zip doesn't include them.
        if "DiffuseDirty.dds" not in base_names:
            zout.writestr("DiffuseDirty.dds", build_dds(_default_dirty_rgba(1024)))
        if "DetailsDirty.dds" not in base_names:
            zout.writestr("DetailsDirty.dds", build_dds(_default_dirty_rgba(1024)))
        if "Illum.dds" not in base_names:
            zout.writestr("Illum.dds", build_dds_dxt1(_default_illum_rgb(1024)))
    print(f"Generated: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate V2 Asset-Based Pro Suite.")
    parser.add_argument("--team", required=True)
    parser.add_argument("--base-zip", required=True)
    parser.add_argument("--colors", nargs=3, help="Hex codes: Base Accent Highlight")
    
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parse Colors
    if args.colors:
        cols = {
            "base": skin_utils.hex_to_rgb(args.colors[0]),
            "accent": skin_utils.hex_to_rgb(args.colors[1]),
            "highlight": skin_utils.hex_to_rgb(args.colors[2])
        }
    else:
        cols = {"base": (5,5,5), "accent": (218,165,32), "highlight": (255,223,100)}

    shop = DigitalBodyshop()
    
    # Define V2 Archetypes (Asset-Based)
    
    # 1. TechLuxe V2
    print("Designing TechLuxe V2...")
    tex = shop.compose("matte_black", "circuit_mask", cols, zone_mask_type="sidepods")
    tex["Icon"] = tex["Diffuse"].resize((128,128))
    tex["Diffuse"].save(OUT_DIR / f"{args.team}_TechLuxe_V2_Preview.png")
    package_skin("TechLuxe_V2", tex, Path(args.base_zip), OUT_DIR / f"{args.team}_Pro_TechLuxe_V2.zip")
    
    # 2. Kintsugi V2
    print("Designing Kintsugi V2...")
    tex = shop.compose("carbon_fiber", "kintsugi_mask", cols, zone_mask_type="all") # Kintsugi everywhere on carbon
    tex["Icon"] = tex["Diffuse"].resize((128,128))
    tex["Diffuse"].save(OUT_DIR / f"{args.team}_Kintsugi_V2_Preview.png")
    package_skin("Kintsugi_V2", tex, Path(args.base_zip), OUT_DIR / f"{args.team}_Pro_Kintsugi_V2.zip")
    
    # 3. Stealth V2 (No Pattern, just high quality matte)
    print("Designing Stealth V2...")
    tex = shop.compose("matte_black", None, cols)
    tex["Icon"] = tex["Diffuse"].resize((128,128))
    tex["Diffuse"].save(OUT_DIR / f"{args.team}_Stealth_V2_Preview.png")
    package_skin("Stealth_V2", tex, Path(args.base_zip), OUT_DIR / f"{args.team}_Pro_Stealth_V2.zip")

if __name__ == "__main__":
    main()
