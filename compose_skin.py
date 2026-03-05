import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageChops

# =============================================================================
# THE DIGITAL BODYSHOP COMPOSITOR
# =============================================================================

class DigitalBodyshop:
    def __init__(self, size=2048):
        self.size = size
        self.assets_dir = Path("assets")
        
    def load_asset(self, category, name):
        """Loads a PNG asset."""
        path = self.assets_dir / category / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"Asset missing: {path}")
        return Image.open(path).resize((self.size, self.size), Image.Resampling.LANCZOS)

    def compose(self, base_material_name, pattern_name, colors):
        """
        The core "Photoshop-like" workflow:
        1. Base Layer (Material)
        2. Pattern Layer (Colorized Mask)
        3. Finish Map (Details.dds generation)
        """
        
        # 1. Base Layer
        base = self.load_asset("materials", base_material_name)
        # If the base material is greyscale (like carbon), tint it with base color?
        # For "TechLuxe", we want pure matte black carbon, so maybe just tint slightly.
        if base_material_name == "matte_black":
             # Tint it to the specific team base color (e.g. #050505)
             base = ImageOps.colorize(base.convert("L"), "black", colors["base"])
        
        # 2. Pattern Layer
        # Load the mask (White = Pattern, Transparent = Empty)
        pattern_mask = self.load_asset("patterns", pattern_name).convert("L")
        
        # Create a solid color fill for the pattern (Gold)
        pattern_fill = Image.new("RGBA", base.size, colors["accent"])
        
        # Composite: Paste the Gold Fill using the Pattern Mask
        # But we want to RESTRICT this to the Sidepods (for TechLuxe).
        # We need a "Zone Mask". Since we don't have a 3D model painter, 
        # we'll generate a 2D mask that approximates the sidepods.
        
        zone_mask = Image.new("L", base.size, 0)
        # Draw the "Sidepods" zone (Bottom half + Fenders) on the mask
        from PIL import ImageDraw
        d = ImageDraw.Draw(zone_mask)
        d.rectangle([0, self.size//2, self.size, self.size], fill=255) # Bottom half
        
        # Combine Pattern Mask AND Zone Mask
        # The pattern only appears where BOTH the pattern exists AND it's in the zone.
        final_mask = ImageChops.multiply(pattern_mask, zone_mask)
        
        # Composite
        diffuse = base.convert("RGBA")
        diffuse.paste(pattern_fill, (0,0), final_mask)
        
        # 3. Details Map (Material Definition)
        # Base is Matte (Low Alpha)
        details = Image.new("RGBA", base.size, (0,0,0, 30)) # 30 = Matte
        
        # Pattern is Metallic/Shiny (High Alpha)
        # We use the same final_mask to "paint" shininess
        shiny_fill = Image.new("RGBA", base.size, (0,0,0, 220)) # 220 = Metallic
        details.paste(shiny_fill, (0,0), final_mask)
        
        return {"Diffuse": diffuse, "Details": details}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--colors", nargs=3, help="Hex: Base Accent Highlight")
    args = parser.parse_args()
    
    # Setup Colors
    def hex2rgb(h): return tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0,2,4))
    cols = {
        "base": hex2rgb(args.colors[0]),
        "accent": hex2rgb(args.colors[1]),
        "highlight": hex2rgb(args.colors[2])
    }
    
    shop = DigitalBodyshop()
    
    # Generate Styles using the Bodyshop
    # Style 1: The New TechLuxe (Asset-Based)
    print("Composing TechLuxe V2 (Asset-Based)...")
    tex = shop.compose("matte_black", "circuit_mask", cols)
    
    # Save Preview
    out_dir = Path("out/bodyshop_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex["Diffuse"].save(out_dir / f"{args.team}_TechLuxe_V2.png")
    tex["Details"].save(out_dir / f"{args.team}_TechLuxe_V2_Details.png")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
