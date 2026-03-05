import zipfile
import os
from pathlib import Path
from pro_skin_engine import ProSkinEngine, skin_utils
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter
import logo_gen

# Ensure logos exist
if not Path("assets/generated_logos").exists():
    logo_gen.LogoGenerator().save_demo_logos()

def compress_skin(skin_name):
    # Zip the output folder
    out_dir = Path(f"out/{skin_name}")
    zip_path = Path(f"out/{skin_name}.zip")
    
    if not out_dir.exists():
        print(f"Error: {out_dir} does not exist")
        return

    print(f"Compressing {skin_name} to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.glob("*"):
            zf.write(f, f.name)
    print("Done.")

def create_pro_racing_skin(team_name, colors):
    """
    Pro Racing: High contrast, sponsors, speed stripes.
    """
    engine = ProSkinEngine(team_name=team_name)
    
    # 1. Base: Semi-Gloss
    engine.set_base_material(colors["base"], "satin")
    
    # 2. Pattern: Racing Stripes
    def stripes_pat(size, color):
        img = Image.new("RGBA", (size, size), (0,0,0,0))
        d = ImageDraw.Draw(img)
        # Center stripe
        center = size // 2
        w = size // 8
        d.rectangle([center-w, 0, center+w, size], fill=color+(255,))
        return img
        
    engine.add_pattern(stripes_pat, colors["accent"], opacity=1.0)
    
    # 3. Sponsors
    # Hood
    engine.add_sticker_pro("ARC LABS", (0.5, 0.3), scale=1.2, rotation=180, color_override=colors["highlight"])
    # Doors
    engine.add_sticker_pro("SPEED_OS", (0.8, 0.6), scale=0.8, rotation=0, color_override=(255,255,255))
    engine.add_sticker_pro("SPEED_OS", (0.2, 0.6), scale=0.8, rotation=0, color_override=(255,255,255))
    # Roof
    engine.add_sticker_pro("KINETIC", (0.5, 0.5), scale=1.0, rotation=0)
    
    engine.save()

def create_street_style_skin(team_name, colors):
    """
    Street Style: Graffiti, abstract, sticker bomb.
    """
    engine = ProSkinEngine(team_name=team_name)
    engine.set_base_material(colors["base"], "matte")
    
    # Random sticker bomb
    import random
    logos = ["VOLT", "FLUX", "HEXWORKS", "KINETIC"]
    rng = random.Random(42)
    
    for _ in range(20):
        name = rng.choice(logos)
        x = rng.uniform(0, 1)
        y = rng.uniform(0, 1)
        rot = rng.uniform(0, 360)
        s = rng.uniform(0.3, 0.6)
        engine.add_sticker_pro(name, (x, y), scale=s, rotation=rot)
        
    engine.save()

if __name__ == "__main__":
    # Racing
    cols_racing = {
        "base": (20, 20, 220), # Blue
        "accent": (255, 255, 255), # White
        "highlight": (255, 50, 50) # Red
    }
    create_pro_racing_skin("ProRacing_v1", cols_racing)
    compress_skin("ProRacing_v1")
    
    # Street
    cols_street = {
        "base": (40, 40, 40),
        "accent": (100, 100, 100),
        "highlight": (255, 255, 0)
    }
    create_street_style_skin("Street_Flux", cols_street)
    compress_skin("Street_Flux")
