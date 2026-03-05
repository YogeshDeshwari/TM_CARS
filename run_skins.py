import zipfile
import os
from pathlib import Path
from pro_skin_engine import create_cyber_tech_skin, ProSkinEngine, skin_utils
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter

def create_kintsugi_royal_skin(team_name, colors):
    """
    Kintsugi Royal: White Marble with Glowing Gold cracks.
    """
    engine = ProSkinEngine(team_name=team_name)
    
    # 1. Base: White Marble
    # Generate marble texture
    marble_base = Image.effect_noise((engine.size//4, engine.size//4), 10).resize((engine.size, engine.size), Image.Resampling.BICUBIC).convert("L")
    marble_base = ImageOps.colorize(marble_base, (220, 220, 220), (255, 255, 255))
    
    engine.diffuse.paste(marble_base, (0,0), engine.paint_mask)
    engine.details.paste(skin_utils.apply_material_finish(marble_base, "satin"), (0,0), engine.paint_mask)
    
    # 2. Pattern: Gold Cracks (Glowing)
    gold_col = (255, 200, 0)
    
    def cracks_pat(size, col):
        return skin_utils.generate_kintsugi_cracks(size, col)
    
    # Add visible cracks to Diffuse
    engine.add_pattern(cracks_pat, gold_col, opacity=1.0)
    
    # Add GLOWING cracks to Illum
    # We want them to pulse/glow, so we add them to the Illum map
    engine.add_glow(cracks_pat, gold_col, intensity=0.8)
    
    # 3. Metallic Finish for Cracks in Details
    # We need to manually update details for the cracks to be shiny
    cracks = cracks_pat(engine.size, (255,255,255))
    crack_alpha = cracks.getchannel("A")
    # Make cracks super shiny (255)
    d_alpha = engine.details.getchannel("A")
    d_alpha = ImageChops.lighter(d_alpha, crack_alpha)
    engine.details.putalpha(d_alpha)
    
    engine.save()

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

if __name__ == "__main__":
    # Example Usage
    cols_cyber = {
        "base": (30, 30, 35),
        "accent": (0, 255, 255), # Cyan
        "highlight": (255, 0, 255)
    }
    create_cyber_tech_skin("CyberTech_Pro", cols_cyber)
    compress_skin("CyberTech_Pro")
    
    create_kintsugi_royal_skin("Kintsugi_Royal", {})
    compress_skin("Kintsugi_Royal")
