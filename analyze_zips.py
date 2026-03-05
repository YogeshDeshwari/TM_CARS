import zipfile
import os
from pathlib import Path

def analyze_skins():
    examples_dir = Path("examples")
    zips = list(examples_dir.glob("*.zip"))
    
    print(f"Found {len(zips)} skins to analyze.\n")
    
    for z in zips:
        print(f"--- ANALYZING: {z.name} ---")
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                files = zf.namelist()
                
                # Categorize files
                textures = [f for f in files if f.endswith('.dds')]
                images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                audio = [f for f in files if f.endswith(('.wav', '.ogg'))]
                others = [f for f in files if f not in textures and f not in images and f not in audio and not f.endswith('/')]
                
                print(f"  Total Files: {len(files)}")
                if textures:
                    print(f"  Textures ({len(textures)}): {', '.join(sorted(textures))}")
                if images:
                    print(f"  Images ({len(images)}): {', '.join(sorted(images))}")
                if audio:
                    print(f"  Audio ({len(audio)}): {len(audio)} files (e.g. {audio[0]})")
                if others:
                    print(f"  Other: {', '.join(others)}")
                
                # Check for critical skin files
                has_diffuse = any("Diffuse" in f for f in files)
                has_details = any("Details" in f for f in files)
                has_icon = any("Icon" in f for f in files)
                has_projshad = any("ProjShad" in f for f in files)
                has_illum = any("Illum" in f for f in files)
                
                print("  Key Components:")
                print(f"    - Diffuse (Paint): {'YES' if has_diffuse else 'NO'}")
                print(f"    - Details (Material): {'YES' if has_details else 'NO'}")
                print(f"    - Icon: {'YES' if has_icon else 'NO'}")
                print(f"    - ProjShad (Shadow): {'YES' if has_projshad else 'NO'}")
                print(f"    - SelfIllum (Glow): {'YES' if has_illum else 'NO'}")
                
        except Exception as e:
            print(f"  Error reading zip: {e}")
        print("\n")

if __name__ == "__main__":
    analyze_skins()
