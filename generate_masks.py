import numpy as np
from PIL import Image, ImageFilter
import os

def generate_paint_mask():
    # Load the converted base skin
    if not os.path.exists("base_dds/StadiumCarSkin.png"):
        print("Please run convert_dds.py first.")
        return

    img = Image.open("base_dds/StadiumCarSkin.png").convert("RGBA")
    arr = np.array(img)

    # The default skin is Blue with Grey/Black mechanical parts.
    # We want to isolate the Blue parts.
    r, g, b, a = arr.T
    
    # Mask: Blue is dominant
    mask = (b > r + 20) & (b > g + 20) & (b > 50)
    
    # Create binary mask image
    mask_img = Image.fromarray((mask.T * 255).astype(np.uint8), mode="L")
    
    # Cleanup: Morphological closing to fill holes
    for _ in range(2):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    for _ in range(2):
        mask_img = mask_img.filter(ImageFilter.MinFilter(3))
        
    os.makedirs("assets/masks", exist_ok=True)
    mask_img.save("assets/masks/paint_mask.png")
    print("Generated assets/masks/paint_mask.png")

    # Also generate a "Chassis Mask" (Inverse of Paint)
    chassis_mask = Image.eval(mask_img, lambda x: 255 - x)
    chassis_mask.save("assets/masks/chassis_mask.png")
    print("Generated assets/masks/chassis_mask.png")

if __name__ == "__main__":
    generate_paint_mask()
