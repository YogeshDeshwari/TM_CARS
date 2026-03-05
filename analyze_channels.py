import numpy as np
from PIL import Image
import os

def analyze_channels(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Analyzing {os.path.basename(path)}...")
    img = Image.open(path).convert("RGBA")
    r, g, b, a = img.split()
    
    channels = {'R': r, 'G': g, 'B': b, 'A': a}
    
    for name, chan in channels.items():
        data = np.array(chan)
        min_val = data.min()
        max_val = data.max()
        mean_val = data.mean()
        unique_vals = len(np.unique(data))
        
        print(f"  {name}: Min={min_val}, Max={max_val}, Mean={mean_val:.2f}, Unique={unique_vals}")
        
        # Heuristic for usage
        if unique_vals == 1:
            print(f"    -> Constant {min_val} (Likely unused or fill)")
        elif unique_vals < 20:
            print(f"    -> Low variance (Mask or stepped value?)")
        else:
            print(f"    -> High variance (Texture/Data)")

files = [
    "out/inspection/steve/Details.dds",
    "out/inspection/steve/Illum.dds",
    "out/inspection/steve/Projshad.dds",
    "out/inspection/liquicity/Details.dds"
]

for f in files:
    analyze_channels(f)
