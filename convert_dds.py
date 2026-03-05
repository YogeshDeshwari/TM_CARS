from PIL import Image
import os

def convert_dds_to_png(path):
    try:
        img = Image.open(path)
        out_path = path.replace(".dds", ".png")
        img.save(out_path)
        print(f"Converted {path} to {out_path}")
    except Exception as e:
        print(f"Failed to convert {path}: {e}")

if __name__ == "__main__":
    files = [
        "base_dds/StadiumCarSkin.dds",
        "base_dds/StadiumCarDetails.dds",
        "base_dds/StadiumCarPilot.dds"
    ]
    for f in files:
        if os.path.exists(f):
            convert_dds_to_png(f)
        else:
            print(f"File not found: {f}")
