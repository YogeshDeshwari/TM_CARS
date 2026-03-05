import struct
import os

def get_dds_info(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'DDS ':
            return "Not a DDS file"
        
        header = f.read(124)
        height = struct.unpack_from('<I', header, 8)[0]
        width = struct.unpack_from('<I', header, 12)[0]
        pixel_format_flags = struct.unpack_from('<I', header, 76)[0]
        four_cc = header[80:84].decode('utf-8', 'ignore')
        
        return f"{width}x{height}, Format: {four_cc}"

files = ["Diffuse.dds", "Details.dds", "Icon.dds", "ProjShad.dds"]
base_dir = "out/inspection/liquicity"

print("--- Liquicity Skin Analysis ---")
for fn in files:
    path = os.path.join(base_dir, fn)
    if os.path.exists(path):
        print(f"{fn}: {get_dds_info(path)}")
    else:
        print(f"{fn}: Missing")
