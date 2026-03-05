import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops

class LogoGenerator:
    def __init__(self):
        # We don't have custom fonts installed, so we'll use PIL's default or try to find a system font.
        # Actually, let's try to find a sans-serif font on the system.
        self.font_path = self._find_font()
        
    def _find_font(self):
        # Common paths for a bold sans font
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]
        for c in candidates:
            try:
                ImageFont.truetype(c, 20)
                return c
            except:
                continue
        return None # Fallback to default load_default()

    def generate_logo(self, company_name, color=(255,255,255), style="modern"):
        """Generates a logo with an icon and text."""
        W, H = 512, 128
        img = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(img)
        
        # 1. Icon Generation
        icon_size = 100
        icon = Image.new("RGBA", (icon_size, icon_size), (0,0,0,0))
        d_icon = ImageDraw.Draw(icon)
        
        # Seed based on name for consistency
        rng = random.Random(company_name)
        
        # Procedural Icon Shapes
        shape_type = rng.choice(["hex", "circle", "triangle", "abstract"])
        
        if shape_type == "hex":
            # Hexagon
            points = []
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                x = icon_size/2 + (icon_size/2 - 5) * 1.0 * math.cos(angle)
                y = icon_size/2 + (icon_size/2 - 5) * 1.0 * math.sin(angle)
                points.append((x,y))
            d_icon.polygon(points, outline=color, width=8)
            # Inner dot
            d_icon.ellipse([icon_size/2-10, icon_size/2-10, icon_size/2+10, icon_size/2+10], fill=color)
            
        elif shape_type == "circle":
            # Circle with cut
            d_icon.ellipse([5, 5, icon_size-5, icon_size-5], outline=color, width=8)
            # Cut
            d_icon.rectangle([icon_size/2, 0, icon_size, icon_size/2], fill=(0,0,0,0))
            
        elif shape_type == "triangle":
            # Triangle
            d_icon.polygon([(icon_size/2, 5), (5, icon_size-5), (icon_size-5, icon_size-5)], outline=color, width=8)
            
        else:
            # Abstract Lines
            for _ in range(3):
                y = rng.randint(10, icon_size-10)
                d_icon.line([(0, y), (icon_size, y)], fill=color, width=6)
                
        # Paste Icon
        img.paste(icon, (10, (H-icon_size)//2))
        
        # 2. Text Generation
        if self.font_path:
            font = ImageFont.truetype(self.font_path, 80)
        else:
            font = ImageFont.load_default()
            
        # Draw Text
        # We need to render text to a separate image to handle spacing/kerning? No, simple for now.
        text_bbox = d.textbbox((0,0), company_name, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        
        d.text((icon_size + 40, (H-text_h)//2 - 10), company_name, fill=color, font=font)
        
        # Trim transparency
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        return img

    def save_demo_logos(self):
        companies = ["ARC LABS", "HEXWORKS", "SPEED.OS", "KINETIC", "AERO_DYNAMICS", "VOLT", "FLUX"]
        import os
        os.makedirs("assets/generated_logos", exist_ok=True)
        
        for c in companies:
            logo = self.generate_logo(c)
            logo.save(f"assets/generated_logos/{c.replace('.','_')}.png")
            print(f"Generated assets/generated_logos/{c}.png")

import math

if __name__ == "__main__":
    gen = LogoGenerator()
    gen.save_demo_logos()
