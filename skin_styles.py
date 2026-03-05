import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops
from typing import Tuple, Dict, Any
import skin_utils

class SkinGenerator:
    def __init__(self, size=2048):
        self.size = size
        self.width = size
        self.height = size
        
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        """Returns {'Diffuse': img, 'Details': img, 'Icon': img}"""
        raise NotImplementedError

class TechLuxeStyle(SkinGenerator):
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        base_col = colors["base"]
        accent_col = colors["accent"]
        highlight_col = colors["highlight"]
        
        # 1. Base: Matte Black (or base color)
        diffuse = Image.new("RGBA", (self.size, self.size), base_col + (255,))
        
        # 2. Pattern: Circuit Traces (Gold/Accent)
        # Mask: Apply heavily on Sides (Bottom Half), lightly on Hood (Top Left)
        circuits = skin_utils.generate_circuit_traces(self.size, accent_col, density="medium")
        
        # Create a mask that reveals circuits mostly on the sides
        mask = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(mask)
        # Sidepods area (approximate for Stadium)
        draw.rectangle([0, self.size//2, self.size, self.size], fill=255) 
        # Faint on hood
        draw.rectangle([0, 0, self.size//2, self.size//2], fill=50)
        
        circuits_masked = Image.new("RGBA", (self.size, self.size), (0,0,0,0))
        circuits_masked.paste(circuits, (0,0), mask)
        
        diffuse = Image.alpha_composite(diffuse, circuits_masked)
        
        # 3. Highlights: Glowing cyan/highlight nodes
        nodes = skin_utils.generate_halftone(self.size, highlight_col, density=0.1) # Sparse dots
        diffuse = Image.alpha_composite(diffuse, nodes)
        
        # 4. Text: Simple Futura-like placement
        # (Placeholder for text rendering - would use a font here)
        
        # 5. Details Map: Shiny Circuits, Matte Base
        details = skin_utils.apply_material_finish(diffuse, "matte")
        # Make the circuits shiny in the Details map
        circuit_gloss = circuits.convert("L").point(lambda p: p * 0.8) # High gloss
        details_alpha = details.getchannel("A")
        details_alpha = ImageChops.lighter(details_alpha, circuit_gloss)
        details.putalpha(details_alpha)
        
        return {"Diffuse": diffuse, "Details": details}

class KintsugiRoyalStyle(SkinGenerator):
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        base_col = colors["base"]
        gold_col = (218, 165, 32) # Fixed Gold
        
        # 1. Base: Marble texture (simulated via noise)
        diffuse = Image.new("RGBA", (self.size, self.size), base_col + (255,))
        # Add some cloud noise for marble effect
        noise = Image.effect_noise((self.size//4, self.size//4), 10).resize((self.size, self.size), Image.Resampling.BICUBIC)
        noise = noise.convert("L")
        # Colorize noise slightly
        marble_tint = Image.new("RGBA", (self.size, self.size), colors["secondary"] + (100,))
        diffuse.paste(marble_tint, (0,0), noise)
        
        # 2. Pattern: Gold Cracks
        cracks = skin_utils.generate_kintsugi_cracks(self.size, gold_col)
        diffuse = Image.alpha_composite(diffuse, cracks)
        
        # 3. Details: Satin finish, but cracks are Metallic
        details = skin_utils.apply_material_finish(diffuse, "satin")
        crack_gloss = cracks.convert("L").point(lambda p: 255 if p > 10 else 0)
        details_alpha = details.getchannel("A")
        details_alpha = ImageChops.lighter(details_alpha, crack_gloss)
        details.putalpha(details_alpha)
        
        return {"Diffuse": diffuse, "Details": details}

class RacingSportStyle(SkinGenerator):
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        # 1. Base: Glossy Paint
        diffuse = Image.new("RGBA", (self.size, self.size), colors["base"] + (255,))
        d = ImageDraw.Draw(diffuse)
        
        # 2. Big Asymmetrical Stripes
        # Stripe 1: Thick accent
        d.rectangle([self.size*0.3, 0, self.size*0.45, self.size], fill=colors["accent"]+(255,))
        # Stripe 2: Thin highlight
        d.rectangle([self.size*0.46, 0, self.size*0.48, self.size], fill=colors["highlight"]+(255,))
        
        # 3. Carbon Fiber Splitters (Bottom of texture usually maps to undercarriage/sides)
        carbon = skin_utils._generate_carbon_pattern(self.size)
        carbon_colored = ImageOps.colorize(carbon, "black", (30,30,30))
        
        # Mask for carbon (Simulate side skirts)
        diffuse.paste(carbon_colored, (0, int(self.size*0.8)), mask=carbon) # Simple paste at bottom
        
        # 4. Details: High Gloss Paint, Matte Carbon
        details = skin_utils.apply_material_finish(diffuse, "gloss")
        # Todo: Mask out the carbon area to be matte in details
        
        return {"Diffuse": diffuse, "Details": details}

class StealthOpsStyle(SkinGenerator):
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        # 1. Base: Dark Grey / Camo
        diffuse = Image.new("RGBA", (self.size, self.size), (30,30,30,255))
        
        # 2. Geo Pattern overlay
        # (Simplified for now)
        
        # 3. Warning Accents (Orange/Red)
        d = ImageDraw.Draw(diffuse)
        # Dazzle stripes
        d.rectangle([0, self.size*0.6, self.size, self.size*0.65], fill=colors["highlight"]+(255,))
        
        # 4. Details: Ultra Matte
        details = skin_utils.apply_material_finish(diffuse, "matte")
        
        return {"Diffuse": diffuse, "Details": details}

class HeritageStyle(SkinGenerator):
    def generate(self, team_name: str, colors: Dict[str, Tuple[int,int,int]]) -> Dict[str, Image.Image]:
        # 1. Base: Deep rich color
        diffuse = Image.new("RGBA", (self.size, self.size), colors["base"] + (255,))
        d = ImageDraw.Draw(diffuse)
        
        # 2. Pin Stripes (Center)
        center = self.size // 2
        w = self.size // 20
        d.rectangle([center - w, 0, center + w, self.size], fill=colors["accent"]+(255,))
        # Thin borders
        d.line([(center - w, 0), (center - w, self.size)], fill=colors["highlight"]+(255,), width=4)
        d.line([(center + w, 0), (center + w, self.size)], fill=colors["highlight"]+(255,), width=4)
        
        # 3. Details: Classic Gloss
        details = skin_utils.apply_material_finish(diffuse, "gloss")
        
        return {"Diffuse": diffuse, "Details": details}
