"""
Spatial-Aware Car Geometry Engine

Automatically classifies UV islands into semantic car parts and provides
intelligent color/material assignment based on 3D car anatomy.

Usage:
    from car_geometry import CarGeometry
    
    # From UV island data
    geo = CarGeometry.from_island_data(islands, tex_size=2048)
    
    # Get part classification
    part = geo.get_part(island_id=5)
    print(part.name)  # "SIDEPOD"
    print(part.role)  # "hero"
    
    # Get color for a part
    color = geo.get_color(island_id=5, palette)
    
    # Get all islands by role
    hero_ids = geo.get_islands_by_role("hero")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import json


class ColorRole(Enum):
    """Color role determines how a part should be colored."""
    HERO = "hero"           # Primary attention areas (base + accent)
    SECONDARY = "secondary" # Supporting areas (secondary color)
    ACCENT = "accent"       # Highlight details (accent color)
    DARKEN = "darken"       # Should be darker than base
    NEUTRAL = "neutral"     # Dark/black, minimal attention


class FinishType(Enum):
    """Material finish for the part."""
    HIGH_GLOSS = "high_gloss"
    MEDIUM_GLOSS = "medium_gloss"
    MATTE = "matte"
    CARBON = "carbon"


class FlowDirection(Enum):
    """Gradient flow direction for the part."""
    HORIZONTAL = "horizontal"   # Left-to-right in UV (front-to-back on car)
    VERTICAL = "vertical"       # Top-to-bottom in UV
    RADIAL = "radial"           # From center outward
    DIAGONAL = "diagonal"       # Corner to corner
    NONE = "none"               # No gradient flow


@dataclass
class PartInfo:
    """Information about a classified car part."""
    name: str
    role: ColorRole
    finish: FinishType = FinishType.MEDIUM_GLOSS
    flow: FlowDirection = FlowDirection.NONE
    description: str = ""


@dataclass
class IslandInfo:
    """Geometric information about a UV island."""
    id: int
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    area: int
    center: Tuple[int, int]
    aspect_ratio: float
    rel_x: float  # Normalized X position (0-1)
    rel_y: float  # Normalized Y position (0-1)
    mirror_id: Optional[int] = None  # ID of mirrored pair, if any
    part: Optional[PartInfo] = None


# Part classification rules - VERIFIED against in-game UV debug (Feb 2026)
PART_DEFINITIONS = {
    "MAIN_BODY_SIDE": PartInfo(
        name="MAIN_BODY_SIDE",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Large vertical body panel (Island 1)"
    ),
    "TOP_BODY": PartInfo(
        name="TOP_BODY",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.VERTICAL,
        description="Roof/top body surface (Island 2, green)"
    ),
    "NOSE": PartInfo(
        name="NOSE",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Front nose cone (Island 3, blue-gray)"
    ),
    "UNDERTRAY": PartInfo(
        name="UNDERTRAY",
        role=ColorRole.NEUTRAL,
        finish=FinishType.MATTE,
        flow=FlowDirection.NONE,
        description="Lower body/floor panel (Island 4)"
    ),
    "SIDEPOD": PartInfo(
        name="SIDEPOD",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Side pods (Islands 5/6, purple, mirrored)"
    ),
    "REAR_SECTION": PartInfo(
        name="REAR_SECTION",
        role=ColorRole.SECONDARY,
        finish=FinishType.MEDIUM_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Rear body section (Island 7)"
    ),
    "FRONT_WING": PartInfo(
        name="FRONT_WING",
        role=ColorRole.ACCENT,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Front wing (Island 8, cyan)"
    ),
    "SIDE_SKIRT": PartInfo(
        name="SIDE_SKIRT",
        role=ColorRole.ACCENT,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Side skirts (Islands 9/10, purple bottom, mirrored)"
    ),
    "PILLAR": PartInfo(
        name="PILLAR",
        role=ColorRole.NEUTRAL,
        finish=FinishType.MATTE,
        flow=FlowDirection.VERTICAL,
        description="Structural pillars (Island 11)"
    ),
    "FENDER": PartInfo(
        name="FENDER",
        role=ColorRole.SECONDARY,
        finish=FinishType.MEDIUM_GLOSS,
        flow=FlowDirection.DIAGONAL,
        description="Fenders/wheel arch tops (Islands 12/13, olive, mirrored)"
    ),
    "REAR_WING_ENDPLATE": PartInfo(
        name="REAR_WING_ENDPLATE",
        role=ColorRole.ACCENT,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.VERTICAL,
        description="Rear wing endplates (Islands 15/16, light blue, mirrored)"
    ),
    "MUDGUARD": PartInfo(
        name="MUDGUARD",
        role=ColorRole.DARKEN,
        finish=FinishType.MATTE,
        flow=FlowDirection.RADIAL,
        description="Inner wheel arch liners (Islands 17/18/26/27, mirrored)"
    ),
    "BRAKE_DUCT": PartInfo(
        name="BRAKE_DUCT",
        role=ColorRole.ACCENT,
        finish=FinishType.MATTE,
        flow=FlowDirection.NONE,
        description="Brake cooling ducts (Islands 19/20/24/25, mirrored)"
    ),
    "NOSE_DETAIL": PartInfo(
        name="NOSE_DETAIL",
        role=ColorRole.ACCENT,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.NONE,
        description="Small nose/front details (Islands 21/22/23)"
    ),
    "COCKPIT": PartInfo(
        name="COCKPIT",
        role=ColorRole.NEUTRAL,
        finish=FinishType.MATTE,
        flow=FlowDirection.NONE,
        description="Cockpit area"
    ),
    "WHEEL_CAP": PartInfo(
        name="WHEEL_CAP",
        role=ColorRole.ACCENT,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.RADIAL,
        description="Wheel caps/centers"
    ),
    "TINY_DETAIL": PartInfo(
        name="TINY_DETAIL",
        role=ColorRole.NEUTRAL,
        finish=FinishType.MATTE,
        flow=FlowDirection.NONE,
        description="Very small details"
    ),
    "BODY_PANEL": PartInfo(
        name="BODY_PANEL",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Generic body panel"
    ),
    "BODY_DETAIL": PartInfo(
        name="BODY_DETAIL",
        role=ColorRole.SECONDARY,
        finish=FinishType.MEDIUM_GLOSS,
        flow=FlowDirection.NONE,
        description="Generic body detail"
    ),
    "DETAIL": PartInfo(
        name="DETAIL",
        role=ColorRole.NEUTRAL,
        finish=FinishType.MATTE,
        flow=FlowDirection.NONE,
        description="Generic small detail (Island 14)"
    ),
    # Legacy aliases for backwards compatibility
    "NOSE_TAIL": PartInfo(
        name="NOSE_TAIL",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.HORIZONTAL,
        description="Front nose or rear tail section (legacy)"
    ),
    "HOOD": PartInfo(
        name="HOOD",
        role=ColorRole.HERO,
        finish=FinishType.HIGH_GLOSS,
        flow=FlowDirection.VERTICAL,
        description="Hood/bonnet area (legacy - use FRONT_WING for Island 8)"
    ),
}


class CarGeometry:
    """
    Spatial-aware car geometry engine.
    
    Analyzes UV island data to classify car parts and provide intelligent
    color/material assignment based on 3D car anatomy.
    """
    
    def __init__(self, tex_size: int = 2048):
        self.tex_size = tex_size
        self.islands: Dict[int, IslandInfo] = {}
        self._mirror_map: Dict[int, int] = {}
        self._role_index: Dict[ColorRole, List[int]] = {role: [] for role in ColorRole}
        self._part_index: Dict[str, List[int]] = {}
    
    @classmethod
    def from_island_data(
        cls,
        islands: List[Dict],
        tex_size: int = 2048,
    ) -> "CarGeometry":
        """
        Create CarGeometry from UV island data.
        
        Args:
            islands: List of island dicts with 'id', 'bbox', 'area', 'center'
            tex_size: Texture size (default 2048)
        
        Returns:
            Configured CarGeometry instance
        """
        geo = cls(tex_size=tex_size)
        
        # First pass: build island info
        for island_data in islands:
            island = geo._build_island_info(island_data)
            geo.islands[island.id] = island
        
        # Second pass: find mirrored pairs
        geo._detect_mirrored_pairs()
        
        # Third pass: classify parts
        for island_id in geo.islands:
            geo._classify_island(island_id)
        
        # Build indices
        geo._build_indices()
        
        return geo
    
    @classmethod
    def from_json_file(cls, path: str) -> "CarGeometry":
        """Load from UV atlas JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_island_data(data['islands'], tex_size=data.get('size', 2048))
    
    def _build_island_info(self, data: Dict) -> IslandInfo:
        """Build IslandInfo from raw data."""
        x0, y0, x1, y1 = data['bbox']
        cx, cy = data['center']
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        
        return IslandInfo(
            id=data['id'],
            bbox=(x0, y0, x1, y1),
            area=data['area'],
            center=(cx, cy),
            aspect_ratio=w / h,
            rel_x=cx / self.tex_size,
            rel_y=cy / self.tex_size,
        )
    
    def _detect_mirrored_pairs(self):
        """Detect left/right mirrored island pairs."""
        y_mirror_sum = self.tex_size * 1.0547  # ~2160 for 2048
        used: Set[int] = set()
        
        island_list = list(self.islands.values())
        for a in island_list:
            if a.id in used:
                continue
            
            for b in island_list:
                if a.id == b.id or b.id in used:
                    continue
                
                # Check X alignment
                ax0, ay0, ax1, ay1 = a.bbox
                bx0, by0, bx1, by1 = b.bbox
                x_match = abs(ax0 - bx0) < 20 and abs(ax1 - bx1) < 20
                
                # Check area similarity
                area_match = abs(a.area - b.area) < 100
                
                # Check Y mirroring
                y_sum = a.center[1] + b.center[1]
                y_mirror = abs(y_sum - y_mirror_sum) < 40
                
                if x_match and area_match and y_mirror:
                    self._mirror_map[a.id] = b.id
                    self._mirror_map[b.id] = a.id
                    a.mirror_id = b.id
                    b.mirror_id = a.id
                    used.add(a.id)
                    used.add(b.id)
                    break
    
    def _classify_island(self, island_id: int):
        """Classify an island into a car part."""
        island = self.islands[island_id]
        part_name = self._determine_part_name(island)
        island.part = PART_DEFINITIONS.get(part_name, PART_DEFINITIONS["DETAIL"])
    
    def _determine_part_name(self, island: IslandInfo) -> str:
        """
        Determine the part name based on geometric heuristics.
        
        VERIFIED against in-game UV debug (Feb 2026):
        - Island 1: MAIN_BODY_SIDE (large vertical panel)
        - Island 2: TOP_BODY (roof/top, green in debug)
        - Island 3: NOSE (front nose cone, blue-gray)
        - Island 4: UNDERTRAY (lower body/floor)
        - Island 5/6: SIDEPOD (purple, mirrored pair)
        - Island 7: REAR_SECTION (rear body)
        - Island 8: FRONT_WING (cyan, NOT hood)
        - Island 9/10: SIDE_SKIRT (purple bottom panels, mirrored)
        - Island 11: PILLAR (vertical strip)
        - Island 12/13: FENDER (olive, wheel arch tops, mirrored)
        - Island 14: DETAIL
        - Island 15/16: REAR_WING_ENDPLATE (light blue, mirrored)
        - Island 17/18: MUDGUARD (inner wheel arch, mirrored)
        - Island 19/20: BRAKE_DUCT (mirrored)
        - Island 21/22/23: NOSE_DETAIL (small front details)
        - Island 24/25: BRAKE_DUCT (mirrored)
        - Island 26/27: MUDGUARD (inner wheel arch, mirrored)
        """
        island_id = island.id
        
        # VERIFIED MAPPINGS (from in-game UV debug screenshots)
        VERIFIED_PARTS = {
            1: "MAIN_BODY_SIDE",
            2: "TOP_BODY",
            3: "NOSE",
            4: "BODY_PANEL",
            5: "SIDEPOD",
            6: "SIDEPOD",
            7: "REAR_SECTION",
            8: "FRONT_WING",
            9: "SIDE_SKIRT",
            10: "SIDE_SKIRT",
            11: "PILLAR",
            12: "FENDER",
            13: "FENDER",
            14: "DETAIL",
            15: "REAR_WING_ENDPLATE",
            16: "REAR_WING_ENDPLATE",
            17: "MUDGUARD",
            18: "MUDGUARD",
            19: "BRAKE_DUCT",
            20: "BRAKE_DUCT",
            21: "NOSE_DETAIL",
            22: "NOSE_DETAIL",
            23: "NOSE_DETAIL",
            24: "BRAKE_DUCT",
            25: "BRAKE_DUCT",
            26: "MUDGUARD",
            27: "MUDGUARD",
        }
        
        if island_id in VERIFIED_PARTS:
            return VERIFIED_PARTS[island_id]
        
        # Fallback to geometric heuristics for unknown islands
        area = island.area
        ar = island.aspect_ratio
        has_mirror = island.mirror_id is not None
        rel_y = island.rel_y
        
        if area > 25000:
            return "BODY_PANEL"
        if area > 10000:
            return "BODY_PANEL"
        if area > 5000:
            return "BODY_DETAIL"
        if area > 1500:
            if has_mirror:
                return "BODY_DETAIL"
            return "DETAIL"
        if area > 500:
            return "DETAIL"
        return "TINY_DETAIL"
    
    def _build_indices(self):
        """Build lookup indices for fast queries."""
        self._role_index = {role: [] for role in ColorRole}
        self._part_index = {}
        
        for island_id, island in self.islands.items():
            if island.part:
                role = island.part.role
                self._role_index[role].append(island_id)
                
                part_name = island.part.name
                if part_name not in self._part_index:
                    self._part_index[part_name] = []
                self._part_index[part_name].append(island_id)
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_part(self, island_id: int) -> Optional[PartInfo]:
        """Get part classification for an island."""
        island = self.islands.get(island_id)
        return island.part if island else None
    
    def get_islands_by_role(self, role: ColorRole) -> List[int]:
        """Get all island IDs with a specific color role."""
        if isinstance(role, str):
            role = ColorRole(role)
        return list(self._role_index.get(role, []))
    
    def get_islands_by_part(self, part_name: str) -> List[int]:
        """Get all island IDs classified as a specific part."""
        return list(self._part_index.get(part_name, []))
    
    def get_mirror(self, island_id: int) -> Optional[int]:
        """Get the mirrored pair ID for an island."""
        return self._mirror_map.get(island_id)
    
    def get_color_for_island(
        self,
        island_id: int,
        base_rgb: Tuple[int, int, int],
        accent_rgb: Tuple[int, int, int],
        secondary_rgb: Tuple[int, int, int],
        neutral_rgb: Tuple[int, int, int] = (30, 30, 30),
    ) -> Tuple[int, int, int]:
        """
        Get the recommended color for an island based on its role.
        
        Args:
            island_id: The island ID
            base_rgb: Base/primary color
            accent_rgb: Accent/highlight color
            secondary_rgb: Secondary/stripe color
            neutral_rgb: Neutral/dark color (default dark gray)
        
        Returns:
            RGB tuple for the island
        """
        island = self.islands.get(island_id)
        if not island or not island.part:
            return base_rgb
        
        role = island.part.role
        
        if role == ColorRole.HERO:
            return base_rgb
        elif role == ColorRole.SECONDARY:
            return secondary_rgb
        elif role == ColorRole.ACCENT:
            return accent_rgb
        elif role == ColorRole.DARKEN:
            # Darken the base color by 40%
            return (
                int(base_rgb[0] * 0.6),
                int(base_rgb[1] * 0.6),
                int(base_rgb[2] * 0.6),
            )
        else:  # NEUTRAL
            return neutral_rgb
    
    def get_finish_alpha_for_island(self, island_id: int) -> int:
        """
        Get the recommended finish alpha value for an island.
        
        Returns alpha value for TMNF finish channel (0x8E = neutral).
        """
        island = self.islands.get(island_id)
        if not island or not island.part:
            return 0x8E  # Neutral
        
        finish = island.part.finish
        
        if finish == FinishType.HIGH_GLOSS:
            return 0x68  # More reflective
        elif finish == FinishType.MEDIUM_GLOSS:
            return 0x8E  # Neutral
        elif finish == FinishType.MATTE:
            return 0xA8  # Less reflective
        elif finish == FinishType.CARBON:
            return 0xB0  # Very matte
        else:
            return 0x8E
    
    def get_flow_direction(self, island_id: int) -> FlowDirection:
        """Get the gradient flow direction for an island."""
        island = self.islands.get(island_id)
        if not island or not island.part:
            return FlowDirection.NONE
        return island.part.flow
    
    def summary(self) -> str:
        """Get a human-readable summary of the car geometry."""
        lines = ["Car Geometry Summary", "=" * 50]
        
        for role in ColorRole:
            ids = self._role_index.get(role, [])
            if ids:
                lines.append(f"\n{role.value.upper()} ({len(ids)} islands):")
                for island_id in ids:
                    island = self.islands[island_id]
                    mirror = f" <-> {island.mirror_id}" if island.mirror_id else ""
                    lines.append(f"  {island_id:3d}: {island.part.name}{mirror}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export geometry to dictionary for serialization."""
        return {
            "tex_size": self.tex_size,
            "islands": {
                id: {
                    "bbox": island.bbox,
                    "area": island.area,
                    "center": island.center,
                    "aspect_ratio": island.aspect_ratio,
                    "mirror_id": island.mirror_id,
                    "part_name": island.part.name if island.part else None,
                    "role": island.part.role.value if island.part else None,
                    "finish": island.part.finish.value if island.part else None,
                    "flow": island.part.flow.value if island.part else None,
                }
                for id, island in self.islands.items()
            },
        }


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def load_stadium_geometry() -> CarGeometry:
    """Load the standard Stadium car geometry from the atlas file."""
    import os
    atlas_path = os.path.join(
        os.path.dirname(__file__),
        "assets/uv_atlas/standard_stadium_islands_2048.json"
    )
    return CarGeometry.from_json_file(atlas_path)


if __name__ == "__main__":
    # Demo: load and print summary
    try:
        geo = load_stadium_geometry()
        print(geo.summary())
        print()
        
        # Show color recommendations for a sample palette
        print("Sample Color Recommendations:")
        print("-" * 50)
        base = (200, 50, 50)      # Red
        accent = (255, 200, 0)    # Yellow
        secondary = (100, 100, 100)  # Gray
        
        for island_id in sorted(geo.islands.keys()):
            color = geo.get_color_for_island(island_id, base, accent, secondary)
            alpha = geo.get_finish_alpha_for_island(island_id)
            flow = geo.get_flow_direction(island_id)
            part = geo.get_part(island_id)
            print(f"Island {island_id:2d} ({part.name:15s}): RGB={color}, Alpha=0x{alpha:02X}, Flow={flow.value}")
    except FileNotFoundError as e:
        print(f"Atlas file not found: {e}")
        print("Run this from the TM_CARS directory or generate the atlas first.")
