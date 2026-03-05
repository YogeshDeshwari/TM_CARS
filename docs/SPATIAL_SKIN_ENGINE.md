# Spatial-Aware Skin Engine

## Vision

Transform skin generation from "paint by region" to **"design with car anatomy knowledge"**.

The engine understands the 3D car structure through UV island analysis, enabling:
- Automatic part classification (sidepods, nose, fenders, etc.)
- Intelligent color placement based on part semantics
- Gradients that follow the car's body flow
- Per-part material properties (matte cockpit, glossy sidepods)
- Design rules that create professional-looking skins automatically

---

## 1. Automatic Part Classification

### Detection Algorithm

```
For each UV island:
1. Compute geometric properties:
   - Area (total pixels)
   - Aspect ratio (width/height)
   - Position in UV space (rel_x, rel_y)
   - Shape compactness

2. Find mirrored pairs:
   - Y coords sum to ~2160 (for 2048 texture)
   - Same X position, same area
   - Left/right symmetry in 3D

3. Apply classification rules:
   - HUGE + center = MAIN_BODY
   - HUGE + mirrored + wide = SIDEPOD
   - MEDIUM + mirrored + wide = FENDER
   - SMALL + mirrored + square = MUDGUARD
   - etc.
```

### Part Taxonomy

| Part Name | Characteristics | Color Role | Example Islands |
|-----------|----------------|------------|-----------------|
| MAIN_BODY_SIDE | HUGE, vertical | hero | 1 |
| TOP_BODY | HUGE, center Y | hero | 2 |
| NOSE_TAIL | HUGE, top/bottom Y | hero | 3, 4 |
| SIDEPOD | HUGE, mirrored, wide | hero | 5, 6 |
| HOOD | LARGE, top Y | hero | 8 |
| REAR_SECTION | LARGE, bottom Y | secondary | 7 |
| SIDE_SKIRT | MEDIUM, mirrored, horizontal | accent | 9, 10 |
| FENDER | MEDIUM, mirrored, wide | secondary | 12, 13 |
| BRAKE_DUCT | SMALL, mirrored, horizontal | accent | 15, 16, 19, 20, 24, 25 |
| MUDGUARD | SMALL, mirrored, square | darken | 17, 18, 26, 27 |
| PILLAR | MEDIUM, vertical | neutral | 11, 14 |
| NOSE_DETAIL | SMALL, edge Y | accent | 21, 22, 23 |

---

## 2. Color Role System

### Roles

| Role | Description | Typical Treatment |
|------|-------------|-------------------|
| **hero** | Primary attention areas | Base color + accent pattern |
| **secondary** | Supporting areas | Secondary/stripe color |
| **accent** | Highlight details | Accent color, can be bold |
| **darken** | Should recede | Darker than base, muted |
| **neutral** | Minimal attention | Dark/black/carbon |

### Automatic Color Assignment

```python
def get_island_color(island_id, palette):
    part = classify_island(island_id)
    role = part.color_role
    
    if role == 'hero':
        return blend(palette.base, palette.accent, 0.2)
    elif role == 'secondary':
        return palette.secondary
    elif role == 'accent':
        return palette.accent
    elif role == 'darken':
        return darken(palette.base, 0.4)
    else:  # neutral
        return (30, 30, 30)
```

---

## 3. Gradient Flow System

### The Problem

Current gradients are applied in UV space, which doesn't match the car's 3D flow.
A "front-to-back" gradient in UV might look random on the actual car.

### The Solution: Part-Aware Gradients

Each part has a **flow direction** that maps to the car's 3D form:

| Part | Flow Direction | 3D Meaning |
|------|---------------|------------|
| SIDEPOD | horizontal (in UV) | Front-to-back on car |
| NOSE_TAIL | horizontal | Nose tip to body |
| HOOD | vertical | Front to windshield |
| FENDER | diagonal | Wheel arch curve |
| MUDGUARD | radial | Around wheel |

### Implementation

```python
def apply_part_gradient(img, island_mask, part_type, color_start, color_end):
    direction = PART_FLOW_DIRECTIONS[part_type]
    
    if direction == 'horizontal':
        # Gradient left-to-right in island bbox
        gradient = create_horizontal_gradient(...)
    elif direction == 'radial':
        # Gradient from center outward
        gradient = create_radial_gradient(...)
    # etc.
    
    return blend_with_mask(img, gradient, island_mask)
```

---

## 4. Per-Part Material Properties

### Finish Zones

Different parts of the car should have different material finishes:

| Part | Finish | Alpha Value | Reasoning |
|------|--------|-------------|-----------|
| SIDEPOD | High gloss | 0x60-0x70 | Hero areas should shine |
| HOOD | High gloss | 0x60-0x70 | Prominent, reflective |
| COCKPIT | Matte | 0xA0-0xB0 | Reduce glare |
| MUDGUARD | Matte | 0xA0-0xC0 | Recessed, less attention |
| FENDER | Medium gloss | 0x80-0x90 | Balanced |

### Implementation

```python
def compute_finish_alpha(island_id):
    part = classify_island(island_id)
    finish = PART_FINISH_MAP.get(part.name, 'medium')
    
    if finish == 'high_gloss':
        return 0x68
    elif finish == 'matte':
        return 0xA5
    else:
        return 0x8E  # neutral
```

---

## 5. Design Rules Engine

### Coherence Rules

```python
DESIGN_RULES = {
    # Color flow rules
    'accent_must_touch_hero': True,  # Accent colors should connect to hero areas
    'secondary_bridges_hero_neutral': True,  # Secondary fills gaps
    
    # Contrast rules
    'hero_accent_contrast_min': 0.3,  # Minimum luminance difference
    'mudguard_darker_than_base': True,
    
    # Pattern rules
    'stripes_follow_body_lines': True,  # Stripes align with part edges
    'gradients_follow_flow': True,  # Gradients use part flow direction
    
    # Detail rules
    'preserve_detail_on_small_islands': True,  # Don't overwhelm tiny parts
    'logos_on_flat_areas_only': True,  # Place logos on sidepods, not curves
}
```

### Validation

```python
def validate_design(skin, palette):
    issues = []
    
    # Check accent connectivity
    if not accent_touches_hero(skin):
        issues.append('Accent color is isolated')
    
    # Check mudguard darkness
    if mudguard_brighter_than_base(skin, palette):
        issues.append('Mudguards should be darker')
    
    # Check contrast
    if hero_accent_contrast(skin) < 0.3:
        issues.append('Need more contrast between base and accent')
    
    return issues
```

---

## 6. Smart Skin Presets

### "Pro Racing" Preset

```python
PRO_RACING = {
    'hero': {
        'pattern': 'solid_with_accent_stripe',
        'accent_position': 'lower_third',  # Stripe at bottom
        'finish': 'high_gloss',
    },
    'secondary': {
        'pattern': 'gradient_to_base',
        'finish': 'medium_gloss',
    },
    'accent': {
        'pattern': 'solid_bright',
        'finish': 'high_gloss',
    },
    'mudguard': {
        'color': 'darken_base_40%',
        'finish': 'matte',
    },
}
```

### "Stealth" Preset

```python
STEALTH = {
    'hero': {
        'pattern': 'dark_gradient',
        'finish': 'matte',
    },
    'accent': {
        'pattern': 'minimal_highlight',
        'color': 'desaturate_50%',
    },
    'mudguard': {
        'color': 'near_black',
    },
}
```

---

## 7. Implementation Plan

### Phase 1: Part Classification (Foundation)
- [ ] Implement `_classify_uv_island()` function
- [ ] Build `CarGeometry` class holding all island metadata
- [ ] Add `--spatial-debug` mode to visualize classifications
- [ ] Validate against known Stadium car parts

### Phase 2: Color Role System
- [ ] Implement automatic color assignment per role
- [ ] Add `--smart-colors` flag to use spatial-aware coloring
- [ ] Per-part color overrides via CLI

### Phase 3: Gradient Flow
- [ ] Define flow directions for each part type
- [ ] Implement part-aware gradient rendering
- [ ] Add `--gradient-flow` option

### Phase 4: Per-Part Finish
- [ ] Implement finish zones in alpha channel
- [ ] Add `--finish-zones` flag
- [ ] Allow per-part finish overrides

### Phase 5: Design Rules & Presets
- [ ] Implement validation engine
- [ ] Create preset library
- [ ] Add `--preset pro_racing` style shortcuts

---

## 8. Example Output

### Before (Current)
```
Base color: applied uniformly
Accent: placed by user-specified regions
Mudguards: whatever the base had
Finish: uniform across entire car
```

### After (Spatial-Aware)
```
Base color: applied to hero areas
Accent: automatically placed on accent parts (side skirts, brake ducts)
Secondary: fills fenders, rear section
Mudguards: automatically darkened
Finish: glossy on sidepods, matte on cockpit, medium elsewhere
Gradients: flow front-to-back on sidepods
```

---

## 9. Technical Architecture

```
                    +------------------+
                    |  UV Island Data  |
                    |  (JSON atlas)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Part Classifier  |
                    | (geometry rules) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   CarGeometry    |
                    |   (semantic map) |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
+--------v-------+  +--------v-------+  +--------v-------+
| Color Assigner |  | Gradient Flow  |  | Finish Zones   |
| (per-role)     |  | (per-part dir) |  | (per-part mat) |
+--------+-------+  +--------+-------+  +--------+-------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                    +--------v---------+
                    |   Skin Output    |
                    +------------------+
```

---

## 10. Benefits

1. **Automatic good design**: Colors end up in the right places
2. **Consistency**: All skins follow the same spatial rules
3. **Speed**: No manual region specification needed
4. **Pro look**: Gradients, finishes, and colors all work together
5. **Extensibility**: Easy to add new presets and rules
6. **Pack-agnostic**: Works for any Stadium-compatible UV layout

---

## References

- UV Atlas: `out/uv_atlas/standard_stadium_islands_2048.json`
- Current implementation: `generate_tmnf_skin.py`
- Mudguard detection: `_detect_mudguard_island_ids()`
