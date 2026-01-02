# Cavern Hunters - TMUF/TMNF Skin Generator

Custom Stadium car skin generator for **Cavern Hunters** team.

## Quick Start

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Generate a team skin
python3 generate_tmnf_skin.py \
  --name "CH_IceFade" \
  --base-zip "examples/KACKIEST-KACKY-10-SKIN-(dark-gray)_by_MINA_TM.zip" \
  --out out \
  --logo cavern_final_logo.png \
  --team-name "Cavern Hunters" \
  --tag "CH" \
  --style fade \
  --base-color "#0a1828ff" \
  --accent-color "#4fc3f7ff" \
  --stripe-color "#e1f5feff" \
  --wheel-color "#4fc3f7" \
  --nose-logo --proj-logo
```

## Generate player-specific skin (custom wing text)

```bash
python3 generate_tmnf_skin.py \
  --name "CH_PlayerName" \
  --base-zip "examples/KACKIEST-KACKY-10-SKIN-(dark-gray)_by_MINA_TM.zip" \
  --out out \
  --logo cavern_final_logo.png \
  --team-name "Cavern Hunters" \
  --tag "CH" \
  --wing-text "CH >> PlayerName" \
  --style fade_splatter \
  --base-color "#151618ff" \
  --accent-color "#4a4d52ff" \
  --stripe-color "#f0f2f5ff" \
  --wheel-color "#f0f2f5" \
  --nose-logo --proj-logo
```

## Available Styles

- `solid` - Clean single color
- `fade` - Gradient fade (Kacky-style dithered)
- `fade_splatter` - Fade + splatter texture overlay
- `splatter` - Splatter/halftone texture
- `fluid` - Fluid/marble swirls
- `topo` - Topographic contour lines
- `carbon` - Carbon fiber texture
- `galaxy` - Starfield/nebula effect
- `neon` - Neon glow lines

## Skin Sharing (TMUF)

For other players to see your skin in multiplayer, you need `.loc` files.

### Setup:
1. Download skins from `out/` folder
2. Download matching `.loc` files from `loc/` folder
3. Place both in: `Documents/TrackMania United/Skins/Vehicles/StadiumCar/`

The `.loc` file tells the game where to download the skin from when other players encounter you.

## Pre-made Skins

| Skin | Style | Download |
|------|-------|----------|
| CH_DarkGreyWhiteFade_Team | Fade + Splatter | [zip](out/CH_DarkGreyWhiteFade_Team.zip) |
| CH_IceFade_Frost | Ice Fade | [zip](out/CH_IceFade_Frost.zip) |
| CH_OceanTeal_Fluid | Ocean Fluid | [zip](out/CH_OceanTeal_Fluid.zip) |
| CH_EmberVolcanic_Topo | Volcanic Topo | [zip](out/CH_EmberVolcanic_Topo.zip) |
| CH_MidnightGold_Carbon | Purple/Gold Carbon | [zip](out/CH_MidnightGold_Carbon.zip) |

## Requirements

- Python 3.9+
- Pillow

```bash
pip install -r requirements.txt
```
