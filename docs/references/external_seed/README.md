# External seed corpus

Community-made TMNF Stadium skins, restored from commit `0eeb862`
(originally lived under `examples/`).  Hand-curated by the project
owner -- treat as **permanent** corpus members.  Never auto-deleted
by the corpus refresh job; never overwritten by scraped entries.

## Layout

```
external_seed/
  <skin_basename>/
    skin.zip          # the original TMNF skin pack (Diffuse, Details, etc.)
    origin.txt        # the original path inside the old examples/ dir
    meta.json         # populated by the corpus bootstrap (palette, pHash, ...)
    angle_*.png       # populated by the corpus bootstrap (4 canonical angles)
```

## Provenance

Authors on the skin filenames where given (`_by_MINA_TM`, `_by_SparkyTM`,
`_by_WiiTRO`).  These are not our work.  They are reference quality bars
for what TMNF players consider good (or, for the `not_car/` originals,
notable / canonical gimmicks).

Three groupings (visible in `origin.txt`):

- `examples/*.zip`          -- canonical community TMNF skins
- `examples/no_mudguard/`   -- TMNF skin built on a no-mudguard mesh
- `examples/not_car/`       -- gimmick "skins" that turn the car into
                                another object (bob-omb, minecart, ...).
                                Useful only as a creativity/extreme
                                reference, not as an aesthetic target.

## How the agent uses this folder

See `AGENT_PROMPT.md` -> "Reference Corpus".  In short:

- Bootstrap renders four canonical-angle previews from each `skin.zip`.
- CLIP embeddings + palette pHashes go into the global manifest.
- Originality (axis 7) is measured against this folder *plus* the
  auto-scraped `external/` folder, with `external_seed/` weighted 1.5x
  the same as `external/` (community-curated weight).
- Conviction calibration (axis 8) consults this folder when checking
  if a proposed concept is too close to existing reference work.
