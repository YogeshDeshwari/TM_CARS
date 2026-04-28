"""
DDS writer for Trackmania 2020 skins.

Uses quicktex (C++ backend) for high-quality BC1/BC3/BC4/BC5 block compression,
with hand-written D3D9-compatible DDS headers (TM2020 requires D3D9 presets,
not DX10 extended).

Texture format reference (from Nadeo):
    Skin_B      BC1   Basecolor RGB
    Skin_R      BC5   R=Roughness, G=Metalness
    Skin_CoatR  BC4   Clearcoat grayscale
    Skin_DirtMask BC4 Dirt mask grayscale
    Details_B   BC1   Basecolor RGB
    Details_R   BC5   Roughness/Metalness
    Details_I   BC3   Self-illumination RGBA
    Details_N   BC5   Normal map (OpenGL Y+)
    Details_DirtMask BC4
    Wheels_B    BC1
    Wheels_R    BC5
    Wheels_N    BC5
    Wheels_DirtMask BC4
    Glass_D     BC1
    Glass_I     BC5
"""

from __future__ import annotations

import os
import struct
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "1")

from quicktex import RawTexture
from quicktex.s3tc.bc1 import BC1Encoder
from quicktex.s3tc.bc3 import BC3Encoder
from quicktex.s3tc.bc4 import BC4Encoder
from quicktex.s3tc.bc5 import BC5Encoder


class BCFormat(Enum):
    BC1 = "BC1"
    BC3 = "BC3"
    BC4 = "BC4"
    BC5 = "BC5"


# FourCC codes for D3D9-style DDS headers
_FOURCC = {
    BCFormat.BC1: b"DXT1",
    BCFormat.BC3: b"DXT5",
    BCFormat.BC4: b"ATI1",
    BCFormat.BC5: b"ATI2",
}

# Bytes per 4x4 block
_BLOCK_SIZE = {
    BCFormat.BC1: 8,
    BCFormat.BC3: 16,
    BCFormat.BC4: 8,
    BCFormat.BC5: 16,
}

# Singleton encoders (reused across calls)
_enc_bc1 = BC1Encoder()
_enc_bc3 = BC3Encoder()
_enc_bc4_r = BC4Encoder(0)
_enc_bc4_g = BC4Encoder(1)
_enc_bc5 = BC5Encoder(0, 1)


def _pil_to_raw(img: Image.Image) -> RawTexture:
    rgba = img.convert("RGBA")
    return RawTexture.frombytes(rgba.tobytes("raw", "RGBA"), *rgba.size)


def _compress_level(img: Image.Image, fmt: BCFormat) -> bytes:
    raw = _pil_to_raw(img)
    if fmt == BCFormat.BC1:
        return bytes(_enc_bc1.encode(raw))
    elif fmt == BCFormat.BC3:
        return bytes(_enc_bc3.encode(raw))
    elif fmt == BCFormat.BC4:
        return bytes(_enc_bc4_r.encode(raw))
    elif fmt == BCFormat.BC5:
        return bytes(_enc_bc5.encode(raw))
    raise ValueError(f"Unknown format: {fmt}")


def _generate_mip_chain(img: Image.Image) -> List[Image.Image]:
    """Generate full mipmap chain down to 1x1."""
    img = img.convert("RGBA")
    levels = [img]
    w, h = img.size
    while w > 1 or h > 1:
        w = max(1, w // 2)
        h = max(1, h // 2)
        levels.append(img.resize((w, h), Image.Resampling.BOX))
    return levels


def _build_dds_header(
    width: int,
    height: int,
    fmt: BCFormat,
    mip_count: int,
) -> bytes:
    """Build a 128-byte DDS file header (magic + 124-byte header struct)."""
    DDSD_CAPS        = 0x1
    DDSD_HEIGHT      = 0x2
    DDSD_WIDTH       = 0x4
    DDSD_LINEARSIZE  = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDPF_FOURCC      = 0x4
    DDSCAPS_COMPLEX  = 0x8
    DDSCAPS_MIPMAP   = 0x400000
    DDSCAPS_TEXTURE  = 0x1000

    block_size = _BLOCK_SIZE[fmt]
    linear_size = ((width + 3) // 4) * ((height + 3) // 4) * block_size

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    caps = DDSCAPS_TEXTURE
    if mip_count > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP

    fourcc_int = struct.unpack("<I", _FOURCC[fmt])[0]

    header = struct.pack(
        "<4s I I I I I I I 11I I I I I I I I I I I I I I",
        b"DDS ",
        124,              # dwSize
        flags,
        height,
        width,
        linear_size,
        0,                # dwDepth
        mip_count,
        *([0] * 11),      # dwReserved1[11]
        32,               # ddspf.dwSize
        DDPF_FOURCC,     # ddspf.dwFlags
        fourcc_int,       # ddspf.dwFourCC
        0,                # ddspf.dwRGBBitCount
        0, 0, 0, 0,      # ddspf.dwR/G/B/ABitMask
        caps,             # dwCaps
        0, 0, 0,          # dwCaps2/3/4
        0,                # dwReserved2
    )
    assert len(header) == 128, f"Header size {len(header)}, expected 128"
    return header


def build_dds_bytes(
    img: Image.Image,
    fmt: BCFormat,
    *,
    mipmaps: bool = True,
) -> bytes:
    """
    Compress a PIL Image to DDS bytes in the given BC format.

    Returns the complete DDS file content (header + compressed mip chain).
    """
    base = img.convert("RGBA")
    levels = _generate_mip_chain(base) if mipmaps else [base]
    mip_count = len(levels) if mipmaps else 1

    header = _build_dds_header(base.size[0], base.size[1], fmt, mip_count)
    chunks = [header]
    for level in levels:
        chunks.append(_compress_level(level, fmt))
    return b"".join(chunks)


def save_dds(
    path: Path,
    img: Image.Image,
    fmt: BCFormat,
    *,
    mipmaps: bool = True,
) -> None:
    """Save a PIL Image as a DDS file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_dds_bytes(img, fmt, mipmaps=mipmaps))


# Convenience functions matching TM2020 texture naming
def save_basecolor(path: Path, img: Image.Image, **kw) -> None:
    """Save a BC1 basecolor texture (_B / _D)."""
    save_dds(path, img, BCFormat.BC1, **kw)


def save_roughness_metalness(path: Path, img: Image.Image, **kw) -> None:
    """Save a BC5 roughness/metalness texture (_R). R=rough, G=metal."""
    save_dds(path, img, BCFormat.BC5, **kw)


def save_grayscale(path: Path, img: Image.Image, **kw) -> None:
    """Save a BC4 single-channel texture (_CoatR / _DirtMask)."""
    save_dds(path, img, BCFormat.BC4, **kw)


def save_illumination(path: Path, img: Image.Image, **kw) -> None:
    """Save a BC3 self-illumination texture (_I). RGBA."""
    save_dds(path, img, BCFormat.BC3, **kw)


def save_normal(path: Path, img: Image.Image, **kw) -> None:
    """Save a BC5 normal map (_N). OpenGL Y+ convention."""
    save_dds(path, img, BCFormat.BC5, **kw)
