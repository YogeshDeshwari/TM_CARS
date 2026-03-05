from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def iter_mip_sizes(width: int, height: int) -> Iterable[Tuple[int, int]]:
    w, h = width, height
    yield (w, h)
    while w > 1 or h > 1:
        w = max(1, w // 2)
        h = max(1, h // 2)
        yield (w, h)


def generate_mipmaps(img: Image.Image) -> List[Image.Image]:
    """Downscale with BOX filter (good for mipmaps)."""
    img = img.convert("RGBA")
    levels = [img]
    for (w, h) in list(iter_mip_sizes(*img.size))[1:]:
        levels.append(levels[-1].resize((w, h), Image.Resampling.BOX))
    return levels


def rgba_to_bgra_bytes(img: Image.Image) -> bytes:
    """DDS A8R8G8B8 masks correspond to little-endian BGRA byte order."""
    rgba = img.convert("RGBA")
    return rgba.tobytes("raw", "BGRA")


def build_dds_rgba8_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    """Build an uncompressed DDS (RGBA8) as bytes with optional mipmaps."""
    base = img.convert("RGBA")
    width, height = base.size

    if mipmaps:
        levels = generate_mipmaps(base)
        mip_count = len(levels)
    else:
        levels = [base]
        mip_count = 0

    DDS_MAGIC = b"DDS "

    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000

    DDPF_ALPHAPIXELS = 0x1
    DDPF_RGB = 0x40

    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_PITCH
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT
    dwHeight = height
    dwWidth = width
    dwPitchOrLinearSize = width * 4
    dwDepth = 0
    dwMipMapCount = mip_count
    dwReserved1 = (0,) * 11

    # Pixel format: RGBA8 (A8R8G8B8 masks)
    pfSize = 32
    pfFlags = DDPF_RGB | DDPF_ALPHAPIXELS
    pfFourCC = 0
    pfRGBBitCount = 32
    pfRBitMask = 0x00FF0000
    pfGBitMask = 0x0000FF00
    pfBBitMask = 0x000000FF
    pfABitMask = 0xFF000000

    dwCaps = DDSCAPS_TEXTURE
    if mipmaps:
        dwCaps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        pfSize,
        pfFlags,
        pfFourCC,
        pfRGBBitCount,
        pfRBitMask,
        pfGBitMask,
        pfBBitMask,
        pfABitMask,
        dwCaps,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    if len(header) != 124:
        raise RuntimeError(f"Internal error: DDS header length is {len(header)}, expected 124.")

    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(rgba_to_bgra_bytes(level))
    return b"".join(chunks)


def _rgb_to_565(r: int, g: int, b: int) -> int:
    r5 = (r * 31 + 127) // 255
    g6 = (g * 63 + 127) // 255
    b5 = (b * 31 + 127) // 255
    return (r5 << 11) | (g6 << 5) | b5


def _rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (r, g, b)


def _compress_dxt1_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    if len(pixels) != 16:
        raise ValueError("DXT1 block must have exactly 16 pixels.")

    best_min = pixels[0]
    best_max = pixels[0]
    lum_min = 77 * best_min[0] + 150 * best_min[1] + 29 * best_min[2]
    lum_max = lum_min
    for p in pixels[1:]:
        lum = 77 * p[0] + 150 * p[1] + 29 * p[2]
        if lum < lum_min:
            lum_min = lum
            best_min = p
        elif lum > lum_max:
            lum_max = lum
            best_max = p

    c0 = _rgb_to_565(best_max[0], best_max[1], best_max[2])
    c1 = _rgb_to_565(best_min[0], best_min[1], best_min[2])

    if c0 == c1:
        c1 = c0 - 1 if c0 > 0 else 1

    if c0 < c1:
        c0, c1 = c1, c0

    r0, g0, b0 = _rgb565_to_rgb888(c0)
    r1, g1, b1 = _rgb565_to_rgb888(c1)
    palette = [
        (r0, g0, b0),
        (r1, g1, b1),
        ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3),
        ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3),
    ]

    bits = 0
    for i, p in enumerate(pixels):
        pr, pg, pb = p[0], p[1], p[2]
        best_idx = 0
        best_err = 10**18
        for idx, (cr, cg, cb) in enumerate(palette):
            dr = pr - cr
            dg = pg - cg
            db = pb - cb
            err = dr * dr + dg * dg + db * db
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        bits |= (best_idx & 0x3) << (2 * i)

    return struct.pack("<HHI", c0, c1, bits)


def _compress_image_to_dxt1(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4

    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt1_block(block))
    return bytes(out)


def _compress_dxt3_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    if len(pixels) != 16:
        raise ValueError("DXT3 block must have exactly 16 pixels.")

    alpha_bits = 0
    for i, p in enumerate(pixels):
        a = int(p[3])
        a4 = (a * 15 + 127) // 255
        alpha_bits |= (a4 & 0xF) << (4 * i)
    alpha_bytes = alpha_bits.to_bytes(8, "little")

    color_bytes = _compress_dxt1_block(pixels)
    return alpha_bytes + color_bytes


def _compress_image_to_dxt3(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4

    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt3_block(block))
    return bytes(out)


def _compress_dxt5_block(pixels: Sequence[Tuple[int, int, int, int]]) -> bytes:
    if len(pixels) != 16:
        raise ValueError("DXT5 block must have exactly 16 pixels.")

    alphas = [p[3] for p in pixels]
    a0 = max(alphas)
    a1 = min(alphas)

    if a0 > a1:
        alpha_palette = [
            a0,
            a1,
            (6 * a0 + 1 * a1) // 7,
            (5 * a0 + 2 * a1) // 7,
            (4 * a0 + 3 * a1) // 7,
            (3 * a0 + 4 * a1) // 7,
            (2 * a0 + 5 * a1) // 7,
            (1 * a0 + 6 * a1) // 7,
        ]
    else:
        alpha_palette = [
            a0,
            a1,
            (4 * a0 + 1 * a1) // 5 if a0 != a1 else a0,
            (3 * a0 + 2 * a1) // 5 if a0 != a1 else a0,
            (2 * a0 + 3 * a1) // 5 if a0 != a1 else a0,
            (1 * a0 + 4 * a1) // 5 if a0 != a1 else a0,
            0,
            255,
        ]

    alpha_bits = 0
    for i, a in enumerate(alphas):
        best_idx = 0
        best_err = 10**9
        for idx, pa in enumerate(alpha_palette):
            err = abs(a - pa)
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        alpha_bits |= (best_idx & 0x7) << (3 * i)

    alpha_bytes = bytes((a0, a1)) + alpha_bits.to_bytes(6, "little")

    # Color block (DXT1-style, always 4-color mode)
    best_min = pixels[0]
    best_max = pixels[0]
    lum_min = 77 * best_min[0] + 150 * best_min[1] + 29 * best_min[2]
    lum_max = lum_min
    for p in pixels[1:]:
        lum = 77 * p[0] + 150 * p[1] + 29 * p[2]
        if lum < lum_min:
            lum_min = lum
            best_min = p
        elif lum > lum_max:
            lum_max = lum
            best_max = p

    c0 = _rgb_to_565(best_max[0], best_max[1], best_max[2])
    c1 = _rgb_to_565(best_min[0], best_min[1], best_min[2])

    if c0 == c1:
        c1 = c0 - 1 if c0 > 0 else 1
    if c0 < c1:
        c0, c1 = c1, c0

    r0, g0, b0 = _rgb565_to_rgb888(c0)
    r1, g1, b1 = _rgb565_to_rgb888(c1)

    palette = [
        (r0, g0, b0),
        (r1, g1, b1),
        ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3),
        ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3),
    ]

    color_bits = 0
    for i, p in enumerate(pixels):
        pr, pg, pb = p[0], p[1], p[2]
        best_idx = 0
        best_err = 10**18
        for idx, (cr, cg, cb) in enumerate(palette):
            dr = pr - cr
            dg = pg - cg
            db = pb - cb
            err = dr * dr + dg * dg + db * db
            if err < best_err:
                best_err = err
                best_idx = idx
                if err == 0:
                    break
        color_bits |= (best_idx & 0x3) << (2 * i)

    color_bytes = struct.pack("<HHI", c0, c1, color_bits)
    return alpha_bytes + color_bytes


def _compress_image_to_dxt5(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    w, h = img.size
    data = img.tobytes()
    out = bytearray()

    bw = (w + 3) // 4
    bh = (h + 3) // 4
    for by in range(bh):
        y0 = by * 4
        for bx in range(bw):
            x0 = bx * 4
            block: List[Tuple[int, int, int, int]] = []
            for dy in range(4):
                y = y0 + dy
                if y >= h:
                    y = h - 1
                row = (y * w) * 4
                for dx in range(4):
                    x = x0 + dx
                    if x >= w:
                        x = w - 1
                    i = row + x * 4
                    block.append((data[i], data[i + 1], data[i + 2], data[i + 3]))
            out.extend(_compress_dxt5_block(block))
    return bytes(out)


def build_dds_dxt1_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    base = img.convert("RGBA")
    width, height = base.size
    levels = generate_mipmaps(base) if mipmaps else [base]
    mip_count = len(levels) if mipmaps else 0

    DDS_MAGIC = b"DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDPF_FOURCC = 0x4
    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 8
    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        height,
        width,
        top_linear,
        0,
        mip_count,
        *([0] * 11),
        32,
        DDPF_FOURCC,
        struct.unpack("<I", b"DXT1")[0],
        0,
        0,
        0,
        0,
        0,
        (DDSCAPS_TEXTURE | (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP if mipmaps else 0)),
        0,
        0,
        0,
        0,
    )
    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt1(level))
    return b"".join(chunks)


def build_dds_dxt3_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    base = img.convert("RGBA")
    width, height = base.size
    levels = generate_mipmaps(base) if mipmaps else [base]
    mip_count = len(levels) if mipmaps else 0

    DDS_MAGIC = b"DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDPF_FOURCC = 0x4
    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 16
    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        height,
        width,
        top_linear,
        0,
        mip_count,
        *([0] * 11),
        32,
        DDPF_FOURCC,
        struct.unpack("<I", b"DXT3")[0],
        0,
        0,
        0,
        0,
        0,
        (DDSCAPS_TEXTURE | (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP if mipmaps else 0)),
        0,
        0,
        0,
        0,
    )
    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt3(level))
    return b"".join(chunks)


def build_dds_dxt5_bytes(img: Image.Image, *, mipmaps: bool = True) -> bytes:
    base = img.convert("RGBA")
    width, height = base.size
    levels = generate_mipmaps(base) if mipmaps else [base]
    mip_count = len(levels) if mipmaps else 0

    DDS_MAGIC = b"DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_LINEARSIZE = 0x80000
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDPF_FOURCC = 0x4
    DDSCAPS_COMPLEX = 0x8
    DDSCAPS_MIPMAP = 0x400000
    DDSCAPS_TEXTURE = 0x1000

    top_linear = ((width + 3) // 4) * ((height + 3) // 4) * 16
    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    if mipmaps:
        dwFlags |= DDSD_MIPMAPCOUNT

    header = struct.pack(
        "<I I I I I I I 11I 8I 5I",
        dwSize,
        dwFlags,
        height,
        width,
        top_linear,
        0,
        mip_count,
        *([0] * 11),
        32,
        DDPF_FOURCC,
        struct.unpack("<I", b"DXT5")[0],
        0,
        0,
        0,
        0,
        0,
        (DDSCAPS_TEXTURE | (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP if mipmaps else 0)),
        0,
        0,
        0,
        0,
    )
    chunks = [DDS_MAGIC, header]
    for level in levels:
        chunks.append(_compress_image_to_dxt5(level))
    return b"".join(chunks)


def save_dds_dxt1(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_dds_dxt1_bytes(img, mipmaps=mipmaps))


def save_dds_dxt3(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_dds_dxt3_bytes(img, mipmaps=mipmaps))


def save_dds_dxt5(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_dds_dxt5_bytes(img, mipmaps=mipmaps))


def save_dds_rgba8(out_path: Path, img: Image.Image, *, mipmaps: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_dds_rgba8_bytes(img, mipmaps=mipmaps))


def read_dds_dimensions_from_bytes(buf: bytes) -> Tuple[int, int]:
    if len(buf) < 128:
        raise ValueError("DDS buffer too small to contain header.")
    if buf[:4] != b"DDS ":
        raise ValueError("Not a DDS file (missing magic).")
    hdr = buf[4 : 4 + 124]
    dwHeight = struct.unpack("<I", hdr[8:12])[0]
    dwWidth = struct.unpack("<I", hdr[12:16])[0]
    return (dwWidth, dwHeight)


def read_dds_fourcc_from_bytes(buf: bytes) -> Optional[str]:
    if len(buf) < 128 or buf[:4] != b"DDS ":
        return None
    hdr = buf[4 : 4 + 124]
    pf = hdr[72 : 72 + 32]
    pfFlags = struct.unpack("<I", pf[4:8])[0]
    fourCC = pf[8:12]
    if (pfFlags & 0x4) == 0:
        return None
    try:
        return fourCC.decode("ascii", errors="replace")
    except Exception:
        return None


def read_dds_mipmap_count_from_bytes(buf: bytes) -> int:
    """
    Return declared mipmap count from DDS header.

    If the DDSD_MIPMAPCOUNT flag is not set, returns 1 (top level only).
    """
    if len(buf) < 128 or buf[:4] != b"DDS ":
        return 1
    hdr = buf[4 : 4 + 124]
    dwFlags = struct.unpack("<I", hdr[4:8])[0]
    dwMipMapCount = struct.unpack("<I", hdr[24:28])[0]
    DDSD_MIPMAPCOUNT = 0x20000
    if (dwFlags & DDSD_MIPMAPCOUNT) == 0:
        return 1
    try:
        return max(1, int(dwMipMapCount))
    except Exception:
        return 1

