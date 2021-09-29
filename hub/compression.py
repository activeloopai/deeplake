"""
Image and Byte compression types.

Supported image compressions (formats):

 * BMP
 * DIB
 * GIF
 * ICO
 * JPEG
 * JPEG 2000
 * PCX
 * PNG
 * PPM
 * SGI
 * TGA
 * TIFF
 * WEBP
 * WMF
 * XBM

Supported byte compression:

 * LZ4
"""
from PIL import Image  # type: ignore


BYTE_COMPRESSIONS = [
    "lz4",
]


IMAGE_COMPRESSIONS = [
    "bmp",
    "dib",
    "gif",
    "ico",
    "jpeg",
    "jpeg2000",
    "pcx",
    "png",
    "ppm",
    "sgi",
    "tga",
    "tiff",
    "webp",
    "wmf",
    "xbm",
]


BYTE_COMPRESSION = "byte"
IMAGE_COMPRESSION = "image"
COMPRESSION_TYPES = [BYTE_COMPRESSION, IMAGE_COMPRESSION]


# Pillow plugins for some formats might not be installed:
Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
]
SUPPORTED_COMPRESSIONS = list(sorted(set(SUPPORTED_COMPRESSIONS)))  # type: ignore
SUPPORTED_COMPRESSIONS.append(None)  # type: ignore

COMPRESSION_ALIASES = {"jpg": "jpeg"}

# If `True`  compression format has to be the same between samples in the same tensor.
# If `False` compression format can   be different between samples in the same tensor.
USE_UNIFORM_COMPRESSION_PER_SAMPLE = True


_compression_types = {}
for c in IMAGE_COMPRESSIONS:
    _compression_types[c] = IMAGE_COMPRESSION
for c in BYTE_COMPRESSIONS:
    _compression_types[c] = BYTE_COMPRESSION


def get_compression_type(c):
    return _compression_types[c]
