from PIL import Image  # type: ignore


BYTE_COMPRESSIONS = [
    "lz4",
]


IMAGE_COMPRESSIONS = [
    "bmp",
    "dib",
    "pcx",
    "gif",
    "png",
    "jpeg2000",
    "ico",
    "tiff",
    "jpeg",
    "ppm",
    "sgi",
    "tga",
    "webp",
    "wmf",
    "xbm",
]


# Pillow plugins for some formats might not be installed:
if not Image.SAVE:
    Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
]
SUPPORTED_COMPRESSIONS.append(None)  # type: ignore
SUPPORTED_COMPRESSIONS = list(set(SUPPORTED_COMPRESSIONS))  # type: ignore

COMPRESSION_ALIASES = {"jpg": "jpeg"}

# If `True`  compression format has to be the same between samples in the same tensor.
# If `False` compression format can   be different between samples in the same tensor.
USE_UNIFORM_COMPRESSION_PER_SAMPLE = True
