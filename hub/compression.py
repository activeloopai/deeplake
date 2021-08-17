from PIL import Image  # type: ignore


SAMPLE_COMPRESSIONS = [
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

CHUNK_COMPRESSIONS = [
    "lz4",
]


# Pillow plugins for some formats might not be installed:
if not Image.SAVE:
    Image.init()
SUPPORTED_COMPRESSIONS = [
    c
    for c in SAMPLE_COMPRESSIONS
    if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]

SUPPORTED_COMPRESSIONS = SAMPLE_COMPRESSIONS + CHUNK_COMPRESSIONS
SUPPORTED_COMPRESSIONS.append(None)  # type: ignore

COMPRESSION_ALIASES = {"jpg": "jpeg"}

# If `True`  compression format has to be the same between samples in the same tensor.
# If `False` compression format can   be different between samples in the same tensor.
USE_UNIFORM_COMPRESSION_PER_SAMPLE = True
