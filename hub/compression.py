"""
Supported compressions (formats):
    Image : bmp, dib, gif, ico, jpeg, jp2, pcx, png, ppm, sgi, tga, tiff, webp, wmf, xbm
    Audio : flac, mp3, wav
    Video : mp4, mkv, avi
    Bytes : lz4

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

VIDEO_COMPRESSIONS = ["mp4", "mkv", "avi"]

AUDIO_COMPRESSIONS = ["mp3", "flac", "wav"]


# Just constants
BYTE_COMPRESSION = "byte"
IMAGE_COMPRESSION = "image"
VIDEO_COMPRESSION = "video"
AUDIO_COMPRESSION = "audio"


COMPRESSION_TYPES = [BYTE_COMPRESSION, IMAGE_COMPRESSION, AUDIO_COMPRESSION]


COMPRESSION_TYPES = [
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    AUDIO_COMPRESSION,
    VIDEO_COMPRESSION,
]

# Pillow plugins for some formats might not be installed:
Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]

IMAGE_COMPRESSIONS.insert(0, "apng")

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS,
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
for c in VIDEO_COMPRESSIONS:
    _compression_types[c] = VIDEO_COMPRESSION
for c in AUDIO_COMPRESSIONS:
    _compression_types[c] = AUDIO_COMPRESSION


def get_compression_type(c):
    if c is None:
        return None
    ret = _compression_types.get(c)
    if ret is None and c.upper() in Image.OPEN:
        ret = IMAGE_COMPRESSION
    if ret is None:
        raise KeyError(c)
    return ret
