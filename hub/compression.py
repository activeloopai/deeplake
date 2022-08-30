import itertools


"""
Supported compressions (formats):

    Image : bmp, dib, gif, ico, jpeg, jpeg2000, pcx, png, ppm, sgi, tga, tiff, webp, wmf, xbm
    Audio : flac, mp3, wav
    Video : mp4, mkv, avi
    Bytes : lz4

__Note__:- 

For video compressions, we only support already compressed data read using hub.read. We do not actually compress the video data. 

Also, when using hub.read with one of the video compressions, ensure that the compression matches, otherwise hub will be unable to compress the data to the specified compression.

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

IMAGE_COMPRESSION_EXT_DICT = {
    "apng": [".png"],
    "bmp": [".bmp"],
    "dib": [".dib"],
    "gif": [".gif"],
    "ico": [".ico"],
    "jpeg": [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp"],
    "jpeg2000": [
        ".jp2",
        ".j2k",
        ".jpf",
        ".jpm",
        ".jpg2",
        ".j2c",
        ".jpc",
        ".jpx",
        ".mj2",
    ],
    "pcx": [".pcx"],
    "png": [".png"],
    "ppm": [".pbm", ".pgm", ".ppm", ".pnm"],
    "sgi": [".sgi"],
    "tga": [".tga"],
    "tiff": [".tiff", ".tif"],
    "webp": [".webp"],
    "wmf": [".wmf"],
    "xbm": [".xbm"],
}


IMAGE_COMPRESSION_EXTENSIONS = list(
    set(itertools.chain(*IMAGE_COMPRESSION_EXT_DICT.values()))
)

VIDEO_COMPRESSIONS = ["mp4", "mkv", "avi"]

AUDIO_COMPRESSIONS = ["mp3", "flac", "wav"]

POINT_CLOUD_COMPRESSIONS = ["las"]


# Just constants
BYTE_COMPRESSION = "byte"
IMAGE_COMPRESSION = "image"
VIDEO_COMPRESSION = "video"
AUDIO_COMPRESSION = "audio"
POINT_CLOUD_COMPRESSION = "point_cloud"


COMPRESSION_TYPES = [
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    AUDIO_COMPRESSION,
    VIDEO_COMPRESSION,
    POINT_CLOUD_COMPRESSION,
]

# Pillow plugins for some formats might not be installed:
Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]


IMAGE_COMPRESSIONS.insert(0, "apng")
IMAGE_COMPRESSIONS.insert(2, "dcm")

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS,
    *POINT_CLOUD_COMPRESSIONS,
]
SUPPORTED_COMPRESSIONS = list(sorted(set(SUPPORTED_COMPRESSIONS)))  # type: ignore
SUPPORTED_COMPRESSIONS.append(None)  # type: ignore

COMPRESSION_ALIASES = {"jpg": "jpeg", "tif": "tiff", "jp2": "jpeg2000"}

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
for c in POINT_CLOUD_COMPRESSIONS:
    _compression_types[c] = POINT_CLOUD_COMPRESSION


def get_compression_type(c):
    """Returns the compression type for the given compression name."""
    if c is None:
        return None
    ret = _compression_types.get(c)
    if ret is None and c.upper() in Image.OPEN:
        ret = IMAGE_COMPRESSION
    if ret is None:
        raise KeyError(c)
    return ret
