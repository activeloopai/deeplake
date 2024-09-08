import itertools
from PIL import Image  # type: ignore


BYTE_COMPRESSIONS = [
    "lz4",
]


IMAGE_COMPRESSIONS = [
    "bmp",
    "dib",
    "eps",
    "fli",
    "gif",
    "ico",
    "im",
    "jpeg",
    "jpeg2000",
    "msp",
    "mpo",
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
    # "apng": [".png"],
    "bmp": [".bmp"],
    "eps": [".eps"],
    "fli": [".fli"],
    "dib": [".dib"],
    "gif": [".gif"],
    "ico": [".ico"],
    "im": [".im"],
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
    "msp": [".msp"],
    "mpo": [".mpo"],
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
NIFTI_COMPRESSIONS = ["nii", "nii.gz"]
POINT_CLOUD_COMPRESSIONS = ["las"]
MESH_COMPRESSIONS = ["ply", "stl"]

READONLY_COMPRESSIONS = [
    "mpo",
    "fli",
    "dcm",
    *NIFTI_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS,
]


# Just constants
BYTE_COMPRESSION = "byte"
IMAGE_COMPRESSION = "image"
VIDEO_COMPRESSION = "video"
AUDIO_COMPRESSION = "audio"
POINT_CLOUD_COMPRESSION = "point_cloud"
MESH_COMPRESSION = "mesh"
NIFTI_COMPRESSION = "nifti"


COMPRESSION_TYPES = [
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    AUDIO_COMPRESSION,
    VIDEO_COMPRESSION,
    POINT_CLOUD_COMPRESSION,
    MESH_COMPRESSION,
    NIFTI_COMPRESSION,
]

# Pillow plugins for some formats might not be installed:
Image.init()
IMAGE_COMPRESSIONS = [
    c for c in IMAGE_COMPRESSIONS if c.upper() in Image.SAVE and c.upper() in Image.OPEN
]


# IMAGE_COMPRESSIONS.insert(0, "apng")
IMAGE_COMPRESSIONS.insert(1, "dcm")
IMAGE_COMPRESSIONS.insert(2, "mpo")
IMAGE_COMPRESSIONS.insert(3, "fli")

SUPPORTED_COMPRESSIONS = [
    *BYTE_COMPRESSIONS,
    *IMAGE_COMPRESSIONS,
    *AUDIO_COMPRESSIONS,
    *VIDEO_COMPRESSIONS,
    *POINT_CLOUD_COMPRESSIONS,
    *MESH_COMPRESSIONS,
    *NIFTI_COMPRESSIONS,
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
for c in MESH_COMPRESSIONS:
    _compression_types[c] = MESH_COMPRESSION
for c in NIFTI_COMPRESSIONS:
    _compression_types[c] = NIFTI_COMPRESSION


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


def is_readonly_compression(c):
    """Returns if the file exists in READONLY_COMPRESSOINS or not."""
    return c in READONLY_COMPRESSIONS
