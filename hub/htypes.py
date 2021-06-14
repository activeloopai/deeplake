from hub.constants import DEFAULT_CHUNK_SIZE


DEFAULT_HTYPE = "generic"
DEFAULT_DTYPE = "float64"

HTYPE_CONFIGURATIONS = {
    DEFAULT_HTYPE: {"dtype": DEFAULT_DTYPE, "chunk_size": DEFAULT_CHUNK_SIZE},
    "image": {"dtype": "uint8", "chunk_size": DEFAULT_CHUNK_SIZE},
    "class_label": {
        "dtype": "uint32",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "class_names": [],
    },
}
