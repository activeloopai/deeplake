from re import L
from typing import Dict
from hub.constants import (
    DEFAULT_CHUNK_COMPRESSION,
    DEFAULT_HTYPE,
    DEFAULT_SAMPLE_COMPRESSION,
    UNCOMPRESSED,
    DEFAULT_CHUNK_MIN_TARGET,
)

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {
        "dtype": "uint8",
        "sample_compression": "png",
        "chunk_compression": UNCOMPRESSED,
    },
    "class_label": {
        "dtype": "int32",
        "class_names": [],
    },
    "bbox": {"dtype": "float32"},
    "video": {"dtype": "uint8"},
    "binary_mask": {
        "dtype": "bool"
    },  # TODO: pack numpy arrays to store bools as 1 bit instead of 1 byte
    "segment_mask": {"dtype": "int32"},
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "chunk_size": DEFAULT_CHUNK_MIN_TARGET,
    "custom_meta": {},
    "chunk_compression": DEFAULT_CHUNK_COMPRESSION,
    "sample_compression": DEFAULT_SAMPLE_COMPRESSION,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v
