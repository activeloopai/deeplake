from re import L
from typing import Dict
from hub.constants import (
    DEFAULT_CHUNK_COMPRESSION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HTYPE,
    DEFAULT_SAMPLE_COMPRESSION,
    UNCOMPRESSED,
)

# TODO: how do we allow the user to keep their data uncompressed?
HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {
        "dtype": "uint8",
        "sample_compression": "png",
        "chunk_compression": UNCOMPRESSED,
    },
    "class_label": {
        "dtype": "uint32",
        "class_names": [],
    },
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "custom_meta": {},
    "chunk_compression": DEFAULT_CHUNK_COMPRESSION,
    "sample_compression": DEFAULT_SAMPLE_COMPRESSION,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():

        # only update if not specified explicitly
        if key not in config:
            config[key] = v
