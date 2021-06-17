from re import L
from typing import Dict
from hub.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION, DEFAULT_HTYPE

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {"dtype": "uint8", "default_compression": "PNG"},
    "class_label": {
        "dtype": "uint32",
        "class_names": [],
    },
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "custom_meta": {},
    "default_compression": DEFAULT_COMPRESSION,
}

for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():

        # only update if not specified explicitly
        if key not in config:
            config[key] = v
