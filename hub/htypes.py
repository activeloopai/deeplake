from typing import Dict
from hub.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION


DEFAULT_HTYPE = "generic"
DEFAULT_DTYPE = "float64"

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": DEFAULT_DTYPE},
    "image": {"dtype": "uint8"},
    "class_label": {
        "dtype": "uint32",
        "class_names": [],
    },
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "custom_meta": {},
    "compression": DEFAULT_COMPRESSION,
}

for config in HTYPE_CONFIGURATIONS.values():
    config.update(COMMON_CONFIGS)
