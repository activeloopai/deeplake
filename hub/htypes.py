from typing import Dict
from hub.constants import DEFAULT_CHUNK_SIZE

DEFAULT_HTYPE = "generic"

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {"dtype": "uint8"},
    "class_label": {
        "dtype": "uint32",
        "class_names": [],
    },
    "bbox": {"dtype": "float32"},
    "video": {"dtype": "uint8"},
    "binary_mask": {
        "dtype": "bool"
    },  # TODO: pack numpy arrays to store bools as 1 bit instead of 1 byte
    "segment_mask": {"dtype": "uint32"},
}

# these configs are added to every `htype`
COMMON_CONFIGS = {"chunk_size": DEFAULT_CHUNK_SIZE, "custom_meta": {}}

for config in HTYPE_CONFIGURATIONS.values():
    config.update(COMMON_CONFIGS)
