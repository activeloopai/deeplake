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
}

# these configs are added to every `htype`
COMMON_CONFIGS = {"chunk_size": DEFAULT_CHUNK_SIZE, "custom_meta": {}}

for config in HTYPE_CONFIGURATIONS.values():
    config.update(COMMON_CONFIGS)
