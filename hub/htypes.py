"""
"htype" is the class of a tensor: image, bounding box, generic tensor, etc.

These are used when creating a new tensor as follows:
```
>>> ds.create_tensor(some_data, htype="image")
```

Specifying an htype allows the [activeloop platform](https://app.activeloop.ai/)
to know how to best visualize your tensor. 
They are also used to inform default compression modes and data types.
"""

from re import L
from typing import Dict
from hub.constants import (
    DEFAULT_HTYPE,
    REQUIRE_USER_SPECIFICATION,
)

HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {
        "dtype": "uint8",
        "sample_compression": REQUIRE_USER_SPECIFICATION,
    },
    "class_label": {
        "dtype": "uint32",
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
    "sample_compression": None,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v
