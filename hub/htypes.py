"""
"htype" is the class of a tensor: image, bounding box, generic tensor, etc.

When not specified, the unspecified options will be inferred from the data:
```
>>> ds.create_tensor("my_tensor")
>>> ds.my_tensor.append(1)
>>> ds.my_tensor.dtype
int64
```

If you know beforehand, you can use htype at creation:
```
>>> ds.create_tensor("my_tensor", htype="image", sample_compression=None)
```

Specifying an htype allows for strict settings and error handling, and it is critical for increasing the performance of hub datasets containing rich data such as images and videos.

Supported htypes and their respective defaults are:

| HTYPE         |  DTYPE    |  COMPRESSION  |
| ------------  |  -------  |  -----------  |
| image         |  uint8    |  png          |
| class_label   |  uint32   |  none         |
| bbox          |  float32  |  none         |
| video         |  uint8    |  none         |
| binary_mask   |  bool     |  none         |
| segment_mask  |  int32    |  none         |

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
        "_info": ["class_names"],  # class_names should be stored in info, not meta
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
    "dtype": None,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v
