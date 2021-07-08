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

Specifying an htype allows for strict settings and errors whenever they are violated.
Also, tensors with htypes can be easily visualized on the [activeloop platform](https://app.activeloop.ai/).

Supported htypes are:

- "image"

- "class_label"

- "bbox"

- "video"

- "binary_mask"

- "segment_mask"
"""

from re import L
from typing import Dict
from hub.constants import (
    DEFAULT_CHUNK_COMPRESSION,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_HTYPE,
    DEFAULT_SAMPLE_COMPRESSION,
    UNCOMPRESSED,
)

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
    "bbox": {"dtype": "float32"},
    "video": {"dtype": "uint8"},
    "binary_mask": {
        "dtype": "bool"
    },  # TODO: pack numpy arrays to store bools as 1 bit instead of 1 byte
    "segment_mask": {"dtype": "int32"},
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "chunk_size": DEFAULT_MAX_CHUNK_SIZE,
    "chunk_compression": DEFAULT_CHUNK_COMPRESSION,
    "sample_compression": DEFAULT_SAMPLE_COMPRESSION,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v
