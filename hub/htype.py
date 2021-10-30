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

| HTYPE          |  DTYPE    |  COMPRESSION  |
| ------------   |  -------  |  -----------  |
| image          |  uint8    |  none         |
| class_label    |  uint32   |  none         |
| bbox           |  float32  |  none         |
| video          |  uint8    |  none         |
| binary_mask    |  bool     |  none         |
| segment_mask   |  uint32   |  none         |
| keypoints_coco |  int32    |  none         |

"""

from typing import Dict

DEFAULT_HTYPE = "generic"

# used for requiring the user to specify a value for htype properties. notates that the htype property has no default.
REQUIRE_USER_SPECIFICATION = "require_user_specification"

# used for `REQUIRE_USER_SPECIFICATION` enforcement. this should be used instead of `None` for default user method arguments.
UNSPECIFIED = "unspecified"


HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    DEFAULT_HTYPE: {"dtype": None},
    "image": {
        "dtype": "uint8",
    },
    "class_label": {
        "dtype": "uint32",
        "class_names": [],
        "_info": ["class_names"],  # class_names should be stored in info, not meta
    },
    "bbox": {"dtype": "float32"},
    "audio": {"dtype": "float64"},
    "video": {"dtype": "uint8"},
    "binary_mask": {
        "dtype": "bool"
    },  # TODO: pack numpy arrays to store bools as 1 bit instead of 1 byte
    "segment_mask": {"dtype": "uint32", "class_names": [], "_info": ["class_names"]},
    "keypoints_coco": {"dtype": "int32"},
    "json": {
        "dtype": "Any",
    },
    "list": {"dtype": "List"},
    "text": {"dtype": "str"},
}

# these configs are added to every `htype`
COMMON_CONFIGS = {
    "sample_compression": None,
    "chunk_compression": None,
    "dtype": None,
    "max_chunk_size": None,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v
