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
| audio          |  float64  |  none         |
| text           |  str      |  none         |
| json           |  Any      |  none         |
| list           |  List     |  none         |
"""

from typing import Dict
from hub.compression import (
    IMAGE_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    COMPRESSION_ALIASES,
)

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
    "bbox": {"dtype": "float32", "coords": {}, "_info": ["coords"]},
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
    "dicom": {"sample_compression": "dcm"},
}

HTYPE_VERIFICATIONS: Dict[str, Dict] = {
    "bbox": {"coords": {"type": dict, "keys": ["type", "mode"]}}
}

_image_compressions = IMAGE_COMPRESSIONS[:]
_image_compressions.remove("dcm")

HTYPE_SUPPORTED_COMPRESSIONS = {
    "image": _image_compressions + BYTE_COMPRESSIONS + list(COMPRESSION_ALIASES),
    "video": VIDEO_COMPRESSIONS[:],
    "audio": AUDIO_COMPRESSIONS[:],
    "text": BYTE_COMPRESSIONS[:],
    "list": BYTE_COMPRESSIONS[:],
    "json": BYTE_COMPRESSIONS[:],
    "dicom": ["dcm"],
}


# these configs are added to every `htype`
COMMON_CONFIGS = {
    "sample_compression": None,
    "chunk_compression": None,
    "dtype": None,
    "max_chunk_size": None,
    "is_sequence": False,
    "is_link": False,
    "hidden": False,
    "links": None,
    "verify": False,
}


for config in HTYPE_CONFIGURATIONS.values():
    for key, v in COMMON_CONFIGS.items():
        # only update if not specified explicitly
        if key not in config:
            config[key] = v


def verify_htype_key_value(htype, key, value):
    htype_verifications = HTYPE_VERIFICATIONS.get(htype, {})
    if key in htype_verifications:
        expected_type = htype_verifications[key].get("type")
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"{key} must be of type {expected_type}, not {type(value)}")
        if expected_type == dict:
            expected_keys = set(htype_verifications[key].get("keys"))
            present_keys = set(value.keys())
            if expected_keys and not present_keys.issubset(expected_keys):
                raise KeyError(f"{key} must have keys belong to {expected_keys}")
