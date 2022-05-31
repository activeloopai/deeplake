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
| image.rgb      |  uint8    |  none         |
| image.gray     |  uint8    |  none         |
| class_label    |  uint32   |  none         |
| bbox           |  float32  |  none         |
| video          |  uint8    |  none         |
| binary_mask    |  bool     |  none         |
| segment_mask   |  uint32   |  none         |
| keypoints_coco |  int32    |  none         |
| point          |  int32    |  none         |
| audio          |  float64  |  none         |
| text           |  str      |  none         |
| json           |  Any      |  none         |
| list           |  List     |  none         |
| link           |  str      |  none         |
| sequence       |  none     |  none         |

## Sequence htype

- An htype for tensors where each sample is a sequence. The items in the sequence are samples of another htype.
- It is a wrapper htype that can wrap other htypes like `sequence[image]`, `sequence[video]`, `sequence[text]`, etc.

### Examples:
```
>>> ds.create_tensor("seq", htype="sequence")
>>> ds.seq.append([1, 2, 3])
>>> ds.seq.append([4, 5, 6])
>>> ds.seq.numpy()
array([[[1],
        [2],
        [3]],

       [[4],
        [5],
        [6]]])
```

```
>>> ds.create_tensor("image_seq", htype="sequence[image]", sample_compression="jpg")
>>> ds.image_seq.append([hub.read("img01.jpg"), hub.read("img02.jpg")])
```

## Link htype

- Link htype is a special htype that allows linking of external data (files) to the dataset, without storing the data in the dataset itself.
- Moreover, there can be variations in this htype, such as `link[image]`, `link[video]`, `link[audio]`, etc. that would enable the activeloop visualizer to correctly display the data.
- No data is actually loaded until you try to read the sample from a dataset.
- There are a few exceptions to this:-
    - If `verify=True` was specified DURING `create_tensor` of the tensor to which this is being added, some metadata is read to verify the integrity of the sample.
    - If `create_shape_tensor=True` was specified DURING `create_tensor` of the tensor to which this is being added, the shape of the sample is read.
    - If `create_sample_info_tensor=True` was specified DURING `create_tensor` of the tensor to which this is being added, the sample info is read.
### Examples:
```
>>> ds = hub.dataset(“......“)
```

Add the names of the creds you want to use (not needed for http/local urls)

```
>>> ds.add_creds(“MY_S3_KEY”)
>>> ds.add_creds(“GCS_KEY”)
```

Populate the names added with creds dictionary
These creds are only present temporarily and will have to be repopulated on every reload
```
>>> ds.populate_creds(“MY_S3_KEY”, {})
>>> ds.populate_creds(“GCS_KEY”, {})
```

Create a tensor that can contain links
```
>>> ds.create_tensor(“img”, htype=“link[image]“, verify=True, create_shape_tensor=False, create_sample_info_tensor=False)
```

Populate the tensor with links
```
>>> ds.img.append(hub.link(“s3://abc/def.jpeg”, creds_key=“MY_S3_KEY”))
>>> ds.img.append(hub.link(“gcs://ghi/jkl.png”, creds_key=“GCS_KEY”))
>>> ds.img.append(hub.link(“https://picsum.photos/200/300”)) # doesn’t need creds
>>> ds.img.append(hub.link(“s3://abc/def.jpeg”))  # will use creds from environment
>>> ds.img.append(hub.link(“s3://abc/def.jpeg”, creds_key=“ENV”))  # this will also use creds from environment
```

Accessing the data
```
>>> for i in range(5):
>>>     ds.img[i].numpy()
```

Updating a sample
```
>>> ds.img[0] = hub.link(“./data/cat.jpeg”)
```
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
    "image.rgb": {
        "dtype": "uint8",
    },
    "image.gray": {
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
    "point": {"dtype": "int32"},
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

_image_compressions = (
    IMAGE_COMPRESSIONS[:] + BYTE_COMPRESSIONS + list(COMPRESSION_ALIASES)
)
_image_compressions.remove("dcm")

HTYPE_SUPPORTED_COMPRESSIONS = {
    "image": _image_compressions,
    "image.rgb": _image_compressions,
    "image.gray": _image_compressions,
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
    "tiling_threshold": None,
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
