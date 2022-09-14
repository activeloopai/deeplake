"""
"htype" is the class of a tensor: image, bounding box, generic tensor, etc.

When not specified, the unspecified options will be inferred from the data:

>>> ds.create_tensor("my_tensor")
>>> ds.my_tensor.append(1)
>>> ds.my_tensor.dtype
int64

If you know beforehand, you can use htype at creation:

>>> ds.create_tensor("my_tensor", htype="image", sample_compression=None)

Specifying an htype allows for strict settings and error handling, and it is critical for increasing the performance of hub datasets containing rich data such as images and videos.

Supported htypes and their respective defaults are:

+----------------+-----------+---------------+
| HTYPE          |  DTYPE    |  COMPRESSION  |
+================+===========+===============+
| image          |  uint8    |  None         |
+----------------+-----------+---------------+
| image.rgb      |  uint8    |  None         |
+----------------+-----------+---------------+
| image.gray     |  uint8    |  None         |
+----------------+-----------+---------------+
| class_label    |  uint32   |  None         |
+----------------+-----------+---------------+
| bbox           |  float32  |  None         |
+----------------+-----------+---------------+
| video          |  uint8    |  None         |
+----------------+-----------+---------------+
| binary_mask    |  bool     |  None         |
+----------------+-----------+---------------+
| segment_mask   |  uint32   |  None         |
+----------------+-----------+---------------+
| keypoints_coco |  int32    |  None         |
+----------------+-----------+---------------+
| point          |  int32    |  None         |
+----------------+-----------+---------------+
| audio          |  float64  |  None         |
+----------------+-----------+---------------+
| text           |  str      |  None         |
+----------------+-----------+---------------+
| json           |  Any      |  None         |
+----------------+-----------+---------------+
| list           |  List     |  None         |
+----------------+-----------+---------------+
| dicom          |  None     |  dcm          |
+----------------+-----------+---------------+
| link           |  str      |  None         |
+----------------+-----------+---------------+
| sequence       |  None     |  None         |
+----------------+-----------+---------------+

Sequence htype
~~~~~~~~~~~~~~

- A special meta htype for tensors where each sample is a sequence. The items in the sequence are samples of another htype.
- It is a wrapper htype that can wrap other htypes like ``sequence[image]``, ``sequence[video]``, ``sequence[text]``, etc.

Examples
--------

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

>>> ds.create_tensor("image_seq", htype="sequence[image]", sample_compression="jpg")
>>> ds.image_seq.append([hub.read("img01.jpg"), hub.read("img02.jpg")])

Link htype
~~~~~~~~~~

- Link htype is a special meta htype that allows linking of external data (files) to the dataset, without storing the data in the dataset itself.
- Moreover, there can be variations in this htype, such as ``link[image]``, ``link[video]``, ``link[audio]``, etc. that would enable the activeloop visualizer to correctly display the data.
- No data is actually loaded until you try to read the sample from a dataset.
- There are a few exceptions to this:-
    - If ``verify=True`` was specified during ``create_tensor`` of the tensor to which this is being added, some metadata is read to verify the integrity of the sample.
    - If ``create_shape_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the shape of the sample is read.
    - If ``create_sample_info_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the sample info is read.

Examples
--------

>>> ds = hub.dataset("......")

Add the names of the creds you want to use (not needed for http/local urls)

>>> ds.add_creds_key("MY_S3_KEY")
>>> ds.add_creds_key("GCS_KEY")

Populate the names added with creds dictionary
These creds are only present temporarily and will have to be repopulated on every reload

>>> ds.populate_creds("MY_S3_KEY", {})   # add creds here
>>> ds.populate_creds("GCS_KEY", {})    # add creds here

Create a tensor that can contain links

>>> ds.create_tensor("img", htype="link[image]", verify=True, create_shape_tensor=False, create_sample_info_tensor=False)

Populate the tensor with links

>>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="MY_S3_KEY"))
>>> ds.img.append(hub.link("gcs://ghi/jkl.png", creds_key="GCS_KEY"))
>>> ds.img.append(hub.link("https://picsum.photos/200/300")) # http path doesn’t need creds
>>> ds.img.append(hub.link("./path/to/cat.jpeg")) # local path doesn’t need creds
>>> ds.img.append(hub.link("s3://abc/def.jpeg"))  # this will throw an exception as cloud paths always need creds_key
>>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="ENV"))  # this will use creds from environment

Accessing the data

>>> for i in range(5):
...     ds.img[i].numpy()
...

Updating a sample

>>> ds.img[0] = hub.link("./data/cat.jpeg")

"""

from typing import Dict
from hub.compression import (
    IMAGE_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    COMPRESSION_ALIASES,
    POINT_CLOUD_COMPRESSIONS,
)


class htype:
    DEFAULT = "generic"
    IMAGE = "image"
    IMAGE_RGB = "image.rgb"
    IMAGE_GRAY = "image.gray"
    CLASS_LABEL = "class_label"
    BBOX = "bbox"
    BBOX_3D = "bbox.3d"
    VIDEO = "video"
    BINARY_MASK = "binary_mask"
    INSTANCE_LABEL = "instance_label"
    SEGMENT_MASK = "segment_mask"
    KEYPOINTS_COCO = "keypoints_coco"
    POINT = "point"
    AUDIO = "audio"
    TEXT = "text"
    JSON = "json"
    LIST = "list"
    DICOM = "dicom"
    POINT_CLOUD = "point_cloud"
    POINT_CLOUD_CALIBRATION_MATRIX = "point_cloud.calibration_matrix"


# used for requiring the user to specify a value for htype properties. notates that the htype property has no default.
REQUIRE_USER_SPECIFICATION = "require_user_specification"

# used for `REQUIRE_USER_SPECIFICATION` enforcement. this should be used instead of `None` for default user method arguments.
UNSPECIFIED = "unspecified"


HTYPE_CONFIGURATIONS: Dict[str, Dict] = {
    htype.DEFAULT: {"dtype": None},
    htype.IMAGE: {
        "dtype": "uint8",
    },
    htype.IMAGE_RGB: {
        "dtype": "uint8",
    },
    htype.IMAGE_GRAY: {
        "dtype": "uint8",
    },
    htype.CLASS_LABEL: {
        "dtype": "uint32",
        "class_names": [],
        "_info": ["class_names"],  # class_names should be stored in info, not meta
        "_disable_temp_transform": False,
    },
    htype.BBOX: {"dtype": "float32", "coords": {}, "_info": ["coords"]},
    htype.BBOX_3D: {"dtype": "float32", "coords": {}, "_info": ["coords"]},
    htype.AUDIO: {"dtype": "float64"},
    htype.VIDEO: {"dtype": "uint8"},
    htype.BINARY_MASK: {
        "dtype": "bool"
    },  # TODO: pack numpy arrays to store bools as 1 bit instead of 1 byte
    htype.INSTANCE_LABEL: {"dtype": "uint32"},
    htype.SEGMENT_MASK: {
        "dtype": "uint32",
        "class_names": [],
        "_info": ["class_names"],
    },
    htype.KEYPOINTS_COCO: {"dtype": "int32"},
    htype.POINT: {"dtype": "int32"},
    htype.JSON: {
        "dtype": "Any",
    },
    htype.LIST: {"dtype": "List"},
    htype.TEXT: {"dtype": "str"},
    htype.DICOM: {"sample_compression": "dcm"},
    htype.POINT_CLOUD: {"sample_compression": "las"},
    htype.POINT_CLOUD_CALIBRATION_MATRIX: {"dtype": "float32"},
}

HTYPE_VERIFICATIONS: Dict[str, Dict] = {
    htype.BBOX: {"coords": {"type": dict, "keys": ["type", "mode"]}}
}

_image_compressions = (
    IMAGE_COMPRESSIONS[:] + BYTE_COMPRESSIONS + list(COMPRESSION_ALIASES)
)
_image_compressions.remove("dcm")

HTYPE_SUPPORTED_COMPRESSIONS = {
    htype.IMAGE: _image_compressions,
    htype.IMAGE_RGB: _image_compressions,
    htype.IMAGE_GRAY: _image_compressions,
    htype.VIDEO: VIDEO_COMPRESSIONS[:],
    htype.AUDIO: AUDIO_COMPRESSIONS[:],
    htype.TEXT: BYTE_COMPRESSIONS[:],
    htype.LIST: BYTE_COMPRESSIONS[:],
    htype.JSON: BYTE_COMPRESSIONS[:],
    htype.POINT_CLOUD: POINT_CLOUD_COMPRESSIONS[:],
    htype.DICOM: ["dcm"],
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
