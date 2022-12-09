from typing import Dict
from deeplake.compression import (
    IMAGE_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    COMPRESSION_ALIASES,
    POINT_CLOUD_COMPRESSIONS,
    MESH_COMPRESSIONS,
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
    POLYGON = "polygon"
    MESH = "mesh"


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
    htype.POINT_CLOUD: {"dtype": "float32"},
    htype.POINT_CLOUD_CALIBRATION_MATRIX: {"dtype": "float32"},
    htype.POLYGON: {"dtype": "float32"},
    htype.MESH: {"sample_compression": "ply"},
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
    htype.POLYGON: BYTE_COMPRESSIONS[:],
    htype.MESH: MESH_COMPRESSIONS[:],
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
