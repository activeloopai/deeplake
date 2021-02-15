"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.schema.audio import Audio
from hub.schema.bbox import BBox
from hub.schema.image import Image
from hub.schema.mask import Mask
from hub.schema.polygon import Polygon
from hub.schema.sequence import Sequence
from hub.schema.video import Video
from hub.schema import Segmentation
from hub.schema.class_label import ClassLabel
from hub.schema.class_label_array import ClassLabelArray
from hub.schema.bbox_array import BBoxArray
from hub.schema.image_array import ImageArray


from hub.schema.features import HubSchema, SchemaDict, Tensor
import pytest


def test_hub_feature_flatten():
    base_object = HubSchema()
    with pytest.raises(NotImplementedError):
        base_object._flatten()


def test_feature_dict_str():
    input_dict = {"myint": "int64", "mystr": str}
    feature_dict_object = SchemaDict(input_dict)
    expected_output = "SchemaDict({'myint': 'int64', 'mystr': '<U0'})"
    assert expected_output == feature_dict_object.__str__()


def test_feature_dict_repr():
    input_dict = {"myint": "int64", "mystr": str}
    feature_dict_object = SchemaDict(input_dict)
    expected_output = "SchemaDict({'myint': 'int64', 'mystr': '<U0'})"
    assert expected_output == feature_dict_object.__repr__()


def test_segmentation_repr():
    seg1 = Segmentation(shape=(3008, 3008), dtype="uint8", num_classes=5)
    seg2 = Segmentation(
        shape=(3008, 3008), dtype="uint8", names=["apple", "orange", "banana"]
    )

    text1 = "Segmentation(shape=(3008, 3008), dtype='uint8', num_classes=5)"
    text2 = "Segmentation(shape=(3008, 3008), dtype='uint8', names=['apple', 'orange', 'banana'], num_classes=3)"
    assert seg1.__repr__() == text1
    assert seg2.__repr__() == text2


def test_segmentation_classes():
    seg1 = Segmentation(shape=(3008, 3008), dtype="uint8", num_classes=5)
    seg2 = Segmentation(
        shape=(3008, 3008), dtype="uint8", names=["apple", "orange", "banana"]
    )
    assert seg1.get_segmentation_classes() == ["0", "1", "2", "3", "4"]
    assert seg2.get_segmentation_classes() == ["apple", "orange", "banana"]


def test_class_label():
    cl1 = ClassLabel(num_classes=5)
    cl2 = ClassLabel(names=["apple", "orange", "banana"])
    with pytest.raises(ValueError):
        cl3 = ClassLabel(names=["apple", "orange", "banana", "apple"])
    with pytest.raises(ValueError):
        cl4 = ClassLabel(names=["apple", "orange", "banana", "apple"], num_classes=2)
    cl5 = ClassLabel()
    cl6 = ClassLabel(names_file="./hub/schema/tests/class_label_names.txt")

    assert cl1.names == ["0", "1", "2", "3", "4"]
    assert cl2.names == ["apple", "orange", "banana"]
    assert cl6.names == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert cl1.num_classes == 5
    assert cl2.num_classes == 3
    assert cl1.str2int("3") == 3
    assert cl2.str2int("orange") == 1
    assert cl1.int2str(4) == "4"
    assert cl2.int2str(2) == "banana"

    with pytest.raises(KeyError):
        cl2.str2int("2")
    with pytest.raises(ValueError):
        cl1.str2int("8")
    with pytest.raises(ValueError):
        cl1.str2int("abc")
    with pytest.raises(ValueError):
        cl1.names = ["ab", "cd", "ef", "gh"]
    with pytest.raises(ValueError):
        cl2.names = ["ab", "cd", "ef", "gh"]


def test_class_label_array():
    cl1 = ClassLabelArray(shape=(5,), num_classes=5)
    cl2 = ClassLabelArray(
        shape=(None,), max_shape=(10,), names=["apple", "orange", "banana"]
    )
    with pytest.raises(ValueError):
        cl3 = ClassLabelArray(shape=(5,), names=["apple", "orange", "banana", "apple"])
    with pytest.raises(ValueError):
        cl4 = ClassLabelArray(
            shape=(5,), names=["apple", "orange", "banana", "apple"], num_classes=2
        )
    with pytest.raises(ValueError):
        cl7 = ClassLabelArray(
            shape=(None, 5), max_shape=(10, 5), names=["apple", "orange", "banana"]
        )
    cl5 = ClassLabel()
    cl6 = ClassLabel(names_file="./hub/schema/tests/class_label_names.txt")

    assert cl1.names == ["0", "1", "2", "3", "4"]
    assert cl2.names == ["apple", "orange", "banana"]
    assert cl6.names == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert cl1.num_classes == 5
    assert cl2.num_classes == 3
    assert cl1.str2int("3") == 3
    assert cl2.str2int("orange") == 1
    assert cl1.int2str(4) == "4"
    assert cl2.int2str(2) == "banana"

    with pytest.raises(KeyError):
        cl2.str2int("2")
    with pytest.raises(ValueError):
        cl1.str2int("8")
    with pytest.raises(ValueError):
        cl1.str2int("abc")
    with pytest.raises(ValueError):
        cl1.names = ["ab", "cd", "ef", "gh"]
    with pytest.raises(ValueError):
        cl2.names = ["ab", "cd", "ef", "gh"]


def test_polygon():
    with pytest.raises(ValueError):
        poly1 = Polygon(shape=(11, 3))
    with pytest.raises(ValueError):
        poly2 = Polygon(shape=(11, 4, 2))


def test_mask():
    with pytest.raises(ValueError):
        mask1 = Mask(shape=(11, 5))
    with pytest.raises(ValueError):
        mask2 = Mask(shape=(11, 4, 2))


test_image_inputs = [
    "uint32",
    "int16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "double",
]


@pytest.mark.parametrize("test_image", test_image_inputs)
def test_image(test_image):
    with pytest.raises(ValueError):
        image = Image((1920, 1080, 3), test_image)
    with pytest.raises(ValueError):
        image = Image((500, 500, 500, 500))


@pytest.mark.parametrize("test_image", test_image_inputs)
def test_image_array(test_image):
    with pytest.raises(ValueError):
        image_arr = ImageArray((50, 1920, 1080, 3), test_image)
    with pytest.raises(ValueError):
        image_arr = ImageArray((500, 500, 500, 500, 500))


def test_audio():
    with pytest.raises(ValueError):
        audio = Audio((1920, 3), "float32")


def test_bbox_array():
    with pytest.raises(ValueError):
        bbox = BBoxArray((100, 2))
    bbox = BBoxArray((100, 4))
    bbox = BBoxArray((None, 4), max_shape=(50, 4))


def test_image_repr():
    image = Image((1920, 1080, 3))
    text = "Image(shape=(1920, 1080, 3), dtype='uint8')"
    assert image.__repr__() == text


def test_image_array_repr():
    image_arr = ImageArray((None, 1920, 1080, 3), max_shape=(10, 1920, 1080, 3))
    text = "ImageArray(shape=(None, 1920, 1080, 3), dtype='uint8', max_shape=(10, 1920, 1080, 3))"
    assert image_arr.__repr__() == text


def test_classlabel_repr():
    cl1 = ClassLabel(num_classes=5)
    cl2 = ClassLabel(names=["apple", "orange", "banana"])

    text1 = "ClassLabel(shape=(), dtype='int64', num_classes=5)"
    text2 = "ClassLabel(shape=(), dtype='int64', names=['apple', 'orange', 'banana'], num_classes=3)"
    assert cl1.__repr__() == text1
    assert cl2.__repr__() == text2


def test_classlabel_array_repr():
    cl1 = ClassLabelArray(shape=(5,), num_classes=5)
    cl2 = ClassLabelArray(
        shape=(None,), max_shape=(10,), names=["apple", "orange", "banana"]
    )

    text1 = "ClassLabelArray(shape=(5,), dtype='int64', num_classes=5)"
    text2 = "ClassLabelArray(shape=(None,), dtype='int64', max_shape=(10,), names=['apple', 'orange', 'banana'], num_classes=3)"
    assert cl1.__repr__() == text1
    assert cl2.__repr__() == text2


def test_video_repr():
    vid = Video(shape=(120, 1920, 1080, 3))
    text = "Video(shape=(120, 1920, 1080, 3), dtype='uint8')"
    assert vid.__repr__() == text


def test_seq_repr():
    seq = Sequence(dtype=Tensor((10, 100, 100)))
    text = "Sequence(shape=(), dtype=Tensor(shape=(10, 100, 100), dtype='float64'))"
    assert seq.__repr__() == text


def test_polygon_repr():
    poly = Polygon(shape=(10, 2), chunks=10)
    text = "Polygon(shape=(10, 2), dtype='int32', chunks=(10,))"
    assert poly.__repr__() == text


def test_mask_repr():
    mask = Mask(shape=(1920, 1080, 1))
    text = "Mask(shape=(1920, 1080, 1), dtype='bool')"
    assert mask.__repr__() == text


def test_bbox_repr():
    bbox = BBox(dtype="uint32")
    text = "BBox(shape=(4,), dtype='uint32')"
    assert bbox.__repr__() == text


def test_bbox_array_repr():
    bbox_array = BBoxArray((None, 4), "uint32", (10, 4))
    text = "BBoxArray(shape=(None, 4), dtype='uint32', max_shape=(10, 4))"
    assert bbox_array.__repr__() == text


def test_audio_repr():
    audio = Audio((100,))
    text = "Audio(shape=(100,), dtype='int64')"
    assert audio.__repr__() == text


if __name__ == "__main__":
    test_class_label()
    test_hub_feature_flatten()
    test_feature_dict_str()
    test_feature_dict_repr()
    test_classlabel_repr()
    test_segmentation_repr()
    test_seq_repr()
    test_segmentation_classes()
    test_polygon_repr()
    test_polygon()
    test_mask()
    test_mask_repr()
    test_image()
    test_image_repr()
    test_bbox_repr()
    test_audio_repr()
    test_audio()
