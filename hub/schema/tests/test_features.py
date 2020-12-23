from hub.schema import Segmentation
from hub.schema.class_label import ClassLabel, _load_names_from_file
from hub.schema.features import HubSchema, SchemaDict
import pytest


names_file = "./hub/schema/tests/class_label_names.txt"


def test_load_names_from_file():
    assert _load_names_from_file(names_file) == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_class_label():
    bel1 = ClassLabel(num_classes=4)
    bel2 = ClassLabel(names=["alpha", "beta", "gamma"])
    ClassLabel(names_file=names_file)
    assert bel1.names == ["0", "1", "2", "3"]
    assert bel2.names == ["alpha", "beta", "gamma"]
    assert bel1.str2int("1") == 1
    assert bel2.str2int("gamma") == 2
    assert bel1.int2str(2) is None  # FIXME This is a bug, should raise an error
    assert bel2.int2str(0) == "alpha"
    assert bel1.num_classes == 4
    assert bel2.num_classes == 3
    bel1.get_attr_dict()


def test_hub_feature_flatten():
    base_object = HubSchema()
    with pytest.raises(NotImplementedError):
        base_object._flatten()


def test_feature_dict_str():
    input_dict = {"myint": int, "mystr": str}
    feature_dict_object = SchemaDict(input_dict)
    expected_output = "SchemaDict({'myint': 'int64', 'mystr': '<U0'})"
    assert expected_output == feature_dict_object.__str__()


def test_feature_dict_repr():
    input_dict = {"myint": int, "mystr": str}
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


def test_classlabel_repr():
    cl1 = ClassLabel(num_classes=5)
    cl2 = ClassLabel(names=["apple", "orange", "banana"])

    text1 = "ClassLabel(shape=(), dtype='int64', num_classes=5)"
    text2 = "ClassLabel(shape=(), dtype='int64', names=['apple', 'orange', 'banana'], num_classes=3)"
    assert cl1.__repr__() == text1
    assert cl2.__repr__() == text2


if __name__ == "__main__":
    test_load_names_from_file()
    test_class_label()
    test_hub_feature_flatten()
    test_feature_dict_str()
    test_feature_dict_repr()
    test_classlabel_repr()
    test_segmentation_repr()
