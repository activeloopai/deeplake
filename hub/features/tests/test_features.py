from hub.features.class_label import ClassLabel, _load_names_from_file
from hub.features.features import HubFeature, FeatureDict
import pytest


names_file = "./hub/features/tests/class_label_names.txt"


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
    base_object = HubFeature()
    with pytest.raises(NotImplementedError):
        base_object._flatten()


def test_feature_dict_str():
    input_dict = {"myint": int, "mystr": str}
    feature_dict_object = FeatureDict(input_dict)
    expected_output = "FeatureDict({'myint': 'int64', 'mystr': '<U0'})"
    assert expected_output == feature_dict_object.__str__()


def test_feature_dict_repr():
    input_dict = {"myint": int, "mystr": str}
    feature_dict_object = FeatureDict(input_dict)
    expected_output = "FeatureDict({'myint': 'int64', 'mystr': '<U0'})"
    assert expected_output == feature_dict_object.__repr__()


if __name__ == "__main__":
    test_load_names_from_file()
    test_class_label()
    test_hub_feature_flatten()
    test_feature_dict_str()
    test_feature_dict_repr()
