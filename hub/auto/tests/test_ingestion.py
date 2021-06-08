from hub import from_path
from hub.auto.tests.common import get_dummy_data_path


def test_image_classification():
    path = get_dummy_data_path("image_classification")
    ds = from_path(path)

    assert ds.keys() == ("images", "labels")
    assert ds.images.numpy().shape == (3, 900, 900, 3)
    assert ds.labels.numpy().shape == (3, 1)
    assert ds.labels.meta["class_names"] == ("class0", "class1", "class2")