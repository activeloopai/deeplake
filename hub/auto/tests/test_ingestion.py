from hub.api.dataset import Dataset
from hub import from_path
from hub.auto.tests.common import get_dummy_data_path


def test_image_classification(memory_storage):
    path = get_dummy_data_path("image_classification")
    ds = from_path(source=path, destination=memory_storage, use_progress_bar=False)

    assert ds.mode == "r"
    assert ds.keys() == ("images", "labels")
    assert ds.images.numpy().shape == (3, 900, 900, 3)
    assert ds.labels.numpy().shape == (3, 1)
    assert ds.labels.meta.class_names == ["class0", "class1", "class2"]


def test_image_classification_with_sets(memory_storage):
    path = get_dummy_data_path("image_classification_with_sets")
    ds = from_path(source=path, destination=memory_storage, use_progress_bar=False)

    assert ds.mode == "r"
    assert ds.keys() == ("test/images", "test/labels", "train/images", "train/labels")

    assert ds["test/images"].numpy().shape == (3, 900, 900, 3)
    assert ds["test/labels"].numpy().shape == (3, 1)
    assert ds["test/labels"].meta.class_names == ["class0", "class1", "class2"]

    assert ds["train/images"].numpy().shape == (3, 900, 900, 3)
    assert ds["train/labels"].numpy().shape == (3, 1)
    assert ds["train/labels"].meta.class_names == ["class0", "class1", "class2"]
    del ds

    # user wants to update their class names
    ds = Dataset(storage=memory_storage, mode="w")
    ds["train/labels"].meta.class_names[2] = "dog"
    ds.flush()
    del ds

    ds = Dataset(storage=memory_storage, mode="w")
    assert ds["train/labels"].meta.class_names == ["class0", "class1", "dog"]
