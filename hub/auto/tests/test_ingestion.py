from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.auto.unstructured.image_classification import ImageClassification
from hub.util.exceptions import InvalidPathException
import pytest
import hub


def test_ingestion_simple(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")
    ds = hub.ingest(src=path, dest=memory_ds.path, src_creds=None, overwrite=False)

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.meta.class_names == ("class0", "class1", "class2")


def test_image_classification_sets(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    ds = hub.ingest(src=path, dest=memory_ds.path, src_creds=None, overwrite=False)

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3,)
    assert ds["test/labels"].meta.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3,)
    assert ds["train/labels"].meta.class_names == ("class0", "class1", "class2")