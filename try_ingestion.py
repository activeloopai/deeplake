from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.auto.unstructured.image_classification import ImageClassification
import hub


def test_image_classification_simple():

    path = get_dummy_data_path("tests_auto/image_classification")
    destination = "./test_ingestion"

    ds = hub.dataset.ingest(src=path, dest=destination, src_creds=None, overwrite=False)

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.meta.class_names == ("class0", "class1", "class2")


def test_image_classification_sets():
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    destination = "./test_ingestion_sets"

    ds = hub.dataset.ingest(src=path, dest=destination, src_creds=None, overwrite=False)

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
