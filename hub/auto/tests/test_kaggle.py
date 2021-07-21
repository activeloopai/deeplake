from hub import Dataset
from hub.auto.tests.common import get_dummy_data_path
from hub.auto.unstructured.image_classification import ImageClassification
from hub.auto.unstructured.kaggle import download_kaggle_dataset
import os
import hub


def test_ingestion_simple():
    local = "./datasets/source/kaggle/simple"
    hub_path = "./datasets/destination/kaggle/simple"
    download_kaggle_dataset(
        "andradaolteanu/birdcall-recognition-data", local_path=local
    )
    ds = hub.Dataset(hub_path)
    unstructured = ImageClassification(source=local)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "jpeg"})
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images[9].numpy().shape == (570, 570, 3)
    assert ds.labels.numpy().shape == (10,)


def test_ingestion_sets():
    local = "./datasets/source/kaggle/sets"
    hub_path = "./datasets/destination/kaggle/sets"
    download_kaggle_dataset("thisiseshan/bird-classes", local_path=local)
    ds = hub.Dataset(hub_path)
    unstructured = ImageClassification(source=local)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "png"})
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
