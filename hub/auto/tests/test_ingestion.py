from hub.api.dataset import Dataset
from hub.auto.tests.common import get_dummy_data_path
from hub.auto.unstructured.image_classification import ImageClassification
from hub.auto.unstructured.kaggle import download_kaggle_dataset
import hub

kaggle_credentials = {
    "username": "thisiseshan",
    "key": "c5a2a9fe75044da342e95a341f882f31",
}


def test_local_ingestion_image_classification():
    path = get_dummy_data_path("image_classification")
    destination = "./datasets/destination/classification"
    ds = Dataset(destination)
    unstructured = ImageClassification(source=path)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "jpeg"})
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.meta.class_names == ("class0", "class1", "class2")


def test_local_image_classification_with_sets():
    path = get_dummy_data_path("image_classification_with_sets")
    destination = "./datasets/destination/classification_sets"
    ds = Dataset(destination)
    unstructured = ImageClassification(source=path)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "jpeg"})

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


def test_kaggle_ingestion_simple():
    tag = "andradaolteanu/birdcall-recognition-data"
    local = "./datasets/source/kaggle/simple"
    hub_path = "./datasets/destination/kaggle/simple"
    download_kaggle_dataset(
        tag, local_path=local, kaggle_credentials=kaggle_credentials
    )
    ds = hub.Dataset(hub_path)
    unstructured = ImageClassification(source=local)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "jpeg"})
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images[9].numpy().shape == (570, 570, 3)
    assert ds.labels.numpy().shape == (10,)


def test_kaggle_ingestion_sets():
    tag = "thisiseshan/bird-classes"
    local = "./datasets/source/kaggle/sets"
    hub_path = "./datasets/destination/kaggle/sets"
    download_kaggle_dataset(
        tag, local_path=local, kaggle_credentials=kaggle_credentials
    )
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
