from hub.api.dataset import Dataset
from hub.auto.tests.common import get_dummy_data_path
from hub.auto.unstructured.image_classification import ImageClassification
import matplotlib.pyplot as plt
from hub.auto.unstructured.kaggle import download_kaggle_dataset, kaggle_credentials
import hub


def test_local_ingestion_image_classification():
    path = get_dummy_data_path("image_classification")
    destination = "./datasets/horse/destination/classification"
    ds = Dataset(destination)
    unstructured = ImageClassification(source=path)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "png"})
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 900, 900, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.meta.class_names == ("class0", "class1", "class2")
    plt.imshow(ds["images"][0].numpy())
    plt.show()


def test_local_image_classification_with_sets():
    path = get_dummy_data_path("image_classification_with_sets")
    destination = "./datasets/horse/destination/classification_sets"
    ds = Dataset(destination)
    unstructured = ImageClassification(source=path)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "png"})

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 900, 900, 3)
    assert ds["test/labels"].numpy().shape == (3,)
    assert ds["test/labels"].meta.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 900, 900, 3)
    assert ds["train/labels"].numpy().shape == (3,)
    assert ds["train/labels"].meta.class_names == ("class0", "class1", "class2")


def test_kaggle_ingestion_simple():
    tag = ""
    local = "./datasets/source/kaggle"
    hub_path = "./datasets/destination/kaggle"
    download_kaggle_dataset(
        tag, local_path=local, kaggle_credentials=kaggle_credentials
    )
    ds = hub.Dataset(hub_path)
    unstructured = ImageClassification(source=local)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "png"})
