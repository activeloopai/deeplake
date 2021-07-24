from hub.api.dataset import Dataset
from hub.auto.unstructured.image_classification import ImageClassification
from hub.auto.unstructured.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError
import pytest
import os
import hub


def test_ingestion_simple(local_ds: Dataset):
    ds = local_ds
    kaggle_path = os.path.join(ds.path, "unstructured_kaggle_data_simple")
    download_kaggle_dataset("andradaolteanu/birdcall-recognition-data", kaggle_path)
    unstructured = ImageClassification(source=kaggle_path)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "jpeg"})

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds["labels"].numpy().shape == (10,)


def test_ingestion_sets(local_ds: Dataset):
    ds = local_ds
    kaggle_path = os.path.join(ds.path, "unstructured_kaggle_data_sets")
    download_kaggle_dataset("thisiseshan/bird-classes", local_path=kaggle_path)
    unstructured = ImageClassification(source=kaggle_path)
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


def test_kaggle_exception(local_ds: Dataset):
    ds = local_ds
    kaggle_path = os.path.join(ds.path, "unstructured_kaggle_data")

    with pytest.raises(KaggleDatasetAlreadyDownloadedError):
        download_kaggle_dataset("thisiseshan/bird-classes", local_path=kaggle_path)
        download_kaggle_dataset("thisiseshan/bird-classes", local_path=kaggle_path)
