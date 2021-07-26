from hub.api.dataset import Dataset
from hub.auto.unstructured.image_classification import ImageClassification
from hub.auto.unstructured.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError
import pytest
import os
import hub


def test_ingestion_simple(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_simple")
    ds = hub.ingest_kaggle(
        tag="andradaolteanu/birdcall-recognition-data",
        src=kaggle_path,
        dest=local_ds.path,
        src_creds={},
        compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds["labels"].numpy().shape == (10,)


def test_ingestion_sets(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_sets")
    ds = hub.ingest_kaggle(
        tag="thisiseshan/bird-classes",
        src=kaggle_path,
        dest=local_ds.path,
        src_creds={},
        compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3,)
    assert ds["test/labels"].info.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3,)
    assert ds["train/labels"].info.class_names == ("class0", "class1", "class2")


def test_kaggle_exception(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data")

    with pytest.raises(KaggleDatasetAlreadyDownloadedError):
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            src_creds={},
            compression="jpeg",
            overwrite=False,
        )
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            src_creds={},
            compression="jpeg",
            overwrite=False,
        )
