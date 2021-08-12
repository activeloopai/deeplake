from hub.auto.unstructured import kaggle
from hub.api.dataset import Dataset
from hub.util.exceptions import (
    KaggleDatasetAlreadyDownloadedError,
    SamePathException,
    KaggleMissingCredentialsError,
    ExternalCommandError,
)
from hub.tests.common import get_dummy_data_path
import pytest
import os
import hub


def test_ingestion_simple(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_simple")
    ds = hub.ingest_kaggle(
        tag="andradaolteanu/birdcall-recognition-data",
        src=kaggle_path,
        dest=local_ds.path,
        images_compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds["labels"].numpy().shape == (10, 1)


def test_ingestion_sets(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_sets")

    ds = hub.ingest_kaggle(
        tag="thisiseshan/bird-classes",
        src=kaggle_path,
        dest=local_ds.path,
        images_compression="jpeg",
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3, 1)
    assert ds["test/labels"].info.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3, 1)
    assert ds["train/labels"].info.class_names == ("class0", "class1", "class2")


def test_kaggle_exception(local_ds: Dataset):
    kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data")
    dummy_path = get_dummy_data_path("tests_auto/image_classification")

    with pytest.raises(SamePathException):
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=dummy_path,
            dest=dummy_path,
            images_compression="jpeg",
            overwrite=False,
        )

    with pytest.raises(KaggleMissingCredentialsError):
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            kaggle_credentials={"not_username": "not_username"},
            overwrite=False,
        )

    with pytest.raises(KaggleMissingCredentialsError):
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            kaggle_credentials={"username": "thisiseshan", "not_key": "not_key"},
            overwrite=False,
        )

    with pytest.raises(ExternalCommandError):
        hub.ingest_kaggle(
            tag="thisiseshan/invalid-dataset",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            overwrite=False,
        )

    hub.ingest_kaggle(
        tag="thisiseshan/bird-classes",
        src=kaggle_path,
        dest=local_ds.path,
        images_compression="jpeg",
        overwrite=False,
    )

    with pytest.raises(KaggleDatasetAlreadyDownloadedError):
        hub.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            overwrite=False,
        )
