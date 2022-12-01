from deeplake.api.dataset import Dataset
from deeplake.api.tests.test_api import convert_string_to_pathlib_if_needed
from deeplake.util.exceptions import (
    KaggleDatasetAlreadyDownloadedError,
    SamePathException,
    KaggleMissingCredentialsError,
    ExternalCommandError,
)
from click.testing import CliRunner
from deeplake.tests.common import get_dummy_data_path
import pytest
import os
import deeplake


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_ingestion_simple(
    local_ds: Dataset, hub_kaggle_credentials, convert_to_pathlib
):
    with CliRunner().isolated_filesystem():
        kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_simple")
        kaggle_path = convert_string_to_pathlib_if_needed(
            kaggle_path, convert_to_pathlib
        )
        username, key = hub_kaggle_credentials

        ds = deeplake.ingest_kaggle(
            tag="andradaolteanu/birdcall-recognition-data",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            kaggle_credentials={"username": username, "key": key},
            progressbar=False,
            summary=False,
            overwrite=False,
        )

        assert list(ds.tensors.keys()) == ["images", "labels"]
        assert ds["labels"].numpy().shape == (10, 1)


def test_ingestion_sets(local_ds: Dataset, hub_kaggle_credentials):
    with CliRunner().isolated_filesystem():
        kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data_sets")
        username, key = hub_kaggle_credentials

        ds = deeplake.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            kaggle_credentials={"username": username, "key": key},
            progressbar=False,
            summary=False,
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


def test_kaggle_exception(local_ds: Dataset, hub_kaggle_credentials):
    with CliRunner().isolated_filesystem():
        kaggle_path = os.path.join(local_ds.path, "unstructured_kaggle_data")
        dummy_path = get_dummy_data_path("tests_auto/image_classification")
        username, key = hub_kaggle_credentials

        with pytest.raises(SamePathException):
            deeplake.ingest_kaggle(
                tag="thisiseshan/bird-classes",
                src=dummy_path,
                dest=dummy_path,
                images_compression="jpeg",
                kaggle_credentials={"username": username, "key": key},
                progressbar=False,
                summary=False,
                overwrite=False,
            )

        with pytest.raises(KaggleMissingCredentialsError):
            deeplake.ingest_kaggle(
                tag="thisiseshan/bird-classes",
                src=kaggle_path,
                dest=local_ds.path,
                images_compression="jpeg",
                kaggle_credentials={"not_username": "not_username"},
                progressbar=False,
                summary=False,
                overwrite=False,
            )

        with pytest.raises(KaggleMissingCredentialsError):
            deeplake.ingest_kaggle(
                tag="thisiseshan/bird-classes",
                src=kaggle_path,
                dest=local_ds.path,
                images_compression="jpeg",
                kaggle_credentials={"username": "thisiseshan", "not_key": "not_key"},
                progressbar=False,
                summary=False,
                overwrite=False,
            )

        with pytest.raises(ExternalCommandError):
            deeplake.ingest_kaggle(
                tag="thisiseshan/invalid-dataset",
                src=kaggle_path,
                dest=local_ds.path,
                images_compression="jpeg",
                kaggle_credentials={"username": username, "key": key},
                progressbar=False,
                summary=False,
                overwrite=False,
            )

        deeplake.ingest_kaggle(
            tag="thisiseshan/bird-classes",
            src=kaggle_path,
            dest=local_ds.path,
            images_compression="jpeg",
            kaggle_credentials={"username": username, "key": key},
            progressbar=False,
            summary=False,
            overwrite=False,
        )

        with pytest.raises(KaggleDatasetAlreadyDownloadedError):
            deeplake.ingest_kaggle(
                tag="thisiseshan/bird-classes",
                src=kaggle_path,
                dest=local_ds.path,
                images_compression="jpeg",
                kaggle_credentials={"username": username, "key": key},
                progressbar=False,
                summary=False,
                overwrite=False,
            )
