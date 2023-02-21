from deeplake.api.dataset import Dataset
from deeplake.api.tests.test_api import convert_string_to_pathlib_if_needed
from deeplake.tests.common import get_dummy_data_path
from deeplake.util.exceptions import (
    InvalidPathException,
    SamePathException,
    DatasetHandlerError,
)
import numpy as np
import pytest
import deeplake
import pandas as pd  # type: ignore


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_ingestion_simple(memory_path: str, convert_to_pathlib: bool, shuffle: bool):
    path = get_dummy_data_path("tests_auto/image_classification")
    src = "tests_auto/invalid_path"

    if convert_to_pathlib:
        path = convert_string_to_pathlib_if_needed(path, convert_to_pathlib)
        src = convert_string_to_pathlib_if_needed(src, convert_to_pathlib)
        memory_path = convert_string_to_pathlib_if_needed(
            memory_path, convert_to_pathlib
        )

    with pytest.raises(InvalidPathException):
        deeplake.ingest_classification(
            src=src,
            dest=memory_path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        deeplake.ingest_classification(
            src=path,
            dest=path,
            image_params={"sample_compression": "jpeg"},
            progressbar=False,
            summary=False,
            overwrite=False,
        )

    ds = deeplake.ingest_classification(
        src=path,
        dest=memory_path,
        progressbar=False,
        summary=False,
        overwrite=False,
        shuffle=shuffle,
    )

    assert ds["images"].meta.sample_compression == "jpeg"
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds["images"].numpy().shape == (3, 200, 200, 3)
    assert ds["labels"].numpy().shape == (3, 1)
    assert ds["labels"].info.class_names == ["class0", "class1", "class2"]


def test_ingestion_with_params(memory_path: str):
    path = get_dummy_data_path("tests_auto/auto_compression")
    ds = deeplake.ingest_classification(
        src=path,
        dest=memory_path,
        progressbar=False,
        summary=False,
        overwrite=False,
    )
    assert ds["images"].meta.sample_compression == "png"

    explicit_images_name = "image_samples"
    explicit_labels_name = "label_samples"
    explicit_compression = "jpeg"

    ds = deeplake.ingest_classification(
        src=path,
        dest=memory_path,
        image_params={
            "name": explicit_images_name,
            "sample_compression": explicit_compression,
        },
        label_params={"name": explicit_labels_name},
        progressbar=False,
        summary=False,
        overwrite=True,
    )
    assert explicit_labels_name in ds.tensors
    assert explicit_images_name in ds.tensors
    assert ds["image_samples"].meta.sample_compression == explicit_compression


def test_image_classification_sets(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    ds = deeplake.ingest_classification(
        src=path,
        dest=memory_ds.path,
        progressbar=False,
        summary=False,
        overwrite=False,
    )

    assert list(ds.tensors) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]

    assert ds["train/images"].meta.sample_compression == "jpeg"
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3, 1)
    assert ds["test/labels"].info.class_names == ["class0", "class1", "class2"]

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3, 1)
    assert ds["train/labels"].info.class_names == ["class0", "class1", "class2"]


def test_ingestion_exception(memory_path: str):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    with pytest.raises(InvalidPathException):
        deeplake.ingest_classification(
            src="tests_auto/invalid_path",
            dest=memory_path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        deeplake.ingest_classification(
            src=path,
            dest=path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )


def test_overwrite(local_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")

    deeplake.ingest_classification(
        src=path,
        dest=local_ds.path,
        progressbar=False,
        summary=False,
        overwrite=True,
    )

    with pytest.raises(DatasetHandlerError):
        deeplake.ingest_classification(
            src=path,
            dest=local_ds.path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )


def test_ingestion_with_connection(
    s3_path,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    path = get_dummy_data_path("tests_auto/image_classification")
    ds = deeplake.ingest_classification(
        src=path,
        dest=s3_path,
        progressbar=False,
        summary=False,
        overwrite=False,
        connect_kwargs={
            "dest_path": hub_cloud_path,
            "creds_key": hub_cloud_dev_managed_creds_key,
            "token": hub_cloud_dev_token,
        },
    )

    assert ds.path == hub_cloud_path
    assert "images" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0


def test_csv(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/csv/deniro.csv")
    with pytest.raises(InvalidPathException):
        deeplake.ingest_classification(
            src="tests_auto/csv/cities.csv",
            dest=memory_ds.path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )

    ds = deeplake.ingest_classification(
        src=path,
        dest=memory_ds.path,
        progressbar=False,
        summary=False,
        overwrite=False,
    )
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    assert list(ds.tensors) == ["Year", "Score", "Title"]

    assert ds["Year"].dtype == df["Year"].dtype
    np.testing.assert_array_equal(ds["Year"].numpy().reshape(-1), df["Year"].values)

    assert ds["Score"].dtype == df["Score"].dtype
    np.testing.assert_array_equal(ds["Score"].numpy().reshape(-1), df["Score"].values)

    assert ds["Title"].htype == "text"
    assert ds["Title"].dtype == str
    np.testing.assert_array_equal(ds["Title"].numpy().reshape(-1), df["Title"].values)


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_dataframe(memory_ds: Dataset, convert_to_pathlib: bool):
    path = get_dummy_data_path("tests_auto/csv/deniro.csv")
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)
    ds = deeplake.ingest_dataframe(df, memory_ds.path, progressbar=False)

    with pytest.raises(Exception):
        memory_ds.path = convert_string_to_pathlib_if_needed(
            memory_ds, convert_to_pathlib
        )
        deeplake.ingest_dataframe(123, memory_ds.path)

    assert list(ds.tensors) == ["Year", "Score", "Title"]

    assert ds["Year"].dtype == df["Year"].dtype
    np.testing.assert_array_equal(ds["Year"].numpy().reshape(-1), df["Year"].values)

    assert ds["Score"].dtype == df["Score"].dtype
    np.testing.assert_array_equal(ds["Score"].numpy().reshape(-1), df["Score"].values)

    assert ds["Title"].htype == "text"
    assert ds["Title"].dtype == str
    np.testing.assert_array_equal(ds["Title"].numpy().reshape(-1), df["Title"].values)


def test_dataframe_with_connect(
    s3_path,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    path = get_dummy_data_path("tests_auto/csv/deniro.csv")
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)
    ds = deeplake.ingest_dataframe(
        df,
        s3_path,
        progressbar=False,
        connect_kwargs={
            "dest_path": hub_cloud_path,
            "creds_key": hub_cloud_dev_managed_creds_key,
            "token": hub_cloud_dev_token,
        },
    )

    assert ds.path == hub_cloud_path
    assert list(ds.tensors) == ["Year", "Score", "Title"]
    assert ds["Year"].dtype == df["Year"].dtype
    np.testing.assert_array_equal(ds["Year"].numpy().reshape(-1), df["Year"].values)

    assert ds["Score"].dtype == df["Score"].dtype
    np.testing.assert_array_equal(ds["Score"].numpy().reshape(-1), df["Score"].values)

    assert ds["Title"].htype == "text"
    assert ds["Title"].dtype == str
    np.testing.assert_array_equal(ds["Title"].numpy().reshape(-1), df["Title"].values)
