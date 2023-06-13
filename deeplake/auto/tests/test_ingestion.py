from deeplake.api.dataset import Dataset
from deeplake.api.tests.test_api import convert_string_to_pathlib_if_needed
from deeplake.tests.common import get_dummy_data_path
from deeplake.util.exceptions import (
    InvalidPathException,
    SamePathException,
    DatasetHandlerError,
    IngestionError,
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


def test_csv(memory_ds: Dataset, dataframe_ingestion_data: dict):
    with pytest.raises(InvalidPathException):
        deeplake.ingest_classification(
            src="tests_auto/csv/cities.csv",
            dest=memory_ds.path,
            progressbar=False,
            summary=False,
            overwrite=False,
        )

    ds = deeplake.ingest_classification(
        src=dataframe_ingestion_data["basic_dataframe_w_sanitize_path"],
        dest=memory_ds.path,
        progressbar=False,
        summary=False,
        overwrite=False,
    )
    tensors_names = list(ds.tensors.keys())

    df = pd.read_csv(
        dataframe_ingestion_data["basic_dataframe_w_sanitize_path"],
        quotechar='"',
        skipinitialspace=True,
    )
    df_keys = df.keys()

    assert (
        df_keys[0] in tensors_names and df_keys[2] in tensors_names
    )  # Second column should have been sanitized and got a new name

    assert ds[tensors_names[0]].dtype == df[df_keys[0]].dtype
    np.testing.assert_array_equal(
        ds[tensors_names[0]].numpy().reshape(-1), df[df_keys[0]].values
    )

    assert ds[tensors_names[1]].dtype == df[df_keys[1]].dtype
    np.testing.assert_array_equal(
        ds[tensors_names[1]].numpy().reshape(-1), df[df_keys[1]].values
    )

    assert ds[tensors_names[2]].htype == "text"
    assert ds[tensors_names[2]].dtype == str
    np.testing.assert_array_equal(
        ds[tensors_names[2]].numpy().reshape(-1), df[df_keys[2]].values
    )


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_dataframe_basic(
    memory_ds: Dataset, dataframe_ingestion_data: dict, convert_to_pathlib: bool
):
    df = pd.read_csv(
        dataframe_ingestion_data["basic_dataframe_w_sanitize_path"],
        quotechar='"',
        skipinitialspace=True,
    )

    df_keys = df.keys()
    key_0_new_name = "year_new"

    ds = deeplake.ingest_dataframe(
        df,
        memory_ds.path,
        progressbar=False,
        column_params={df_keys[0]: {"name": key_0_new_name}},
    )
    tensors_names = list(ds.tensors.keys())

    with pytest.raises(Exception):
        memory_ds.path = convert_string_to_pathlib_if_needed(
            memory_ds, convert_to_pathlib
        )
        deeplake.ingest_dataframe(123, memory_ds.path)

    assert key_0_new_name in tensors_names and df_keys[2] in tensors_names
    assert df_keys[1] not in tensors_names  # Second columnd should have been sanitized

    assert ds[key_0_new_name].dtype == df[df_keys[0]].dtype
    np.testing.assert_array_equal(
        ds[key_0_new_name].numpy().reshape(-1), df[df_keys[0]].values
    )

    assert ds[df_keys[2]].htype == "text"
    assert ds[df_keys[2]].dtype == str
    np.testing.assert_array_equal(
        ds[df_keys[2]].numpy().reshape(-1), df[df_keys[2]].values
    )


def test_dataframe_files(memory_ds: Dataset, dataframe_ingestion_data):
    df = pd.read_csv(dataframe_ingestion_data["dataframe_w_images_path"])
    df_keys = df.keys()

    df[df_keys[0]] = dataframe_ingestion_data["images_basepath"] + df[df_keys[0]]

    ds = deeplake.ingest_dataframe(
        df,
        memory_ds.path,
        column_params={
            df_keys[0]: {"htype": "image"},
            df_keys[2]: {"htype": "class_label"},
        },
        progressbar=False,
    )
    tensors_names = list(ds.tensors.keys())

    assert tensors_names == [df_keys[0], df_keys[1], df_keys[2]]
    assert ds[df_keys[0]].htype == "image"
    assert ds[df_keys[2]].htype == "class_label"

    assert ds[df_keys[0]].meta.sample_compression == "jpeg"

    assert len(ds[df_keys[0]][0].numpy().shape) == 3
    assert ds[df_keys[2]][2].data()["text"][0] == df[df_keys[2]][2]


def test_dataframe_array(memory_ds: Dataset):
    data = {
        "AA": ["Alice", "Bob", "Charlie", None],
        "BB": [
            np.array([3, 22, 1, 3]),
            np.array([1, 22, 10, 1]),
            np.array([2, 22, 1, 2]),
            np.array([0, 56, 34, 2]),
        ],
        "CC": [45, 67, 88, float("nan")],
        "DD": [None, None, None, None],
        "EE": [
            None,
            np.array([1, 22, 10, 1]),
            None,
            np.array([0, 56, 34]),
        ],
        "FF": [None, "Bob", "Charlie", "Dave"],
    }

    df = pd.DataFrame(data)
    df_keys = df.keys()

    ds = deeplake.ingest_dataframe(
        df,
        "mem://dummy",
        progressbar=False,
    )
    tensors_names = list(ds.tensors.keys())

    assert tensors_names == df_keys.tolist()
    assert ds[df_keys[0]].htype == "text"

    np.testing.assert_array_equal(
        ds[df_keys[1]].numpy(), np.stack([arr for arr in df[df_keys[1]].values], axis=0)
    )

    np.testing.assert_array_equal(
        ds[df_keys[2]].numpy().reshape(-1), df[df_keys[2]].values
    )
    assert ds[df_keys[2]].dtype == df[df_keys[2]].dtype

    data_key_4 = ds[df_keys[4]].numpy(aslist=True)
    ds_data = [None if arr.shape[0] == 0 else arr for arr in data_key_4]
    df_data = [arr for arr in df[df_keys[4]].values]

    assert len(ds[df_keys[4]].numpy(aslist=True)) == 4
    assert ds[df_keys[4]][0].numpy().tolist() == []
    assert ds[df_keys[4]][0].numpy().shape[0] == 0
    assert ds[df_keys[4]][1].numpy().shape[0] == 4


def test_dataframe_array_bad(memory_ds: Dataset):
    data = {
        "AA": ["Alice", "Bob", "Charlie", None],
        "BB": [
            np.array([80, 75, 85, 100]),
            None,
            np.array([0, 565, 234]),
            "bad_data",
        ],
        "CC": [45, 67, 88, 77],
    }

    df = pd.DataFrame(data)

    with pytest.raises(IngestionError):
        ds = deeplake.ingest_dataframe(
            df,
            memory_ds.path,
            progressbar=False,
        )


def test_dataframe_all_empty_images(memory_ds: Dataset):
    data = {
        "AA": ["Alice", "Bob", "Charlie", "Steve"],
        "BB": [
            None,
            None,
            None,
            None,
        ],
        "CC": [45, 67, 88, 77],
    }

    df = pd.DataFrame(data)

    with pytest.raises(IngestionError):
        ds = deeplake.ingest_dataframe(
            df,
            "mem://dummy",
            progressbar=False,
            column_params={"BB": {"htype": "image"}},
        )


def test_dataframe_unsupported_file(memory_ds: Dataset, dataframe_ingestion_data):
    df = pd.read_csv(dataframe_ingestion_data["dataframe_w_bad_images_path"])
    df_keys = df.keys()

    df[df_keys[0]] = dataframe_ingestion_data["images_basepath"] + df[df_keys[0]]

    with pytest.raises(IngestionError):
        ds = deeplake.ingest_dataframe(
            df,
            memory_ds.path,
            column_params={
                df_keys[0]: {"htype": "image"},
                df_keys[2]: {"htype": "class_label"},
            },
            progressbar=False,
        )


def test_dataframe_with_connect(
    s3_path,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
    dataframe_ingestion_data,
):
    df = pd.read_csv(
        dataframe_ingestion_data["basic_dataframe_w_sanitize_path"],
        quotechar='"',
        skipinitialspace=True,
    )
    df_keys = df.keys()

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

    assert ds[df_keys[0]].dtype == df[df_keys[0]].dtype
    np.testing.assert_array_equal(
        ds[df_keys[0]].numpy().reshape(-1), df[df_keys[0]].values
    )

    assert ds[df_keys[2]].htype == "text"
    assert ds[df_keys[2]].dtype == str
    np.testing.assert_array_equal(
        ds[df_keys[2]].numpy().reshape(-1), df[df_keys[2]].values
    )
