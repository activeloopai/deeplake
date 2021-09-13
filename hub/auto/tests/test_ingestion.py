from hub.util.auto import ingestion_summary
from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.exceptions import (
    InvalidPathException,
    SamePathException,
    TensorAlreadyExistsError,
)
import pytest
import hub


def test_ingestion_simple(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")

    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            images_compression="auto",
            progress_bar=False,
            summary=False,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(
            src=path,
            dest=path,
            images_compression="jpeg",
            progress_bar=False,
            summary=False,
            overwrite=False,
        )

    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        images_compression="auto",
        progress_bar=False,
        summary=False,
        overwrite=False,
    )

    assert ds["images"].meta.sample_compression == "jpeg"
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds["images"].numpy().shape == (3, 200, 200, 3)
    assert ds["labels"].numpy().shape == (3, 1)
    assert ds["labels"].info.class_names == ("class0", "class1", "class2")


def test_image_classification_sets(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        images_compression="auto",
        progress_bar=False,
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
    assert ds["test/labels"].info.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3, 1)
    assert ds["train/labels"].info.class_names == ("class0", "class1", "class2")


def test_ingestion_exception(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification_with_sets")
    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            images_compression="auto",
            progress_bar=False,
            summary=False,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(
            src=path,
            dest=path,
            images_compression="auto",
            progress_bar=False,
            summary=False,
            overwrite=False,
        )


def test_overwrite(local_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")

    hub.ingest(
        src=path,
        dest=local_ds.path,
        images_compression="auto",
        progress_bar=False,
        summary=False,
        overwrite=False,
    )

    with pytest.raises(TensorAlreadyExistsError):
        hub.ingest(
            src=path,
            dest=local_ds.path,
            images_compression="auto",
            progress_bar=False,
            summary=False,
            overwrite=False,
        )
