from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.exceptions import InvalidPathException, SamePathException
import pytest
import hub


def test_auto_compression_ingestion_simple(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/image_classification")

    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            dest_creds=None,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(src=path, dest=path, dest_creds=None, overwrite=False)

    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        dest_creds=None,
        overwrite=False,
    )

    assert ds.images.meta.sample_compression == "jpeg"
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.info.class_names == ("class0", "class1", "class2")


def test_auto_compression_ingestion(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/auto_compression")

    with pytest.raises(InvalidPathException):
        hub.ingest(
            src="tests_auto/invalid_path",
            dest=memory_ds.path,
            dest_creds=None,
            overwrite=False,
        )

    with pytest.raises(SamePathException):
        hub.ingest(src=path, dest=path, dest_creds=None, overwrite=False)

    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        dest_creds=None,
        overwrite=False,
    )

    assert ds.images.meta.sample_compression == "png"
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3,)
    assert ds.labels.info.class_names == ("jpeg", "png")