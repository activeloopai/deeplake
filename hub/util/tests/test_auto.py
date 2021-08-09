from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.exceptions import InvalidPathException, SamePathException
import pytest
import hub


def test_auto_compression(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/auto_compression")

    ds = hub.ingest(
        src=path,
        dest=memory_ds.path,
        dest_creds=None,
        overwrite=False,
    )

    assert ds.images.meta.sample_compression == "png"
    assert list(ds.tensors.keys()) == ["images", "labels"]
    assert ds.images.numpy().shape == (3, 200, 200, 3)
    assert ds.labels.numpy().shape == (3, 1)
    assert ds.labels.info.class_names == ("jpeg", "png")
