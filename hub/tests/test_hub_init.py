from hub.exceptions import HubDatasetNotFoundException
import pytest

import hub
import hub.config
from hub.utils import dask_loaded


def test_local_mode():
    hub.local_mode()
    assert hub.config.HUB_REST_ENDPOINT == "http://localhost:5000"


def test_dev_mode():
    hub.dev_mode()
    assert hub.config.HUB_REST_ENDPOINT == "https://app.dev.activeloop.ai"


def test_load(caplog):
    if dask_loaded():
        obj = hub.load("./data/new/test")
        assert "Deprecated Warning" in caplog.text

    obj = hub.load("./data/test/test_dataset2")
    assert isinstance(obj, hub.Dataset) == True


def test_load_wrong_dataset():
    try:
        obj = hub.load("./data/dataset_that_does_not_exist")
    except Exception as ex:
        assert isinstance(ex, HubDatasetNotFoundException)


if __name__ == "__main__":
    test_local_mode()
    test_dev_mode()
    # test_load()
