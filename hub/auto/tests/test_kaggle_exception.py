from hub.auto.unstructured.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError
from hub.tests.common import get_dummy_data_path
import pytest


def test_kaggle_Exception():
    path = get_dummy_data_path("tests_auto/test_kaggle_exception")

    with pytest.raises(KaggleDatasetAlreadyDownloadedError):
        download_kaggle_dataset("thisiseshan/bird-classes", local_path=path)