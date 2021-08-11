from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.auto import get_most_common_extension
import pytest


def test_most_common_extension(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/auto_compression")
    path_file = get_dummy_data_path("test_auto/auto_comrpession/jpeg/bird.jpeg")

    compression = get_most_common_extension(path)
    file_compression = get_most_common_extension(path_file)

    assert compression == "png"
    assert file_compression == "jpeg"
