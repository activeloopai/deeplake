from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.auto import get_most_common_extension, ingestion_summary
from io import StringIO
import sys
import pytest


def test_most_common_extension(memory_ds: Dataset):
    path = get_dummy_data_path("tests_auto/auto_compression")
    path_file = get_dummy_data_path("test_auto/auto_comrpession/jpeg/bird.jpeg")

    compression = get_most_common_extension(path)
    file_compression = get_most_common_extension(path_file)

    assert compression == "png"
    assert file_compression == "jpeg"


def test_ingestion_summary():
    path = get_dummy_data_path("tests_auto/auto_compression")

    capturedOutput = StringIO()
    sys.stdout = capturedOutput
    ingestion_summary(path, [], 1)
    sys.stdout = sys.__stdout__

    assert len(capturedOutput.getvalue()) == 251
    assert capturedOutput.getvalue()[225:-5] == "No files were skipped"
