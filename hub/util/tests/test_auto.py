from hub.api.dataset import Dataset
from hub.tests.common import get_dummy_data_path
from hub.util.auto import get_most_common_extension, ingestion_summary
from io import StringIO
import sys
import pytest


def test_most_common_extension():
    path = get_dummy_data_path("tests_auto/auto_compression")
    path_file = get_dummy_data_path("test_auto/auto_comrpession/jpeg/bird.jpeg")

    compression = get_most_common_extension(path)
    file_compression = get_most_common_extension(path_file)

    assert compression == "png"
    assert file_compression == "jpeg"


def test_ingestion_summary_clean():
    clean_path = get_dummy_data_path("tests_auto/ingestion_summary/class1")

    ingest_summary_clean = StringIO()
    sys.stdout = ingest_summary_clean
    ingestion_summary(clean_path, [])
    sys.stdout = sys.__stdout__
    output = ingest_summary_clean.getvalue()

    assert output == "\n\nIngestion Complete. No files were skipped.\n\n\n"


def test_ingestion_summary_skipped():
    skipped_path = get_dummy_data_path("tests_auto/ingestion_summary")

    ingest_summary_skipped = StringIO()
    sys.stdout = ingest_summary_skipped
    ingestion_summary(skipped_path, ["test.json", "test.json"])
    sys.stdout = sys.__stdout__
    output = ingest_summary_skipped.getvalue()

    assert (
        output
        == "\n\ningestion_summary/    \n      [Skipped]  test.json\n      class0/    \n            [Skipped]  test.json\n      class1/    \n"
    )
