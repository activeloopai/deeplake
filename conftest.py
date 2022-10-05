from deeplake.core.lock import _LOCKS, _REFS
import os
import logging

# Disable crash reporting before running tests
# This MUST come before hub imports to bypass import publication.
os.environ["BUGGER_OFF"] = "true"
os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "./testing/local_storage"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.disable(logging.INFO)

# Use staging environment for tests.
import deeplake.client.config

deeplake.client.config.USE_STAGING_ENVIRONMENT = True

from deeplake.constants import *
from deeplake.tests.common import SESSION_ID

# import * so all fixtures can be used accross the project
from deeplake.tests.path_fixtures import *
from deeplake.tests.dataset_fixtures import *
from deeplake.tests.storage_fixtures import *
from deeplake.tests.cache_fixtures import *
from deeplake.tests.client_fixtures import *
import pytest


def pytest_addoption(parser):
    parser.addoption(
        MEMORY_OPT, action="store_true", help="Memory tests will be SKIPPED if enabled."
    )
    parser.addoption(
        LOCAL_OPT, action="store_true", help="Local tests will run if enabled."
    )
    parser.addoption(S3_OPT, action="store_true", help="S3 tests will run if enabled.")
    parser.addoption(
        GCS_OPT, action="store_true", help="GCS tests will run if enabled."
    )
    parser.addoption(
        GDRIVE_OPT, action="store_true", help="Google Drive tests will run if enabled."
    )
    parser.addoption(
        HUB_CLOUD_OPT,
        action="store_true",
        help="Deep Lake cloud tests will run if enabled.",
    )
    parser.addoption(
        S3_PATH_OPT,
        type=str,
        help="Url to s3 bucket with optional key. Example: s3://bucket_name/key/to/tests/",
        default=PYTEST_S3_PROVIDER_BASE_ROOT,
    )
    parser.addoption(
        GDRIVE_PATH_OPT,
        type=str,
        help="Google drive folder id. Example: gdrive://folder_name/folder_name",
        default=PYTEST_GDRIVE_PROVIDER_BASE_ROOT,
    )
    parser.addoption(
        KEEP_STORAGE_OPT,
        action="store_true",
        help="All storage providers/datasets will have their pytest data wiped. \
                Use this option to keep the data after the test run. Note: does not keep memory tests storage.",
    )
    parser.addoption(
        KAGGLE_OPT, action="store_true", help="Kaggle tests will run if enabled."
    )


def print_session_id():
    print("\n\n----------------------------------------------------------")
    print(f"Testing session ID: {SESSION_ID}")
    print("----------------------------------------------------------")


print_session_id()


@pytest.fixture(scope="function", autouse=True)
def gc_lock_threads():
    start_keys = set(_LOCKS.keys())
    yield
    end_keys = set(_LOCKS.keys())
    for k in end_keys - start_keys:
        _LOCKS.pop(k).release()
        del _REFS[k]
