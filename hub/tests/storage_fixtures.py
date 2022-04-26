from hub.core.storage.gcs import GCSProvider
from hub.util.storage import storage_provider_from_hub_path
from hub.core.storage.s3 import S3Provider
from hub.core.storage.google_drive import GDriveProvider
from hub.core.storage.local import LocalProvider
from hub.core.storage.memory import MemoryProvider
from hub.constants import (
    PYTEST_S3_PROVIDER_BASE_ROOT,
    PYTEST_GCS_PROVIDER_BASE_ROOT,
    S3_OPT,
    GCS_OPT,
)
from hub.tests.common import is_opt_true
import pytest


enabled_storages = pytest.mark.parametrize(
    "storage",
    ["memory_storage", "local_storage", "s3_storage", "gcs_storage", "gdrive_storage"],
    indirect=True,
)

enabled_persistent_storages = pytest.mark.parametrize(
    "storage",
    ["local_storage", "s3_storage", "gcs_storage", "gdrive_storage"],
    indirect=True,
)


enabled_remote_storages = pytest.mark.parametrize(
    "storage",
    [
        "s3_storage",
        "gcs_storage",
        "gdrive_storage",
        "gcs_root_storage",
        "s3_root_storage",
    ],
    indirect=True,
)


@pytest.fixture
def memory_storage(memory_path):
    return MemoryProvider(memory_path)


@pytest.fixture
def local_storage(local_path):
    return LocalProvider(local_path)


@pytest.fixture
def s3_storage(s3_path):
    return S3Provider(s3_path)


@pytest.fixture
def gdrive_storage(gdrive_path, gdrive_creds):
    return GDriveProvider(gdrive_path, token=gdrive_creds)


@pytest.fixture
def gcs_storage(gcs_path):
    return GCSProvider(gcs_path)


@pytest.fixture
def s3_root_storage(request):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip()
        return

    return S3Provider(PYTEST_S3_PROVIDER_BASE_ROOT)


@pytest.fixture
def gcs_root_storage(request, gcs_creds):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip()
        return

    return GCSProvider(PYTEST_GCS_PROVIDER_BASE_ROOT, token=gcs_creds)


@pytest.fixture
def hub_cloud_storage(hub_cloud_path, hub_cloud_dev_token):
    return storage_provider_from_hub_path(hub_cloud_path, token=hub_cloud_dev_token)


@pytest.fixture
def storage(request):
    """Used with parametrize to use all enabled storage fixtures."""
    return request.getfixturevalue(request.param)
