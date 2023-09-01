from deeplake.core.storage.gcs import GCSProvider
from deeplake.util.storage import storage_provider_from_hub_path
from deeplake.core.storage.s3 import S3Provider
from deeplake.core.storage.google_drive import GDriveProvider
from deeplake.core.storage.local import LocalProvider
from deeplake.core.storage.memory import MemoryProvider
from deeplake.core.storage.azure import AzureProvider
from deeplake.constants import (
    PYTEST_S3_PROVIDER_BASE_ROOT,
    PYTEST_GCS_PROVIDER_BASE_ROOT,
    PYTEST_AZURE_PROVIDER_BASE_ROOT,
    S3_OPT,
    GCS_OPT,
    AZURE_OPT,
)
from deeplake.tests.common import is_opt_true
import pytest


enabled_storages = pytest.mark.parametrize(
    "storage",
    [
        "memory_storage",
        "local_storage",
        "s3_storage",
        "gcs_storage",
        "azure_storage",
        "gdrive_storage",
    ],
    indirect=True,
)

enabled_persistent_storages = pytest.mark.parametrize(
    "storage",
    ["local_storage", "s3_storage", "gcs_storage", "azure_storage", "gdrive_storage"],
    indirect=True,
)


enabled_remote_storages = pytest.mark.parametrize(
    "storage",
    [
        "s3_storage",
        "gcs_storage",
        "gdrive_storage",
        "gcs_root_storage",
        "azure_root_storage",
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
def azure_storage(azure_path):
    return AzureProvider(azure_path)


@pytest.fixture
def s3_root_storage(request):
    if not is_opt_true(request, S3_OPT):
        pytest.skip(f"{S3_OPT} flag not set")
        return

    return S3Provider(PYTEST_S3_PROVIDER_BASE_ROOT)


@pytest.fixture
def gcs_root_storage(request, gcs_creds):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip(f"{GCS_OPT} flag not set")
        return

    return GCSProvider(PYTEST_GCS_PROVIDER_BASE_ROOT, token=gcs_creds)


@pytest.fixture
def azure_root_storage(request):
    if not is_opt_true(request, AZURE_OPT):
        pytest.skip(f"{AZURE_OPT} flag not set")

    return AzureProvider(PYTEST_AZURE_PROVIDER_BASE_ROOT)


@pytest.fixture
def hub_cloud_storage(hub_cloud_path, hub_cloud_dev_token):
    return storage_provider_from_hub_path(hub_cloud_path, token=hub_cloud_dev_token)


@pytest.fixture
def storage(request):
    """Used with parametrize to use all enabled storage fixtures."""
    return request.getfixturevalue(request.param)
