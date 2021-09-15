from hub.core.storage.gcs import GCSProvider
from hub.util.storage import storage_provider_from_hub_path
from hub.core.storage.s3 import S3Provider
from hub.core.storage.local import LocalProvider
from hub.core.storage.memory import MemoryProvider
import pytest


enabled_storages = pytest.mark.parametrize(
    "storage",
    ["memory_storage", "local_storage", "s3_storage", "gcs_storage"],
    indirect=True,
)

enabled_persistent_storages = pytest.mark.parametrize(
    "storage",
    ["local_storage", "s3_storage", "gcs_storage"],
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
def gcs_storage(gcs_path):
    return GCSProvider(gcs_path)


@pytest.fixture
def hub_cloud_storage(hub_cloud_path, hub_cloud_dev_token):
    return storage_provider_from_hub_path(hub_cloud_path, token=hub_cloud_dev_token)


@pytest.fixture
def storage(request):
    """Used with parametrize to use all enabled storage fixtures."""
    return request.getfixturevalue(request.param)
