import pytest

STORAGE_FIXTURE_NAME = "storage"
DATASET_FIXTURE_NAME = "ds"

MEMORY = "memory"
LOCAL = "local"
S3 = "s3"

ALL_PROVIDERS = [MEMORY, LOCAL, S3]
ALL_CACHES = [(MEMORY, LOCAL), (MEMORY, S3), (LOCAL, S3), (MEMORY, LOCAL, S3)]


parametrize_all_storages = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS,
    indirect=True,
)

parametrize_all_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_CACHES,
    indirect=True,
)

parametrize_all_storages_and_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS + ALL_CACHES,  # type: ignore
    indirect=True,
)

parametrize_all_dataset_storages = pytest.mark.parametrize(
    DATASET_FIXTURE_NAME, ALL_PROVIDERS, indirect=True
)

parametrize_all_dataset_storages_and_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS + ALL_CACHES,  # type: ignore
    indirect=True,
)
