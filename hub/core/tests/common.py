import pytest

ALL_PROVIDERS = ["memory", "local", "s3"]
ALL_CACHES = ["memory,local", "memory,s3", "local,s3", "memory,local,s3"]


parametrize_all_storages = pytest.mark.parametrize(
    "storage",
    ALL_PROVIDERS,
    indirect=True,
)

parametrize_all_caches = pytest.mark.parametrize(
    "storage",  # caches are used the same as `storage`
    ALL_CACHES,
    indirect=True,
)

parametrize_all_storages_and_caches = pytest.mark.parametrize(
    "storage",
    ALL_PROVIDERS + ALL_CACHES,
    indirect=True,
)
