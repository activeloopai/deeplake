import pytest

STORAGE_FIXTURE_NAME = "storage"

MEMORY = "memory"
LOCAL = "local"
S3 = "s3"


def c(*args):
    """Separate args with commas. This is helpful because pytest args will be
    printed as `memory,local` (or whichever storage providers are being used)
    for caches instead of `storage0`.
    """

    return ",".join(args)


ALL_PROVIDERS = [MEMORY, LOCAL, S3]
ALL_CACHES = [c(MEMORY, LOCAL), c(MEMORY, S3), c(LOCAL, S3), c(MEMORY, LOCAL, S3)]


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
    ALL_PROVIDERS + ALL_CACHES,
    indirect=True,
)
