from deeplake.util.cache_chain import get_cache_chain
from deeplake.constants import MIN_FIRST_CACHE_SIZE, MIN_SECOND_CACHE_SIZE
import pytest

_MEMORY = "memory_storage"
_LOCAL = "local_storage"
_S3 = "s3_storage"
_HUB_CLOUD = "hub_cloud_storage"

CACHE_CHAINS = [
    (_MEMORY, _LOCAL),
    (_MEMORY, _S3),
    (_MEMORY, _HUB_CLOUD),
    (_LOCAL, _S3),
    (_LOCAL, _HUB_CLOUD),
    (_MEMORY, _LOCAL, _S3),
    (_MEMORY, _LOCAL, _HUB_CLOUD),
]
CACHE_CHAINS = list(map(lambda i: ",".join(i), CACHE_CHAINS))  # type: ignore


enabled_cache_chains = pytest.mark.parametrize(
    "cache_chain",
    CACHE_CHAINS,
    indirect=True,
)


@pytest.fixture
def cache_chain(request):
    requested_storages = request.param.split(",")

    storages = []
    cache_sizes = []

    # will automatically skip if a storage is not enabled
    for requested_storage in requested_storages:
        storage = request.getfixturevalue(requested_storage)
        storages.append(storage)

        if len(cache_sizes) == 0:
            cache_sizes.append(MIN_FIRST_CACHE_SIZE)
        elif len(cache_sizes) == 1:
            cache_sizes.append(MIN_SECOND_CACHE_SIZE)

    if len(storages) == len(cache_sizes):
        cache_sizes.pop()

    assert (
        len(storages) == len(cache_sizes) + 1
    ), f"Invalid test composition. {len(storages)} != {len(cache_sizes)} - 1"

    return get_cache_chain(storages, cache_sizes)
