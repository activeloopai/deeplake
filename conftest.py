import os
import pytest
from uuid import uuid1

from hub.constants import (
    PYTEST_MEMORY_PROVIDER_BASE_ROOT,
    PYTEST_LOCAL_PROVIDER_BASE_ROOT,
    PYTEST_S3_PROVIDER_BASE_ROOT,
)
from hub.core.storage import S3Provider, MemoryProvider, LocalProvider
from hub.util.cache_chain import get_cache_chain
from hub.constants import MB

SESSION_UUID = str(uuid1())


def _skip_if_none(val):
    if val is None:
        pytest.skip()


def _is_opt_true(request, opt):
    return request.config.getoption(opt)


def pytest_addoption(parser):
    parser.addoption(
        "--memory-skip",
        action="store_true",
        help="if this option is provided, `MemoryProvider` tests will be skipped.",
    )
    parser.addoption(
        "--local",
        action="store_true",
        help="if this option is provided, `LocalProvider` tests will be used.",
    )
    parser.addoption(
        "--s3",
        action="store_true",
        help="if this option is provided, `S3Provider` tests will be used. Credentials are required.",
    )


@pytest.fixture(scope="session")
def memory_storage(request):
    if not _is_opt_true(request, "--memory-skip"):
        return MemoryProvider(PYTEST_MEMORY_PROVIDER_BASE_ROOT)


@pytest.fixture(scope="session")
def local_storage(request):
    if _is_opt_true(request, "--local"):
        # TODO: root as option
        local = LocalProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT)
        return local


@pytest.fixture(scope="session")
def s3_storage(request):
    if _is_opt_true(request, "--s3"):
        # TODO: root as option
        root = os.path.join(PYTEST_S3_PROVIDER_BASE_ROOT, SESSION_UUID)
        return S3Provider(root)


@pytest.fixture(scope="session")
def storage(request, memory_storage, local_storage, s3_storage):
    if request.param == "memory":
        _skip_if_none(memory_storage)
        return memory_storage
    if request.param == "local":
        _skip_if_none(local_storage)
        return local_storage
    if request.param == "s3":
        _skip_if_none(s3_storage)
        return s3_storage


@pytest.fixture(scope="session", autouse=True)
def clear_storages(memory_storage, local_storage):
    # executed before the first test

    print()
    print()

    if memory_storage:
        print("Clearing memory storage provider")
        memory_storage.clear()

    if local_storage:
        print("Clearing local storage provider")
        local_storage.clear()

    print()

    yield

    # executed after the last test


# caches corresponding to pytest paremetrize

# memory_local_cache = get_cache_chain(
#     [
#         MemoryProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#         LocalProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#     ],
#     [32 * MB],
# )

# memory_local_s3_cache = get_cache_chain(
#     [
#         MemoryProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#         LocalProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#         S3Provider(PYTEST_S3_PROVIDER_BASE_ROOT),
#     ],
#     [32 * MB, 160 * MB],
# )

# memory_s3_cache = get_cache_chain(
#     [
#         MemoryProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#         S3Provider(PYTEST_S3_PROVIDER_BASE_ROOT),
#     ],
#     [32 * MB],
# )

# local_s3_cache = get_cache_chain(
#     [
#         LocalProvider(PYTEST_LOCAL_PROVIDER_BASE_ROOT),
#         S3Provider(PYTEST_S3_PROVIDER_BASE_ROOT),
#     ],
#     [32 * MB],
# )
