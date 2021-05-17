import os
from uuid import uuid1

import pytest

from hub.constants import (MIN_LOCAL_CACHE_SIZE, MIN_MEMORY_CACHE_SIZE,
                           PYTEST_LOCAL_PROVIDER_BASE_ROOT,
                           PYTEST_MEMORY_PROVIDER_BASE_ROOT,
                           PYTEST_S3_PROVIDER_BASE_ROOT)
from hub.core.storage import LocalProvider, MemoryProvider, S3Provider
from hub.tests.common import SESSION_ID, current_test_name
from hub.util.cache_chain import get_cache_chain

STORAGE_CONFIG_MAP = {
    "memory": {
        "base_root": PYTEST_MEMORY_PROVIDER_BASE_ROOT,
        "class": MemoryProvider,
        "use_id": False,
        "is_id_prefix": False,  # if is_id_prefix (and use_id=True), the session id comes before test name, otherwise it is reversed
    },
    "local": {
        "base_root": PYTEST_LOCAL_PROVIDER_BASE_ROOT,
        "class": LocalProvider,
        "use_id": False,
        "is_id_prefix": False,
    },
    "s3": {
        "base_root": PYTEST_S3_PROVIDER_BASE_ROOT,
        "class": S3Provider,
        "use_id": True,
        "is_id_prefix": True,
    },
}


def _skip_if_none(val):
    if val is None:
        pytest.skip()


def _is_opt_true(request, opt):
    return request.config.getoption(opt)


def pytest_addoption(parser):
    parser.addoption(
        "--memory-skip",
        action="store_true",
        help="Tests using the `memory_provider` fixture will be skipped. Tests using the `storage` fixture will be skipped if called with \
                `MemoryProvider`.",
    )
    parser.addoption(
        "--local",
        action="store_true",
        help="Tests using the `storage`/`local_provider` fixtures will run with `LocalProvider`.",
    )
    parser.addoption(
        "--s3",
        action="store_true",
        help="Tests using the `storage`/`s3_provider` fixtures will run with `S3Provider`.",
    )
    parser.addoption(
        "--cache-chains",
        action="store_true",
        help="Tests using the `storage` fixture may run with combinations of all enabled providers \
                in cache chains. For example, if the option `--s3` is not provided, all cache chains that use `S3Provider` are skipped.",
    )
    parser.addoption(
        "--cache-chains-only",
        action="store_true",
        help="Force enables `--cache-chains`. `storage` fixture only returns cache chains. For example, if `--s3` is provided, \
            `storage` will never be just `S3Provider`.",
    )


def _get_storage_provider(storage_name, with_current_test_name=True):
    info = STORAGE_CONFIG_MAP[storage_name]
    root = info["base_root"]
    if with_current_test_name:
        path = current_test_name(
            with_id=info["use_id"], is_id_prefix=info["is_id_prefix"]
        )
        root = os.path.join(root, path)
    return info["class"](root)


@pytest.fixture
def memory_storage(request):
    if not _is_opt_true(request, "--memory-skip"):
        return _get_storage_provider("memory")


@pytest.fixture
def local_storage(request):
    if _is_opt_true(request, "--local"):
        # TODO: root as option
        return _get_storage_provider("local")


@pytest.fixture
def s3_storage(request):
    if _is_opt_true(request, "--s3"):
        # TODO: root as option
        return _get_storage_provider("s3")


@pytest.fixture
def storage(request, memory_storage, local_storage, s3_storage):
    requested_providers = request.param.split(",")

    # --cache-chains-only force enables --cache-chains
    use_cache_chains_only = _is_opt_true(request, "--cache-chains-only")
    use_cache_chains = _is_opt_true(request, "--cache-chains") or use_cache_chains_only

    if use_cache_chains_only and len(requested_providers) <= 1:
        pytest.skip()

    if not use_cache_chains and len(requested_providers) > 1:
        pytest.skip()

    storage_providers = []
    cache_sizes = []

    if "memory" in requested_providers:
        _skip_if_none(memory_storage)
        storage_providers.append(memory_storage)
        cache_sizes.append(MIN_MEMORY_CACHE_SIZE)
    if "local" in requested_providers:
        _skip_if_none(local_storage)
        storage_providers.append(local_storage)
        cache_sizes.append(MIN_LOCAL_CACHE_SIZE)
    if "s3" in requested_providers:
        _skip_if_none(s3_storage)
        storage_providers.append(s3_storage)

    if len(storage_providers) == len(cache_sizes):
        cache_sizes.pop()

    return get_cache_chain(storage_providers, cache_sizes)


@pytest.fixture(scope="session", autouse=True)
def clear_storages(request):
    # executed before the first test

    if not _is_opt_true(request, "--memory-skip"):
        storage = _get_storage_provider("memory", with_current_test_name=False)
        storage.clear()

    if _is_opt_true(request, "--local"):
        storage = _get_storage_provider("local", with_current_test_name=False)
        storage.clear()

    # don't clear S3 tests (these will be automatically cleared on occasion)

    yield

    # executed after the last test

    if _is_opt_true(request, "--s3"):
        # s3 is the only storage provider that uses the SESSION_ID prefix
        # if it is enabled, print it out after all tests finish
        print("\n\n")
        print("----------------------------------------------------------")
        print("Testing session ID: %s" % SESSION_ID)
        print("----------------------------------------------------------")
