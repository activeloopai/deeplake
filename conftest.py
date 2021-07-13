import os
import posixpath
import logging
import pytest

# Disable crash reporting before running tests
# This MUST come before hub imports to bypass import publication.
os.environ["BUGGER_OFF"] = "true"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.ERROR)

from hub.constants import *
from hub.tests.common import SESSION_ID
from hub.util.cache_chain import get_cache_chain

# import * so all fixtures can be used accross the project
from hub.tests.path_fixtures import *
from hub.tests.dataset_fixtures import *
from hub.tests.storage_fixtures import *
from hub.tests.client_fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        MEMORY_OPT, action="store_true", help="Memory tests will be SKIPPED if enabled."
    )
    parser.addoption(
        LOCAL_OPT, action="store_true", help="Local tests will run if enabled."
    )
    parser.addoption(S3_OPT, action="store_true", help="S3 tests will run if enabled.")
    parser.addoption(
        HUB_CLOUD_OPT, action="store_true", help="Hub cloud tests will run if enabled."
    )
    parser.addoption(
        CACHE_OPT,
        action="store_true",
        help="Tests using the `storage` fixture may run with combinations of all enabled providers in cache chains. "
        f"For example, if the option `{S3_OPT}` is not provided, all cache chains that use `S3Provider`"
        "  are skipped.",
    )
    parser.addoption(
        CACHE_ONLY_OPT,
        action="store_true",
        help=f"Force enables `{CACHE_OPT}`. `storage` fixture only returns cache chains. For example, if `{S3_OPT}` is provided, \
            `storage` will never be just `S3Provider`.",
    )
    parser.addoption(
        S3_PATH_OPT,
        type=str,
        help="Url to s3 bucket with optional key. Example: s3://bucket_name/key/to/tests/",
        default=PYTEST_S3_PROVIDER_BASE_ROOT,
    )
    parser.addoption(
        KEEP_STORAGE_OPT,
        action="store_true",
        help="All storage providers/datasets will have their pytest data wiped. \
                Use this option to keep the data after the test run.",
    )


@pytest.fixture
def marks(request):
    """Fixture that gets all `@pytest.mark`s. If a test is marked with
    `@pytest.mark.some_mark` the list this fixture returns will contain
    `some_mark` as a string.
    """

    marks = [m.name for m in request.node.iter_markers()]
    if request.node.parent:
        marks += [m.name for m in request.node.parent.iter_markers()]
    yield marks


def _storage_from_request(request):
    requested_providers = request.param.split(",")

    # --cache-chains-only force enables --cache-chains
    use_cache_chains_only = _is_opt_true(request, CACHE_ONLY_OPT)
    use_cache_chains = (
        _is_opt_true(request, CACHE_OPT)
        or use_cache_chains_only
        or _is_opt_true(request, FULL_BENCHMARK_OPT)
    )

    if use_cache_chains_only and len(requested_providers) <= 1:
        pytest.skip()

    if not use_cache_chains and len(requested_providers) > 1:
        pytest.skip()

    storage_providers = []
    cache_sizes = []

    if MEMORY in requested_providers:
        if _is_opt_true(request, MEMORY_OPT):
            pytest.skip()

        storage_providers.append(_get_memory_provider(request))
        cache_sizes.append(MIN_FIRST_CACHE_SIZE)
    if LOCAL in requested_providers:
        if not _is_opt_true(request, LOCAL_OPT):
            pytest.skip()

        storage_providers.append(_get_local_provider(request))
        cache_size = MIN_FIRST_CACHE_SIZE if not cache_sizes else MIN_SECOND_CACHE_SIZE
        cache_sizes.append(cache_size)
    if S3 in requested_providers:
        if not _is_opt_true(request, S3_OPT):
            pytest.skip()

        storage_providers.append(_get_s3_provider(request))

    if len(storage_providers) == len(cache_sizes):
        cache_sizes.pop()

    return get_cache_chain(storage_providers, cache_sizes)


# @pytest.fixture
# def memory_storage(request):
#     if not _is_opt_true(request, MEMORY_OPT):
#         return _get_memory_provider(request)
#     pytest.skip()
#
#
# @pytest.fixture
# def local_storage(request):
#     if _is_opt_true(request, LOCAL_OPT):
#         return _get_local_provider(request)
#     pytest.skip()
#
#
# @pytest.fixture
# def s3_storage(request):
#     if _is_opt_true(request, S3_OPT):
#         return _get_s3_provider(request)
#     pytest.skip()
#
#
# @pytest.fixture
# def storage(request):
#     return _storage_from_request(request)
#
#
# @pytest.fixture
# def memory_ds(memory_storage):
#     return _get_dataset(memory_storage)
#
#
# @pytest.fixture
# def local_ds(local_storage):
#     return _get_dataset(local_storage)
#
#
# @pytest.fixture
# def s3_ds(s3_storage):
#     return _get_dataset(s3_storage)
#
#
# @pytest.fixture
# def ds(request):
#     return _get_dataset(_storage_from_request(request))


def print_session_id():
    print("\n\n----------------------------------------------------------")
    print(f"Testing session ID: {SESSION_ID}")
    print("----------------------------------------------------------")


print_session_id()


# def _clear_storages(request):
#     # clear memory
#     if not _is_opt_true(request, MEMORY_OPT):
#         storage = _get_storage_provider(request, MEMORY, with_current_test_name=False)
#         storage.clear()
#
#     # clear local
#     if _is_opt_true(request, LOCAL_OPT):
#         storage = _get_storage_provider(request, LOCAL, with_current_test_name=False)
#         storage.clear()
#
#     # clear s3
#     if _is_opt_true(request, S3_OPT):
#         storage = _get_storage_provider(request, S3, with_current_test_name=False)
#         storage.clear()
#
#
# @pytest.fixture(scope="session", autouse=True)
# def clear_storages_session(request):
#     # executed before the first test
#     _clear_storages(request)
#
#     yield
#
#     # executed after the last test
#     print_session_id()
#
#
# @pytest.fixture(scope="function", autouse=True)
# def clear_storages_function(request):
#     # executed before the current test
#
#     yield
#
#     # executed after the current test
#     if not _is_opt_true(request, KEEP_STORAGE_OPT):
#         _clear_storages(request)
#
