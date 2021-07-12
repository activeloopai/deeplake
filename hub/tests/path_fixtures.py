import os
from conftest import S3_PATH_OPT
from hub.constants import (
    LOCAL_OPT,
    MEMORY_OPT,
    PYTEST_LOCAL_PROVIDER_BASE_ROOT,
    PYTEST_MEMORY_PROVIDER_BASE_ROOT,
    S3_OPT,
)
import posixpath
from hub.tests.common import (
    SESSION_ID,
    current_test_name,
    get_dummy_data_path,
    is_opt_true,
)
import pytest

MEMORY = "memory"
LOCAL = "local"
S3 = "s3"

ALL_STORAGES = [MEMORY, LOCAL, S3]


def _get_path_composition_configs(request):
    return {
        MEMORY: {
            "base_root": PYTEST_MEMORY_PROVIDER_BASE_ROOT,
            "use_id": False,
            "is_id_prefix": False,
            # if is_id_prefix (and use_id=True), the session id comes before test name, otherwise it is reversed
        },
        LOCAL: {
            "base_root": PYTEST_LOCAL_PROVIDER_BASE_ROOT,
            "use_id": False,
            "is_id_prefix": False,
        },
        S3: {
            "base_root": request.config.getoption(S3_PATH_OPT),
            "use_id": True,
            "is_id_prefix": True,
        },
    }


def _get_storage_path(
    request, storage_name, with_current_test_name=True, info_override={}
):
    info = _get_path_composition_configs(request)[storage_name]
    info.update(info_override)

    root = info["base_root"]

    path = ""
    if with_current_test_name:
        path = current_test_name()

    if info["use_id"]:
        if info["is_id_prefix"]:
            path = posixpath.join(SESSION_ID, path)
        else:
            path = posixpath.join(path, SESSION_ID)

    root = posixpath.join(root, path)
    return root


@pytest.fixture
def memory_path(request):
    if not is_opt_true(request, MEMORY_OPT):
        return _get_storage_path(request, MEMORY)
    pytest.skip()


@pytest.fixture
def local_path(request):
    if is_opt_true(request, LOCAL_OPT):
        return _get_storage_path(request, LOCAL)
    pytest.skip()


@pytest.fixture
def s3_path(request):
    if is_opt_true(request, S3_OPT):
        return _get_storage_path(request, S3)
    pytest.skip()


@pytest.fixture
def cat_path():
    path = get_dummy_data_path("compressed_images")
    return os.path.join(path, "cat.jpeg")


@pytest.fixture
def flower_path():
    path = get_dummy_data_path("compressed_images")
    return os.path.join(path, "flower.png")
