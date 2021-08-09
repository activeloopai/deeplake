from hub.util.storage import storage_provider_from_hub_path
from hub.core.storage.s3 import S3Provider
from hub.core.storage.local import LocalProvider
import os
from conftest import S3_PATH_OPT
from hub.constants import (
    HUB_CLOUD_OPT,
    KEEP_STORAGE_OPT,
    LOCAL_OPT,
    MEMORY_OPT,
    PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT,
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
HUB_CLOUD = "hub_cloud"


def _get_path_composition_configs(request):
    return {
        MEMORY: {
            "base_root": PYTEST_MEMORY_PROVIDER_BASE_ROOT,
            "use_id": False,
            "is_id_prefix": False,
            # if is_id_prefix (and use_id=True), the session id comes before test name, otherwise it is reversed
            "use_underscores": False,
        },
        LOCAL: {
            "base_root": PYTEST_LOCAL_PROVIDER_BASE_ROOT,
            "use_id": False,
            "is_id_prefix": False,
            "use_underscores": False,
        },
        S3: {
            "base_root": request.config.getoption(S3_PATH_OPT),
            "use_id": True,
            "is_id_prefix": True,
            "use_underscores": False,
        },
        HUB_CLOUD: {
            "base_root": PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT,
            "use_id": True,
            "is_id_prefix": True,
            "use_underscores": True,
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

    if info["use_underscores"]:
        path = path.replace("/", "_")

    root = posixpath.join(root, path)
    return root


@pytest.fixture
def memory_path(request):
    if is_opt_true(request, MEMORY_OPT):
        pytest.skip()
        return

    # no need to clear memory paths
    return _get_storage_path(request, MEMORY)


@pytest.fixture
def local_path(request):
    if not is_opt_true(request, LOCAL_OPT):
        pytest.skip()
        return

    path = _get_storage_path(request, LOCAL)
    LocalProvider(path).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        LocalProvider(path).clear()


@pytest.fixture
def s3_path(request):
    if not is_opt_true(request, S3_OPT):
        pytest.skip()
        return

    path = _get_storage_path(request, S3)
    S3Provider(path).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        S3Provider(path).clear()


@pytest.fixture
def hub_cloud_path(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()
        return

    path = _get_storage_path(request, HUB_CLOUD)
    storage_provider_from_hub_path(path, token=hub_cloud_dev_token).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        storage_provider_from_hub_path(path, token=hub_cloud_dev_token).clear()


@pytest.fixture
def cat_path():
    """Path to a cat image in the dummy data folder. Expected shape: (900, 900, 3)"""

    path = get_dummy_data_path("compressed_images")
    return os.path.join(path, "cat.jpeg")


@pytest.fixture
def flower_path():
    """Path to a flower image in the dummy data folder. Expected shape: (513, 464, 4)"""

    path = get_dummy_data_path("compressed_images")
    return os.path.join(path, "flower.png")


@pytest.fixture
def compressed_image_paths():
    paths = {
        "webp": "beach.webp",
        "gif": "boat.gif",
        "bmp": "car.bmp",
        "jpeg": "cat.jpeg",
        "wmf": "crown.wmf",
        "dib": "dog.dib",
        "tiff": "field.tiff",
        "png": "flower.png",
        "ico": "sample_ico.ico",
        "jpeg2000": "sample_jpeg2000.jp2",
        "pcx": "sample_pcx.pcx",
        "ppm": "sample_ppm.ppm",
        "sgi": "sample_sgi.sgi",
        "tga": "sample_tga.tga",
        "xbm": "sample_xbm.xbm",
    }

    parent = get_dummy_data_path("compressed_images")
    for k in paths:
        paths[k] = os.path.join(parent, paths[k])

    return paths
