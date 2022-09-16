from hub.core.storage.gcs import GCSProvider
from hub.core.storage.google_drive import GDriveProvider
from hub.util.storage import storage_provider_from_hub_path
from hub.core.storage.s3 import S3Provider
from hub.core.storage.local import LocalProvider
import os
import hub
from hub.constants import (
    HUB_CLOUD_OPT,
    KEEP_STORAGE_OPT,
    LOCAL_OPT,
    MEMORY_OPT,
    PYTEST_GCS_PROVIDER_BASE_ROOT,
    PYTEST_S3_PROVIDER_BASE_ROOT,
    PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT,
    PYTEST_LOCAL_PROVIDER_BASE_ROOT,
    PYTEST_MEMORY_PROVIDER_BASE_ROOT,
    S3_OPT,
    GCS_OPT,
    GDRIVE_OPT,
    S3_PATH_OPT,
    GDRIVE_PATH_OPT,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GDRIVE_CLIENT_ID,
    ENV_GDRIVE_CLIENT_SECRET,
    ENV_GDRIVE_REFRESH_TOKEN,
)
import posixpath
from hub.tests.common import (
    SESSION_ID,
    current_test_name,
    get_dummy_data_path,
    is_opt_true,
)
import pytest
import sys


MEMORY = "memory"
LOCAL = "local"
S3 = "s3"
GDRIVE = "gdrive"
GCS = "gcs"
HUB_CLOUD = "hub_cloud"

_GIT_CLONE_CACHE_DIR = ".test_resources"

_HUB_TEST_RESOURCES_URL = "https://www.github.com/activeloopai/hub-test-resources.git"
_PILLOW_URL = "https://www.github.com/python-pillow/Pillow.git"


def _repo_name_from_git_url(url):
    repo_name = posixpath.split(url)[-1]
    repo_name = repo_name.split("@", 1)[0]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name


def _git_clone(url):
    _repo_name = _repo_name_from_git_url(url)
    cached_dir = _GIT_CLONE_CACHE_DIR + "/" + _repo_name
    if not os.path.isdir(cached_dir):
        if not os.path.isdir(_GIT_CLONE_CACHE_DIR):
            os.mkdir(_GIT_CLONE_CACHE_DIR)
        cwd = os.getcwd()
        os.chdir(_GIT_CLONE_CACHE_DIR)
        try:
            os.system(f"git clone " + url)
        finally:
            os.chdir(cwd)
    assert os.path.isdir(cached_dir)
    return cached_dir


def _download_hub_test_images():
    path = _git_clone(_HUB_TEST_RESOURCES_URL)
    jpeg_path = path + "/images/jpeg"
    return [os.path.join(jpeg_path, f) for f in os.listdir(jpeg_path)]


def _download_hub_test_videos():
    path = _git_clone(_HUB_TEST_RESOURCES_URL)
    mp4_path = path + "/videos/mp4"
    return [os.path.join(mp4_path, f) for f in os.listdir(mp4_path)]


def _download_pil_test_images(ext=[".jpg", ".png"]):
    paths = {e: [] for e in ext}
    corrupt_file_keys = [
        "broken",
        "_dos",
        "truncated",
        "chunk_no_fctl",
        "syntax_num_frames_zero",
    ]

    path = _git_clone(_PILLOW_URL)
    dirs = [
        path + x
        for x in [
            "/Tests/images",
            "/Tests/images/apng",
            "/Tests/images/imagedraw",
        ]
    ]
    for d in dirs:
        for f in os.listdir(d):
            brk = False
            for k in corrupt_file_keys:
                if k in f:
                    brk = True
                    break
            if brk:
                continue
            for e in ext:
                if f.lower().endswith(e):
                    paths[e].append(os.path.join(d, f))
                    break
    return paths


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
        GDRIVE: {
            "base_root": request.config.getoption(GDRIVE_PATH_OPT),
            "use_id": True,
            "is_id_prefix": True,
            "use_underscores": False,
        },
        GCS: {
            "base_root": PYTEST_GCS_PROVIDER_BASE_ROOT,
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

    root = posixpath.join(root, path).strip("/")
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
def s3_vstream_path(request):
    if not is_opt_true(request, S3_OPT):
        pytest.skip()
        return

    path = f"{PYTEST_S3_PROVIDER_BASE_ROOT}vstream_test"
    yield path


@pytest.fixture(scope="session")
def gcs_creds():
    return os.environ.get(ENV_GOOGLE_APPLICATION_CREDENTIALS, None)


@pytest.fixture(scope="session")
def gdrive_creds():
    client_id = os.environ.get(ENV_GDRIVE_CLIENT_ID)
    client_secret = os.environ.get(ENV_GDRIVE_CLIENT_SECRET)
    refresh_token = os.environ.get(ENV_GDRIVE_REFRESH_TOKEN)
    creds = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    return creds


@pytest.fixture
def gcs_path(request, gcs_creds):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip()
        return

    path = _get_storage_path(request, GCS)
    GCSProvider(path, token=gcs_creds).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        GCSProvider(path, token=gcs_creds).clear()


@pytest.fixture
def gcs_vstream_path(request):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip()
        return

    path = f"{PYTEST_GCS_PROVIDER_BASE_ROOT}vstream_test"
    yield path


@pytest.fixture
def gdrive_path(request, gdrive_creds):
    if not is_opt_true(request, GDRIVE_OPT):
        pytest.skip()
        return

    path = _get_storage_path(request, GDRIVE, with_current_test_name=False)
    GDriveProvider(path, token=gdrive_creds).clear()

    yield path

    if not is_opt_true(request, KEEP_STORAGE_OPT):
        GDriveProvider(path, token=gdrive_creds).clear()


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
        try:
            hub.delete(path, force=True, large_ok=True, token=hub_cloud_dev_token)
        except Exception:
            # TODO: investigate flakey `BadRequestException:
            # Invalid Request. One or more request parameters is incorrect.`
            # (on windows 3.8 only)
            pass


@pytest.fixture
def hub_cloud_vstream_path(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()
        return

    path = f"{PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT}vstream_test_dataset"

    yield path


@pytest.fixture
def cat_path():
    """Path to a cat image in the dummy data folder. Expected shape: (900, 900, 3)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "cat.jpeg")


@pytest.fixture
def flower_path():
    """Path to a flower image in the dummy data folder. Expected shape: (513, 464, 4)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "flower.png")


@pytest.fixture
def color_image_paths():
    base = get_dummy_data_path("images")
    paths = {
        "jpeg": os.path.join(base, "dog2.jpg"),
    }
    return paths


@pytest.fixture
def grayscale_image_paths():
    base = get_dummy_data_path("images")
    paths = {
        "jpeg": os.path.join(base, "hopper_gray.jpg"),
    }
    return paths


@pytest.fixture(scope="session")
def compressed_image_paths():
    paths = {
        "webp": "beach.webp",
        "gif": "boat.gif",
        "bmp": "car.bmp",
        "jpeg": ["cat.jpeg", "dog1.jpg", "dog2.jpg", "car.jpg"],
        "wmf": "crown.wmf",
        "dib": "dog.dib",
        "tiff": ["field.tiff", "field.tif"],
        "png": "flower.png",
        "ico": "sample_ico.ico",
        "jpeg2000": "sample_jpeg2000.jp2",
        "pcx": "sample_pcx.pcx",
        "ppm": "sample_ppm.ppm",
        "sgi": "sample_sgi.sgi",
        "tga": "sample_tga.tga",
        "xbm": "sample_xbm.xbm",
    }

    paths = {k: ([v] if isinstance(v, str) else v) for k, v in paths.items()}

    parent = get_dummy_data_path("images")
    for k in paths:
        paths[k] = [os.path.join(parent, p) for p in paths[k]]

    # Since we implement our own meta data reading for jpegs and pngs,
    # we test against images from PIL repo to cover all edge cases.
    pil_image_paths = _download_pil_test_images()
    paths["jpeg"] += pil_image_paths[".jpg"]
    paths["png"] += pil_image_paths[".png"]
    hub_test_images = _download_hub_test_images()
    paths["jpeg"] += hub_test_images
    yield paths


@pytest.fixture
def corrupt_image_paths():
    paths = {"jpeg": "corrupt_jpeg.jpeg", "png": "corrupt_png.png"}

    parent = get_dummy_data_path("images")
    for k in paths:
        paths[k] = os.path.join(parent, paths[k])

    return paths


@pytest.fixture
def audio_paths():
    if sys.platform.startswith("linux") and sys.version_info[:2] == (3, 6):  # FixMe
        pytest.skip()
        return
    paths = {"mp3": "samplemp3.mp3", "flac": "sampleflac.flac", "wav": "samplewav.wav"}

    parent = get_dummy_data_path("audio")
    for k in paths:
        paths[k] = os.path.join(parent, paths[k])

    return paths


@pytest.fixture
def video_paths():
    paths = {
        "mp4": ["samplemp4.mp4"],
        "mkv": ["samplemkv.mkv"],
        "avi": ["sampleavi.avi"],
    }

    parent = get_dummy_data_path("video")
    for k in paths:
        paths[k] = [os.path.join(parent, fname) for fname in paths[k]]
    paths["mp4"] += _download_hub_test_videos()

    return paths


@pytest.fixture
def point_cloud_paths():
    paths = {
        "las": "point_cloud.las",
    }

    parent = get_dummy_data_path("point_cloud")
    for k in paths:
        paths[k] = os.path.join(parent, paths[k])
    return paths


@pytest.fixture
def vstream_path(request):
    """Used with parametrize to use all video stream test datasets."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def path(request):
    """Used with parametrize to get all dataset paths."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def hub_token(request):
    """Used with parametrize to get hub_cloud_dev_token if hub-cloud option is True else None"""
    if is_opt_true(request, HUB_CLOUD_OPT):
        return request.getfixturevalue(request.param)
    return None
