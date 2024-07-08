import os
import pickle
import posixpath
import pytest
import sys

import deeplake
from deeplake.constants import (
    HUB_CLOUD_OPT,
    KEEP_STORAGE_OPT,
    LOCAL_OPT,
    MEMORY_OPT,
    PYTEST_GCS_PROVIDER_BASE_ROOT,
    PYTEST_AZURE_PROVIDER_BASE_ROOT,
    PYTEST_S3_PROVIDER_BASE_ROOT,
    PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT,
    PYTEST_LOCAL_PROVIDER_BASE_ROOT,
    PYTEST_MEMORY_PROVIDER_BASE_ROOT,
    S3_OPT,
    GCS_OPT,
    AZURE_OPT,
    GDRIVE_OPT,
    S3_PATH_OPT,
    GDRIVE_PATH_OPT,
    ENV_GOOGLE_APPLICATION_CREDENTIALS,
    ENV_GDRIVE_CLIENT_ID,
    ENV_GDRIVE_CLIENT_SECRET,
    ENV_GDRIVE_REFRESH_TOKEN,
    HUB_CLOUD_DEV_USERNAME,
    ENV_AZURE_CLIENT_ID,
    ENV_AZURE_TENANT_ID,
    ENV_AZURE_CLIENT_SECRET,
    ENV_AWS_ACCESS_KEY,
    ENV_AWS_SECRETS_ACCESS_KEY,
    ENV_AWS_ENDPOINT_URL,
)
from deeplake import VectorStore
from deeplake.client.client import DeepMemoryBackendClient
from deeplake.core.storage.gcs import GCSProvider
from deeplake.core.storage.google_drive import GDriveProvider
from deeplake.util.storage import storage_provider_from_hub_path
from deeplake.core.storage.s3 import S3Provider
from deeplake.core.storage.local import LocalProvider
from deeplake.core.storage.azure import AzureProvider
from deeplake.core.vectorstore import utils
from deeplake.tests.common import (
    SESSION_ID,
    current_test_name,
    get_dummy_data_path,
    is_opt_true,
)


MEMORY = "memory"
LOCAL = "local"
S3 = "s3"
GDRIVE = "gdrive"
GCS = "gcs"
AZURE = "azure"
HUB_CLOUD = "hub_cloud"

REPO_ROOT = os.path.abspath(".")
while REPO_ROOT != os.path.dirname(REPO_ROOT):
    if os.path.isfile(os.path.join(REPO_ROOT, "LICENSE")):
        break
    REPO_ROOT = os.path.dirname(REPO_ROOT)
assert REPO_ROOT != "/"

## .test_resources should always be in the root of the repo, regardless of where the tests were ran from
_GIT_CLONE_CACHE_DIR = os.path.join(REPO_ROOT, ".test_resources")

_HUB_TEST_RESOURCES_URL = "https://www.github.com/activeloopai/hub-test-resources.git"
_PILLOW_URL = "https://www.github.com/python-pillow/Pillow.git"
_MMDET_URL = "https://www.github.com/open-mmlab/mmdetection.git"


def _repo_name_from_git_url(url):
    repo_name = posixpath.split(url)[-1]
    repo_name = repo_name.split("@", 1)[0]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name


def _git_clone_with_branch(branch_name, url):
    _repo_name = _repo_name_from_git_url(url)
    cached_dir = _GIT_CLONE_CACHE_DIR + "/" + _repo_name
    if not os.path.isdir(cached_dir):
        if not os.path.isdir(_GIT_CLONE_CACHE_DIR):
            os.mkdir(_GIT_CLONE_CACHE_DIR)
        cwd = os.getcwd()
        os.chdir(_GIT_CLONE_CACHE_DIR)
        try:
            os.system(f"git clone -b {branch_name} {url}")
        finally:
            os.chdir(cwd)
    assert os.path.isdir(cached_dir)
    return cached_dir


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


def _download_hub_test_coco_data():
    path = _git_clone(_HUB_TEST_RESOURCES_URL)
    coco_images_path = path + "/coco/images"
    coco_annotations_path = path + "/coco/annotations"
    return {
        "images_directory": coco_images_path,
        "annotation_files": [
            os.path.join(coco_annotations_path, f)
            for f in os.listdir(coco_annotations_path)
        ],
    }


def _download_hub_test_yolo_data():
    path = _git_clone(_HUB_TEST_RESOURCES_URL)
    return {
        "data_directory": path + "/yolo/data",
        "class_names_file": path + "/yolo/classes.names",
        "data_directory_no_annotations": path + "/yolo/images_only",
        "annotations_directory": path + "/yolo/annotations_only",
        "data_directory_missing_annotations": path + "/yolo/data_missing_annotations",
        "data_directory_unsupported_annotations": path
        + "/yolo/data_unsupported_annotations",
    }


def _download_hub_test_dataframe_data():
    path = _git_clone(_HUB_TEST_RESOURCES_URL)
    return {
        "basic_dataframe_w_sanitize_path": path + "/dataframe/text_w_sanitization.txt",
        "dataframe_w_images_path": path + "/dataframe/csv_w_local_files.csv",
        "dataframe_w_bad_images_path": path + "/dataframe/csv_w_local_bad_file.csv",
        "images_basepath": path + "/dataframe/images",
    }


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
            # "/Tests/images/apng",
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
        AZURE: {
            "base_root": PYTEST_AZURE_PROVIDER_BASE_ROOT,
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
        pytest.skip(f"{MEMORY_OPT} flag not set")
        return

    # no need to clear memory paths
    return _get_storage_path(request, MEMORY)


@pytest.fixture
def local_path(request):
    if not is_opt_true(request, LOCAL_OPT):
        pytest.skip(f"{LOCAL_OPT} flag not set")
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
        pytest.skip(f"{S3_OPT} flag not set")
        return

    path = _get_storage_path(request, S3)
    S3Provider(path).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        S3Provider(path).clear()


@pytest.fixture
def s3_creds(request):
    if not is_opt_true(request, S3_OPT):
        pytest.skip(f"{S3_OPT} flag not set")
        return

    aws_access_key = os.environ.get(ENV_AWS_ACCESS_KEY)
    aws_secrets_key = os.environ.get(ENV_AWS_SECRETS_ACCESS_KEY)
    endpoint_url = os.environ.get(ENV_AWS_ENDPOINT_URL)
    creds = {
        "aws_access_key": aws_access_key,
        "aws_secrets_key": aws_secrets_key,
        "endpoint_url": endpoint_url,
    }
    return creds


@pytest.fixture
def s3_vstream_path(request):
    if not is_opt_true(request, S3_OPT):
        pytest.skip(f"{S3_OPT} flag not set")
        return

    path = f"{PYTEST_S3_PROVIDER_BASE_ROOT}vstream_test"
    yield path


@pytest.fixture(scope="session")
def gcs_creds():
    return os.environ.get(ENV_GOOGLE_APPLICATION_CREDENTIALS, None)


@pytest.fixture
def gcs_path(request, gcs_creds):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip(f"{GCS_OPT} flag not set")
        return

    path = _get_storage_path(request, GCS)
    GCSProvider(path, creds=gcs_creds).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        GCSProvider(path, creds=gcs_creds).clear()


@pytest.fixture
def gcs_vstream_path(request):
    if not is_opt_true(request, GCS_OPT):
        pytest.skip(f"{GCS_OPT} flag not set")
        return

    path = f"{PYTEST_GCS_PROVIDER_BASE_ROOT}vstream_test"
    yield path


@pytest.fixture
def azure_path(request):
    if not is_opt_true(request, AZURE_OPT):
        pytest.skip(f"{AZURE_OPT} flag not set")

    path = _get_storage_path(request, AZURE)
    AzureProvider(path).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        AzureProvider(path).clear()


@pytest.fixture
def azure_vstream_path(request):
    if not is_opt_true(request, AZURE_OPT):
        pytest.skip(f"{AZURE_OPT} flag not set")

    path = f"{PYTEST_AZURE_PROVIDER_BASE_ROOT}vstream_test"
    yield path


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
def azure_creds(request):
    if not is_opt_true(request, AZURE_OPT):
        pytest.skip(f"{AZURE_OPT} flag not set")
        return

    azure_client_id = os.environ.get(ENV_AZURE_CLIENT_ID)
    azure_tenant_id = os.environ.get(ENV_AZURE_TENANT_ID)
    azure_client_secret = os.environ.get(ENV_AZURE_CLIENT_SECRET)
    creds = {
        "azure_client_id": azure_client_id,
        "azure_tenant_id": azure_tenant_id,
        "azure_client_secret": azure_client_secret,
    }
    return creds


@pytest.fixture
def gdrive_path(request, gdrive_creds):
    if not is_opt_true(request, GDRIVE_OPT):
        pytest.skip(f"{GDRIVE_OPT} flag not set")
        return

    path = _get_storage_path(request, GDRIVE, with_current_test_name=False)
    GDriveProvider(path, token=gdrive_creds).clear()

    yield path

    if not is_opt_true(request, KEEP_STORAGE_OPT):
        GDriveProvider(path, token=gdrive_creds).clear()


@pytest.fixture
def hub_cloud_path(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")
        return

    path = _get_storage_path(request, HUB_CLOUD)
    storage_provider_from_hub_path(path, token=hub_cloud_dev_token).clear()

    yield path

    # clear storage unless flagged otherwise
    if not is_opt_true(request, KEEP_STORAGE_OPT):
        try:
            deeplake.delete(path, force=True, large_ok=True, token=hub_cloud_dev_token)
        except Exception:
            # TODO: investigate flakey `BadRequestException:
            # Invalid Request. One or more request parameters is incorrect.`
            # (on windows 3.8 only)
            pass


@pytest.fixture
def hub_cloud_vstream_path(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")
        return

    path = f"{PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT}vstream_test_dataset"

    yield path


@pytest.fixture
def corpus_query_relevances_copy(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")
        return

    corpus = _get_storage_path(request, HUB_CLOUD)
    query_vs = VectorStore(
        path=f"hub://{HUB_CLOUD_DEV_USERNAME}/deepmemory_test_queries",
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )
    queries = query_vs.dataset.text.data()["value"]
    relevance = query_vs.dataset.metadata.data()["value"]
    relevance = [rel["relevance"] for rel in relevance]

    deeplake.deepcopy(
        f"hub://{HUB_CLOUD_DEV_USERNAME}/test-deepmemory10",
        corpus,
        token=hub_cloud_dev_token,
        overwrite=True,
        runtime={"tensor_db": True},
    )

    queries_path = corpus + "_eval_queries"

    yield corpus, queries, relevance, queries_path

    delete_if_exists(corpus, hub_cloud_dev_token)
    delete_if_exists(queries_path, hub_cloud_dev_token)


@pytest.fixture
def deep_memory_local_dataset(request, hub_cloud_dev_token):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")
        return

    corpus_path = _get_storage_path(request, LOCAL)

    deeplake.deepcopy(
        f"hub://{HUB_CLOUD_DEV_USERNAME}/test-deepmemory10",
        corpus_path,
        token=hub_cloud_dev_token,
        overwrite=True,
    )

    yield corpus_path

    delete_if_exists(corpus_path, hub_cloud_dev_token)


def delete_if_exists(path, token):
    try:
        deeplake.delete(path, force=True, large_ok=True, token=token)
    except Exception:
        pass


@pytest.fixture
def corpus_query_pair_path(hub_cloud_dev_token):
    corpus = f"hub://{HUB_CLOUD_DEV_USERNAME}/deepmemory_test_corpus_managed_2"
    query = corpus + "_eval_queries"
    delete_if_exists(query, hub_cloud_dev_token)
    yield corpus, query

    delete_if_exists(query, hub_cloud_dev_token)


@pytest.fixture
def cat_path():
    """Path to a cat image in the dummy data folder. Expected shape: (900, 900, 3)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "cat.jpeg")


@pytest.fixture
def dog_path():
    """Path to a dog image in the dummy data folder. Expected shape: (323, 480, 3)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "dog2.jpg")


@pytest.fixture
def flower_path():
    """Path to a flower image in the dummy data folder. Expected shape: (513, 464, 4)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "flower.png")


@pytest.fixture
def hopper_gray_path():
    """Path to a grayscale hopper image in the dummy data folder. Expected shape: (512, 512)"""

    path = get_dummy_data_path("images")
    return os.path.join(path, "hopper_gray.jpg")


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


@pytest.fixture
def lfpw_links():
    """Mix of working and broken links to images from the LFPW dataset."""
    BROKEN_LINKS = [
        "https://cm1.theinsider.com/media/0/428/93/spl41194_011.0.0.0x0.636x912.jpeg",
        "https://cm1.theinsider.com/media/0/428/93/spl47823_060.0.0.0x0.633x912.jpeg",
        "https://cm1.theinsider.com/media/0/428/90/spl91520_012.0.0.0x0.636x912.jpeg",
        "https://blog.themavenreport.com/wp-content/uploads/2008/02/kimora_show_575.jpg",
        "https://cache.thephoenix.com/secure/uploadedImages/The_Phoenix/Movies/Reviews/FILM_Queen_6.jpg",
        "https://img2.timeinc.net/people/i/2008/features/theysaid/080331/kimora_lee_simmons400.jpg",
        "https://img2.timeinc.net/people/i/cbb/2008/04/05/kylieminogue.jpg",
        "https://i41.tinypic.com/2ih5b7q.png",
        "https://www.todoelmundo.org/archivos/99/imagenes/En_america.jpg",
        "https://image.toutlecine.com/photos/b/l/o/blood-diamond-2006-22-g.jpg",
    ]
    return BROKEN_LINKS


@pytest.fixture(scope="session")
def mmdet_path():
    return _git_clone_with_branch("dev-2.x", _MMDET_URL)


@pytest.fixture(scope="session")
def compressed_image_paths():
    paths = {
        "webp": "beach.webp",
        "fli": "hopper.fli",
        "mpo": "sugarshack.mpo",
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
        pytest.skip("Skipping audio tests on linux 3.6")
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
        "avi": ["sampleavi.avi", "tinyavi.avi"],
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
def mesh_paths():
    paths = {
        "ascii1": "mesh_ascii.ply",
        "ascii2": "mesh_ascii_2.ply",
        "bin": "mesh_bin.ply",
    }

    parent = get_dummy_data_path("mesh")
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
def dest_path(request):
    """Used with parametrize to get all dataset paths."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def creds(request):
    """Used with parametrize to get all dataset creds."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def hub_token(request):
    """Used with parametrize to get hub_cloud_dev_token if hub-cloud option is True else None"""
    if is_opt_true(request, HUB_CLOUD_OPT):
        return request.getfixturevalue(request.param)
    return None


@pytest.fixture(scope="session")
def coco_ingestion_data():
    return _download_hub_test_coco_data()


@pytest.fixture(scope="session")
def yolo_ingestion_data():
    return _download_hub_test_yolo_data()


@pytest.fixture(scope="session")
def dataframe_ingestion_data():
    return _download_hub_test_dataframe_data()


@pytest.fixture
def vector_store_hash_ids(request):
    if getattr(request, "param", True):
        return [f"{i}" for i in range(5)]


@pytest.fixture
def vector_store_row_ids(request):
    if getattr(request, "param", True):
        return [i for i in range(5)]


@pytest.fixture
def vector_store_filter_udf(request):
    def filter_udf(x):
        return x["metadata"].data()["value"] == {"a": 1}

    if getattr(request, "param", True):
        return filter_udf


@pytest.fixture
def vector_store_filters(request):
    if getattr(request, "param", True):
        return {"metadata": {"a": 1}}


@pytest.fixture
def vector_store_query(request):
    if getattr(request, "param", True):
        return "select * where metadata['a']==1"


@pytest.fixture
def jobs_list():
    parent = get_dummy_data_path("deep_memory")

    with open(os.path.join(parent, "jobs_list.txt"), "r") as f:
        jobs = f.read()
    return jobs


@pytest.fixture
def questions_embeddings_and_relevances():
    parent = get_dummy_data_path("deep_memory")

    with open(os.path.join(parent, "questions.pkl"), "rb") as f:
        questions = pickle.load(f)
    with open(os.path.join(parent, "questions_embeddings.pkl"), "rb") as f:
        questions_embeddings = pickle.load(f)
    with open(os.path.join(parent, "questions_relevances.pkl"), "rb") as f:
        question_relevances = pickle.load(f)
    return questions_embeddings, question_relevances, questions


@pytest.fixture
def testing_relevance_query_deepmemory():
    parent = get_dummy_data_path("deep_memory")
    with open(os.path.join(parent, "dm_rq.pkl"), "rb") as f:
        dm_rq = pickle.load(f)

    relevance = dm_rq["relevance"]
    query = dm_rq["query_embedding"]
    return relevance, query


@pytest.fixture
def job_id():
    return "65198efcd28df3238c49a849"


@pytest.fixture
def precomputed_jobs_list():
    parent = get_dummy_data_path("deep_memory")

    with open(os.path.join(parent, "precomputed_jobs_list.txt"), "r") as f:
        jobs = f.read()
    return jobs


@pytest.fixture
def local_dmv2_dataset(request, hub_cloud_dev_token):
    dmv2_path = f"hub://{HUB_CLOUD_DEV_USERNAME}/dmv2"

    local_cache_path = ".deepmemory_tests_cache/"
    if not os.path.exists(local_cache_path):
        os.mkdir(local_cache_path)

    dataset_cache_path = local_cache_path + "dmv2"
    if not os.path.exists(dataset_cache_path):
        deeplake.deepcopy(
            dmv2_path,
            dataset_cache_path,
            token=hub_cloud_dev_token,
            overwrite=True,
        )

    corpus = _get_storage_path(request, LOCAL)

    deeplake.deepcopy(
        dataset_cache_path,
        corpus,
        token=hub_cloud_dev_token,
        overwrite=True,
    )
    yield corpus

    delete_if_exists(corpus, hub_cloud_dev_token)


@pytest.fixture
def deepmemory_small_dataset_copy(request, hub_cloud_dev_token):
    dm_path = f"hub://{HUB_CLOUD_DEV_USERNAME}/tiny_dm_dataset"
    queries_path = f"hub://{HUB_CLOUD_DEV_USERNAME}/queries_vs"

    local_cache_path = ".deepmemory_tests_cache/"
    if not os.path.exists(local_cache_path):
        os.mkdir(local_cache_path)

    dataset_cache_path = local_cache_path + "tiny_dm_queries"
    if not os.path.exists(dataset_cache_path):
        deeplake.deepcopy(
            queries_path,
            dataset_cache_path,
            token=hub_cloud_dev_token,
            overwrite=True,
        )

    corpus = _get_storage_path(request, HUB_CLOUD)
    query_vs = VectorStore(
        path=dataset_cache_path,
    )
    queries = query_vs.dataset.text.data()["value"]
    relevance = query_vs.dataset.metadata.data()["value"]
    relevance = [rel["relevance"] for rel in relevance]

    deeplake.deepcopy(
        dm_path,
        corpus,
        token=hub_cloud_dev_token,
        overwrite=True,
        runtime={"tensor_db": True},
    )

    queries_path = corpus + "_eval_queries"

    yield corpus, queries, relevance, queries_path

    delete_if_exists(corpus, hub_cloud_dev_token)
    delete_if_exists(queries_path, hub_cloud_dev_token)
