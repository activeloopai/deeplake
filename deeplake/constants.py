import os
import sys

import numpy as np


BYTE_PADDING = b"\0"

# number of bytes per unit
B = 1
KB = 1000 * B
MB = 1000 * KB
GB = 1000 * MB

DEFAULT_HTYPE = "generic"

# used for requiring the user to specify a value for htype properties. notates that the htype property has no default.
REQUIRE_USER_SPECIFICATION = "require_user_specification"

# used for `REQUIRE_USER_SPECIFICATION` enforcement. this should be used instead of `None` for default user method arguments.
UNSPECIFIED = "unspecified"

SUPPORTED_MODES = ["r", "a"]

# used to show chunk size
RANDOM_CHUNK_SIZE = 8 * MB
# used to show variation between normal chunk size and maximum chunk size
RANDOM_MINIMAL_CHUNK_SIZE = 2 * MB
# used to show maximum chunk size allowed to have during random update operation
RANDOM_MAX_ALLOWED_CHUNK_SIZE = RANDOM_CHUNK_SIZE + RANDOM_MINIMAL_CHUNK_SIZE

# min chunk size is always half of `DEFAULT_MAX_CHUNK_SIZE`
DEFAULT_MAX_CHUNK_SIZE = 16 * MB

DEFAULT_TILING_THRESHOLD = 16 * MB  # Note: set to -1 to disable tiling

MIN_FIRST_CACHE_SIZE = 32 * MB
MIN_SECOND_CACHE_SIZE = 160 * MB

# without MB multiplication, meant for the dataset API that takes cache size in MBs
DEFAULT_MEMORY_CACHE_SIZE = 2000
DEFAULT_LOCAL_CACHE_SIZE = 0

# maximum allowable size before `large_ok` must be passed to dataset delete methods
DELETE_SAFETY_SIZE = 1 * GB

# meta is deeplake-defined information, necessary for Deep Lake Datasets/Tensors to function
DATASET_META_FILENAME = "dataset_meta.json"
TENSOR_META_FILENAME = "tensor_meta.json"

# info is user-defined information, entirely optional. may be used by the visualizer
DATASET_INFO_FILENAME = "dataset_info.json"
TENSOR_INFO_FILENAME = "tensor_info.json"

COMMIT_INFO_FILENAME = "commit_info.json"
DATASET_LOCK_FILENAME = "dataset_lock.lock"
DATASET_DIFF_FILENAME = "dataset_diff"
TENSOR_COMMIT_CHUNK_MAP_FILENAME = "chunk_set"
TENSOR_COMMIT_DIFF_FILENAME = "commit_diff"
TIMESTAMP_FILENAME = "local_download_timestamp"


DATASET_LOCK_UPDATE_INTERVAL = 120  # seconds
DATASET_LOCK_VALIDITY = 300  # seconds
LOCK_VERIFY_INTERVAL = 0.5  # seconds

META_ENCODING = "utf8"

CHUNKS_FOLDER = "chunks"
ENCODED_TILE_NAMES_FOLDER = "tiles_index"
ENCODED_CREDS_FOLDER = "creds_index"
ENCODED_CHUNK_NAMES_FOLDER = "chunks_index"
ENCODED_SEQUENCE_NAMES_FOLDER = "sequence_index"
ENCODED_PAD_NAMES_FOLDER = "pad_index"

# unsharded naming will help with backwards compatibility
UNSHARDED_ENCODER_FILENAME = "unsharded"

ENCODING_DTYPE = np.uint32

# environment variables
ENV_HUB_DEV_USERNAME = "ACTIVELOOP_HUB_USERNAME"
ENV_HUB_DEV_PASSWORD = "ACTIVELOOP_HUB_PASSWORD"

ENV_HUB_DEV_TOKEN = "ACTIVELOOP_HUB_TOKEN"

ENV_HUB_DEV_MANAGED_CREDS_KEY = "ACTIVELOOP_HUB_MANAGED_CREDS_KEY"

ENV_KAGGLE_USERNAME = "KAGGLE_USERNAME"
ENV_KAGGLE_KEY = "KAGGLE_KEY"

ENV_GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

ENV_AZURE_CLIENT_ID = "AZURE_CLIENT_ID"
ENV_AZURE_TENANT_ID = "AZURE_TENANT_ID"
ENV_AZURE_CLIENT_SECRET = "AZURE_CLIENT_SECRET"

ENV_AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
ENV_AWS_SECRETS_ACCESS_KEY = "AWS_SECRETS_ACCESS_KEY"
ENV_AWS_ENDPOINT_URL = "ENDPOINT_URL"

ENV_GDRIVE_CLIENT_ID = "GDRIVE_CLIENT_ID"
ENV_GDRIVE_CLIENT_SECRET = "GDRIVE_CLIENT_SECRET"
ENV_GDRIVE_REFRESH_TOKEN = "GDRIVE_REFRESH_TOKEN"

HUB_CLOUD_DEV_USERNAME = os.getenv(ENV_HUB_DEV_USERNAME)  # type: ignore
HUB_CLOUD_DEV_PASSWORD = os.getenv(ENV_HUB_DEV_PASSWORD)

# dataset base roots for pytests
PYTEST_MEMORY_PROVIDER_BASE_ROOT = "mem://hub_pytest"
PYTEST_LOCAL_PROVIDER_BASE_ROOT = "./hub_pytest/"
PYTEST_S3_PROVIDER_BASE_ROOT = "s3://deeplake-tests/"
PYTEST_GCS_PROVIDER_BASE_ROOT = "gcs://deeplake-tests/"
PYTEST_AZURE_PROVIDER_BASE_ROOT = "az://activeloopgen2/deeplake-tests/"
PYTEST_GDRIVE_PROVIDER_BASE_ROOT = "gdrive://hubtest"
PYTEST_HUB_CLOUD_PROVIDER_BASE_ROOT = (
    None if HUB_CLOUD_DEV_USERNAME is None else f"hub://{HUB_CLOUD_DEV_USERNAME}/"
)

# pytest options
MEMORY_OPT = "--memory-skip"
LOCAL_OPT = "--local"
S3_OPT = "--s3"
GCS_OPT = "--gcs"
AZURE_OPT = "--azure"
GDRIVE_OPT = "--gdrive"
HUB_CLOUD_OPT = "--hub-cloud"
S3_PATH_OPT = "--s3-path"
GDRIVE_PATH_OPT = "--gdrive-path"
KEEP_STORAGE_OPT = "--keep-storage"
KAGGLE_OPT = "--kaggle"

EMERGENCY_STORAGE_PATH = "/tmp/emergency_storage"
LOCAL_CACHE_PREFIX = "~/.activeloop/cache"
DOWNLOAD_MANAGED_PATH_SUFFIX = "__local-managed-entry__"

# used to identify the first commit so its data will not be in similar directory structure to the rest
FIRST_COMMIT_ID = "firstdbf9474d461a19e9333c2fd19b46115348f"
VERSION_CONTROL_INFO_FILENAME_OLD = "version_control_info"
VERSION_CONTROL_INFO_FILENAME = "version_control_info.json"
VERSION_CONTROL_INFO_LOCK_FILENAME = "version_control_info.lock"

LINKED_CREDS_FILENAME = "linked_creds.json"
LINKED_CREDS_LOCK_FILENAME = "linked_creds.lock"


# when cache is full upto this threshold, it will start suggesting new indexes intelligently based on existing contents
INTELLIGENT_SHUFFLING_THRESHOLD = 0.8

TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL = 5  # seconds


# If True, and if the rest of the dataset is in color (3D), then reshape a grayscale image by appending a 1 to its shape.
CONVERT_GRAYSCALE = True

PARTIAL_NUM_SAMPLES = 0.5
FAST_EXTEND_BAIL = -1

QUERIES_FILENAME = "queries.json"
QUERIES_LOCK_FILENAME = "queries.lock"

ALL_CLOUD_PREFIXES = (
    "s3://",
    "gcs://",
    "gcp://",
    "gs://",
    "az://",
    "azure://",
    "gdrive://",
)

_ENABLE_HUB_SUB_DATASETS = False
_ENABLE_RANDOM_ASSIGNMENT = True

# Frequency for sending progress events and writing to vds
QUERY_PROGRESS_UPDATE_FREQUENCY = 5  # seconds

PYTORCH_DATALOADER_TIMEOUT = 30  # seconds

_NO_LINK_UPDATE = "___!@#_no_link_update_###"

SAMPLE_INFO_TENSOR_MAX_CHUNK_SIZE = 4 * MB

DEFAULT_READONLY = (
    os.environ.get("HUB_DEFAULT_READONLY", "false").strip().lower() == "true"
)

_UNLINK_VIDEOS = False

WANDB_INTEGRATION_ENABLED = True
WANDB_JSON_FILENMAE = "wandb.json"

SHOW_ITERATION_WARNING = True
RETURN_DUMMY_DATA_FOR_DATALOADER = False

# Delay before spinner starts on time consuming functions (in seconds)
SPINNER_START_DELAY = 2

PYTEST_ENABLED = os.environ.get("DEEPLAKE_PYTEST_ENABLED", "").lower().strip() == "true"

SPINNER_ENABLED = not PYTEST_ENABLED

LOCKS_ENABLED = not PYTEST_ENABLED

# Rechunk after transform if average chunk size is less than
# this fraction of min chunk size
TRANSFORM_RECHUNK_AVG_SIZE_BOUND = 0.1

TIME_INTERVAL_FOR_CUDA_MEMORY_CLEANING = 10 * 60

MAX_TENSORS_IN_SHUFFLE_BUFFER = 32000

# Transform cache sizes
DEFAULT_TRANSFORM_SAMPLE_CACHE_SIZE = 16
TRANSFORM_CHUNK_CACHE_SIZE = 64 * MB

DEFAULT_VECTORSTORE_DEEPLAKE_PATH = "./deeplake_vector_store"
MAX_VECTORSTORE_INGESTION_RETRY_ATTEMPTS = 5
MAX_CHECKPOINTING_INTERVAL = 100000
VECTORSTORE_EXTEND_MAX_SIZE = 20000
VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE = {"image": 2000}
DEFAULT_VECTORSTORE_TENSORS = [
    {
        "name": "text",
        "htype": "text",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "metadata",
        "htype": "json",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "embedding",
        "htype": "embedding",
        "dtype": np.float32,
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": True,
        "max_chunk_size": 64 * MB,
    },
    {
        "name": "id",
        "htype": "text",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
]

DEFAULT_QUERIES_VECTORSTORE_TENSORS = [
    {
        "name": "text",
        "htype": "text",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "metadata",
        "htype": "json",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "embedding",
        "htype": "embedding",
        "dtype": np.float32,
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": True,
        "max_chunk_size": 64 * MB,
    },
    {
        "name": "id",
        "htype": "text",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "deep_memory_top_10",
        "htype": "json",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "deep_memory_recall",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "vector_search_top_10",
        "htype": "json",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
    {
        "name": "vector_search_recall",
        "create_id_tensor": False,
        "create_sample_info_tensor": False,
        "create_shape_tensor": False,
    },
]

VIEW_SUMMARY_SAFE_LIMIT = 10000

# openai constants
OPENAI_TOKEN_SIZE = 2 * B
OPENAI_ADA_MAX_TOKENS_PER_MINUTE = 1000000
MAX_BYTES_PER_MINUTE = (
    0.9 * OPENAI_TOKEN_SIZE * OPENAI_ADA_MAX_TOKENS_PER_MINUTE
)  # 0.9 is a safety factor
TARGET_BYTE_SIZE = 10000

# Maximum default message length for saving query views
QUERY_MESSAGE_MAX_SIZE = 1000

DEFAULT_VECTORSTORE_DISTANCE_METRIC = "COS"
DEFAULT_DEEPMEMORY_DISTANCE_METRIC = "deepmemory_distance"

DEFAULT_VECTORSTORE_INDEX_PARAMS = {
    "threshold": -1,
    "distance_metric": DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    "additional_params": {
        "efConstruction": 600,
        "M": 32,
        "partitions": 1,
    },
}
VECTORSTORE_EXTEND_BATCH_SIZE = 500

_INDEX_OPERATION_MAPPING = {
    "ADD": 1,
    "REMOVE": 2,
    "UPDATE": 3,
}


DEFAULT_RATE_LIMITER_KEY_TO_VALUE = {
    "enabled": False,
    "bytes_per_minute": MAX_BYTES_PER_MINUTE,
    "batch_byte_size": TARGET_BYTE_SIZE,
}

# Size of dataset view to expose as indra dataset wrapper.
INDRA_DATASET_SAMPLES_THRESHOLD = 10000000

USE_INDRA = os.environ.get("DEEPLAKE_USE_INDRA", "false").strip().lower() == "true"
