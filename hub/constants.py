import numpy as np

BYTE_PADDING = b"\0"

# number of bytes per unit
B = 1
KB = 1000 * B
MB = 1000 * KB
GB = 1000 * MB

DEFAULT_HTYPE = "generic"

SUPPORTED_COMPRESSIONS = ["png", "jpeg", None]

# used for htypes. if an htype uses this as the default, that means the user needs to specify themselves a value
REQUIRE_USER_SPECIFICATION = "require_user"

# used instead of `None` for setting argument defaults. helpful for `REQUIRE_USER_SPECIFICATION` enforcement
UNSPECIFIED = "unspecified"

# If `True`  compression format has to be the same between samples in the same tensor.
# If `False` compression format can   be different between samples in the same tensor.
USE_UNIFORM_COMPRESSION_PER_SAMPLE = True

SUPPORTED_MODES = ["r", "a"]

DEFAULT_MAX_CHUNK_SIZE = 32 * MB

MIN_FIRST_CACHE_SIZE = 32 * MB
MIN_SECOND_CACHE_SIZE = 160 * MB

# without MB multiplication, meant for the Dataset API that takes cache size in MBs
DEFAULT_MEMORY_CACHE_SIZE = 256
DEFAULT_LOCAL_CACHE_SIZE = 0


DATASET_META_FILENAME = "dataset_meta.json"
TENSOR_META_FILENAME = "tensor_meta.json"
META_ENCODING = "utf8"

CHUNKS_FOLDER = "chunks"

CHUNK_EXTENSION = "npz"
ENCODED_CHUNK_NAMES_FOLDER = "chunks_index"
# unsharded naming will help with backwards compatibility
ENCODED_CHUNK_NAMES_FILENAME = f"unsharded.{CHUNK_EXTENSION}"

ENCODING_DTYPE = np.uint32
# caclulate the number of bits to shift right when converting a 128-bit uuid into `ENCODING_DTYPE`
UUID_SHIFT_AMOUNT = 128 - (8 * ENCODING_DTYPE(1).itemsize)

PYTEST_MEMORY_PROVIDER_BASE_ROOT = "mem://hub_pytest"
PYTEST_LOCAL_PROVIDER_BASE_ROOT = "/tmp/hub_pytest/"  # TODO: may fail for windows
PYTEST_S3_PROVIDER_BASE_ROOT = "s3://hub-2.0-tests/"
