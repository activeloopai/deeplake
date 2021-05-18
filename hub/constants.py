BYTE_PADDING = b"\0"

# number of bytes per unit
KB = 1000
MB = 1000 * KB
GB = 1000 * MB

MIN_FIRST_CACHE_SIZE = 32 * MB
MIN_SECOND_CACHE_SIZE = 160 * MB

CHUNKS_FOLDER = "chunks"
META_FILENAME = "meta.json"
INDEX_MAP_FILENAME = "index_map.json"

PYTEST_MEMORY_PROVIDER_BASE_ROOT = "hub_pytest"
PYTEST_LOCAL_PROVIDER_BASE_ROOT = "/tmp/hub_pytest/"
PYTEST_S3_PROVIDER_BASE_ROOT = "s3://hub-2.0-tests/"
