BYTE_PADDING = b"\0"

# number of bytes per unit
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

MIN_MEMORY_CACHE_SIZE = 32 * MB
MIN_LOCAL_CACHE_SIZE = 160 * MB

CHUNKS_FOLDER = "chunks"
META_FILENAME = "meta.json"
INDEX_MAP_FILENAME = "index_map.json"

PYTEST_MEMORY_PROVIDER_BASE_ROOT = "hub_pytest"
PYTEST_LOCAL_PROVIDER_BASE_ROOT = "/tmp/hub_pytest/"
PYTEST_S3_PROVIDER_BASE_ROOT = "s3://hub-2.0-tests/"
