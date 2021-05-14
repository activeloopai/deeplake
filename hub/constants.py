BYTE_PADDING = b"\0"

# helpful for specifying cache sizes, that are specified in MB. For eg to specify 32MB cache use 32 * MB
MB = 1024 * 1024

CHUNKS_FOLDER = "chunks"
META_FILENAME = "meta.json"
INDEX_MAP_FILENAME = "index_map.json"

PYTEST_MEMORY_PROVIDER_BASE_ROOT = "PYTEST_TMPDIR/memory_storage_provider/"
PYTEST_LOCAL_PROVIDER_BASE_ROOT = "PYTEST_TMPDIR/local_storage_provider/"
PYTEST_S3_PROVIDER_BASE_ROOT = "snark-test/hub-2.0/"  # TODO: new bucket
