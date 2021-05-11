import numpy as np
import pickle

from hub.core.chunk_engine import write_array, read_array
from hub.core.chunk_engine.util import normalize_and_batchify_shape, get_meta_key
from hub.core.storage import MemoryProvider


ROOT = "PYTEST_TENSOR_COLLECTION"
STORAGE_PROVIDERS = (MemoryProvider(ROOT),)


CHUNK_SIZES = (
    128,
    4096,
    16000000,
)


DTYPES = (
    "uint8",
    "int64",
    "float64",
    "bool",
)


def get_min_shape(batch):
    return tuple(np.minimum.reduce([sample.shape for sample in batch]))


def get_max_shape(batch):
    return tuple(np.maximum.reduce([sample.shape for sample in batch]))


def run_engine_test(arrays, storage, batched, chunk_size):
    storage.clear()
    tensor_key = "tensor"

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            tensor_key,
            chunk_size,
            storage,
            batched=batched,
        )

        # TODO: make sure there is no more than 1 incomplete chunk at a time.

        # writing implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        a_out = read_array(tensor_key, storage)

        meta_key = get_meta_key(tensor_key)
        assert meta_key in storage, "Meta was not found."
        meta = pickle.loads(storage[meta_key])

        assert_meta_is_valid(
            meta,
            {
                "chunk_size": chunk_size,
                "length": a_in.shape[0],
                "dtype": a_in.dtype.name,
                "min_shape": get_min_shape(a_in),
                "max_shape": get_max_shape(a_in),
            },
        )

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i

    storage.clear()


def get_random_array(shape, dtype):
    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        return np.random.uniform(shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(shape)
        return a > 0.5


def assert_meta_is_valid(meta, expected_meta):
    for k, v in expected_meta.items():
        assert k in meta
        assert v == meta[k]
