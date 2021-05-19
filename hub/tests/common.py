import pytest
import os
from uuid import uuid1

from hub.constants import B, KB, MB


SESSION_ID = str(uuid1())

TENSOR_KEY = "tensor"

SHAPE_PARAM = "shape"
NUM_BATCHES_PARAM = "num_batches"
DTYPE_PARAM = "dtype"
CHUNK_SIZE_PARAM = "chunk_size"


NUM_BATCHES = (1,)


CHUNK_SIZES = (
    1 * B,
    8 * B,
    1 * KB,
    1 * MB,
    16 * MB,
)


DTYPES = (
    "uint8",
    "int64",
    "float64",
    "bool",
)


parametrize_chunk_sizes = pytest.mark.parametrize(CHUNK_SIZE_PARAM, CHUNK_SIZES)
parametrize_dtypes = pytest.mark.parametrize(DTYPE_PARAM, DTYPES)
parametrize_num_batches = pytest.mark.parametrize(NUM_BATCHES_PARAM, NUM_BATCHES)


def current_test_name(with_id: bool, is_id_prefix: bool) -> str:
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    test_name = full_name.split("::")[1]
    output = os.path.join(test_file, test_name)
    if with_id:
        if is_id_prefix:
            return os.path.join(SESSION_ID, output)
        return os.path.join(output, SESSION_ID)
    return output
