from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO
import os
import pathlib
from typing import List, Optional
from uuid import uuid4

import numpy as np
import posixpath
import pytest

from hub.constants import KB, MB

SESSION_ID = str(uuid4())[:4]  # 4 ascii chars should be sufficient

_THIS_FILE = pathlib.Path(__file__).parent.absolute()
TENSOR_KEY = "tensor"

SHAPE_PARAM = "shape"
NUM_BATCHES_PARAM = "num_batches"
DTYPE_PARAM = "dtype"
CHUNK_SIZE_PARAM = "chunk_size"
NUM_WORKERS_PARAM = "num_workers"

NUM_BATCHES = (1, 5)
NUM_WORKERS = (0, 1, 2, 4)

CHUNK_SIZES = (
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
parametrize_num_workers = pytest.mark.parametrize(NUM_WORKERS_PARAM, NUM_WORKERS)


def current_test_name() -> str:
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    test_name = full_name.split("::")[1]
    output = posixpath.join(test_file, test_name)
    return output


def get_dummy_data_path(subpath: str = ""):
    return os.path.join(_THIS_FILE, "dummy_data" + os.sep, subpath)


def get_actual_compression_from_buffer(buffer: memoryview) -> Optional[str]:
    """Helpful for checking if actual compression matches expected."""
    try:
        bio = BytesIO(buffer)
        img = Image.open(bio)
        return img.format.lower()

    except UnidentifiedImageError:
        return None


def assert_array_lists_equal(l1: List[np.ndarray], l2: List[np.ndarray]):
    """Assert that two lists of numpy arrays are equal"""
    assert len(l1) == len(l2), (len(l1), len(l2))
    for idx, (a1, a2) in enumerate(zip(l1, l2)):
        np.testing.assert_array_equal(a1, a2, err_msg=f"Array mismatch at index {idx}")


def is_opt_true(request, opt) -> bool:
    """Returns if the pytest option `opt` is True. Assumes `opt` is a boolean value."""
    return request.config.getoption(opt)


def assert_images_close(img1: np.ndarray, img2: np.ndarray, eps=0.5):
    """Helpful for testing images after lossy compression"""
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    err = np.sum((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    err /= np.prod(img1.shape) * 256
    assert err < eps, err
