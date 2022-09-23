import deeplake
import numpy as np
import pytest

image_compressions = ["mpo", "fli"]


@pytest.mark.parametrize("compression", image_compressions)
def test_array(compression, compressed_image_paths):
    # TODO: check dtypes and no information loss
    array = np.array(deeplake.read(compressed_image_paths[compression][0]))
    arr = np.array(array)
    for c in arr:
        if c == "fli":
            assert arr.shape[-1] == 3
            assert arr.shape[0] == 480
            assert arr.dtype == "uint8"
        elif c == "mpo":
            assert arr.shape[-1] == 128
            assert arr.shape[0] == 128
            assert arr.dtype == "uint8"
