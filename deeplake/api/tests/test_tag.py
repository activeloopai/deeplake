import numpy as np
import deeplake
import pytest


@pytest.mark.parametrize("chunk_compression", [None, "lz4"])
def test_tag(memory_ds, chunk_compression):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="tag", chunk_compression=chunk_compression)
        ds.abc.append("a")
        ds.abc.append(["a", "b"])
        ds.abc.extend(["a", ["b", "c"]])

    np.testing.assert_array_equal(ds.abc.shapes(), np.array([[1], [2], [1], [2]]))
    assert ds.abc.data()["value"] == [["a"], ["a", "b"], ["a"], ["b", "c"]]
