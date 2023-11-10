import deeplake
import pytest


@pytest.mark.parametrize("chunk_compression", [None, "lz4"])
def test_tag(memory_ds, chunk_compression):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="tag", chunk_compression=chunk_compression)
        ds.abc.append("a")
        ds.abc.append(["a", "b"])
        ds.abc.extend(["a", ["b", "c"]])

    assert ds.abc.data()["value"] == [["a"], ["a", "b"], ["a"], ["b", "c"]]
