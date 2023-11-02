import deeplake


def test_tag(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="tag")
        ds.abc.extend(["a", "b", "c"])

    assert ds.abc.numpy().tolist() == [["a"], ["b"], ["c"]]
    assert ds.abc.data()["value"] == ["a", "b", "c"]
