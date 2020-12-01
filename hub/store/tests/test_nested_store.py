from hub.store.nested_store import NestedStore

import zarr


def test_nested_store():
    store = NestedStore(zarr.MemoryStore(), "hello")
    store["item"] = bytes("Hello World", "utf-8")
    assert store["item"] == bytes("Hello World", "utf-8")
    del store["item"]
    assert store.get("item") is None
    store["item1"] = bytes("Hello World 1", "utf-8")
    store["item2"] = bytes("Hello World 2", "utf-8")
    assert len(store) == 2
    assert tuple(store) == ("item1", "item2")
    try:
        store.close()
    except AttributeError as ex:
        assert "object has no attribute 'close'" in str(ex)


if __name__ == "__main__":
    test_nested_store()
