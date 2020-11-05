from hub.store.cache import Cache
import zarr
import time
import posixpath


class SlowStore(zarr.MemoryStore):
    def __init__(self, **kwargs):
        super(SlowStore, self).__init__(**kwargs)

    def __getitem__(self, key, **kwargs):
        time.sleep(0.001)
        return super(SlowStore, self).__getitem__(key, **kwargs)

    def __setitem__(self, key, value, **kwargs):
        super(SlowStore, self).__setitem__(key, value, **kwargs)


def test_cache():
    store = SlowStore()
    store = Cache(store, max_size=1000000)

    for i in range(10):
        z = zarr.zeros(
            (1000, 1000),
            chunks=(100, 100),
            path=posixpath.realpath(f"./data/test/test_cache/first{i}"),
            store=store,
            overwrite=True,
        )

        z[...] = i
        store.invalidate()

        t1 = time.time()
        z[...]
        t2 = time.time()
        z[...]
        t3 = time.time()
        assert z[0, 0] == i
        # print(t2 - t1, t3 - t2)
        assert t2 - t1 > t3 - t2
    store.commit()


if __name__ == "__main__":
    test_cache()