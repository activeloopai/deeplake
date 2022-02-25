import numpy as np


def test_rechunk(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        for _ in range(10):
            ds.abc.append(np.ones((10, 10)))
        for i in range(5, 10):
            ds.abc[i] = np.ones((1000, 1000))

        for i in range(10):
            target = np.ones((10, 10)) if i < 5 else np.ones((1000, 1000))
            np.testing.assert_array_equal(ds.abc[i].numpy(), target)

        original_num_chunks = ds.abc.chunk_engine.num_chunks
        assert original_num_chunks == 1
        ds.rechunk()
        new_num_chunks = ds.abc.chunk_engine.num_chunks
        assert new_num_chunks == 3

        for i in range(10):
            target = np.ones((10, 10)) if i < 5 else np.ones((1000, 1000))
            np.testing.assert_array_equal(ds.abc[i].numpy(), target)

        ds.create_tensor("xyz")
        for _ in range(10):
            ds.xyz.append(np.ones((1000, 1000)))
        for i in range(10):
            ds.xyz[i] = np.ones((100, 100))

        original_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert original_num_chunks == 5
        ds.rechunk("xyz")
        new_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert new_num_chunks == 1
