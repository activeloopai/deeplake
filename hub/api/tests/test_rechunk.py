import numpy as np
import random
import hub


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
        assert original_num_chunks == 3
        ds.rechunk()
        new_num_chunks = ds.abc.chunk_engine.num_chunks
        assert new_num_chunks == 3

        assert len(ds.abc) == 10
        for i in range(10):
            target = np.ones((10, 10)) if i < 5 else np.ones((1000, 1000))
            np.testing.assert_array_equal(ds.abc[i].numpy(), target)
        assert len(ds.abc) == 10

        ds.create_tensor("xyz")
        for _ in range(10):
            ds.xyz.append(np.ones((1000, 1000)))

        assert len(ds.xyz) == 10
        for i in range(10):
            ds.xyz[i] = np.ones((100, 100))

        original_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert original_num_chunks == 1
        assert len(ds.xyz) == 10
        ds.rechunk("xyz")
        new_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert new_num_chunks == 1

        ds.create_tensor("compr", chunk_compression="lz4")
        for _ in range(100):
            ds.compr.append(np.random.randint(0, 255, size=(175, 350, 3)))

        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 3, size=(10, 10, 10))
        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 255, size=(175, 350, 3))
        assert len(ds.compr) == 100


def test_rechunk_2(local_ds):
    with local_ds as ds:
        ds.create_tensor("compr", dtype="int64")
        for _ in range(100):
            ds.compr.append(np.random.randint(0, 255, size=(175, 350, 3)))

        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 3, size=(10, 10, 10))
        assert len(ds.compr) == 100
        assert ds.compr.chunk_engine.num_chunks == 1


def test_rechunk_3(local_ds):
    NUM_TEST_SAMPLES = 100
    hub.constants._ENABLE_RANDOM_ASSIGNMENT = True
    test_sample = np.random.randint(0, 255, size=(600, 600, 3), dtype=np.uint8)
    with local_ds as ds:
        ds.create_tensor("test", dtype="uint8")
        r = list(range(NUM_TEST_SAMPLES))
        random.seed(20)
        random.shuffle(r)
        for i in r:
            ds.test[i] = test_sample
