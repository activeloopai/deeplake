import deeplake
import numpy as np

from deeplake.core.distance_type import DistanceType
from deeplake.tests.common import requires_libdeeplake
from deeplake.tests.dataset_fixtures import local_auth_ds_generator
from deeplake.util.exceptions import ReadOnlyModeError, EmbeddingTensorPopError
import pytest
import warnings


@requires_libdeeplake
def test_index_management(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(200):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})
    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "cosine_similarity"
    assert es[0]["type"] == "hnsw"
    with pytest.raises(ValueError):
        deeplake_ds.embedding.create_vdb_index("hnsw_1")
    deeplake_ds.embedding.create_vdb_index("hnsw_2")
    assert len(es) == 2
    deeplake_ds.embedding.delete_vdb_index("hnsw_1")
    with pytest.raises(KeyError):
        deeplake_ds.embedding.delete_vdb_index("hnsw_3")
    deeplake_ds.embedding.delete_vdb_index("hnsw_2")
    assert len(deeplake_ds.embedding.get_vdb_indexes()) == 0
    deeplake_ds.read_only = True
    with pytest.raises(ReadOnlyModeError):
        deeplake_ds.embedding.create_vdb_index("hnsw_1")
    with pytest.raises(ReadOnlyModeError):
        deeplake_ds.embedding.delete_vdb_index("hnsw_1")


@requires_libdeeplake
def test_query_recall(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(2000):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})

        deeplake_ds.embedding.create_vdb_index("hnsw_1")
    correct = 0
    for i in range(len(deeplake_ds.embedding)):
        v = deeplake_ds.embedding[i].numpy()
        s = ",".join(str(c) for c in v)
        view = deeplake_ds.query(
            f"select * order by l2_norm(embedding - array[{s}]) limit 1"
        )
        if view.index.values[0].value[0] == i:
            correct += 1

    recall = float(correct) / len(deeplake_ds)
    if recall < 0.7:
        warnings.warn(
            f"Recall is too low - {recall}. Make sure that indexing works properly"
        )
    elif recall >= 1:
        warnings.warn(
            f"Recall is too high - {recall}. Make sure that the query uses indexing instead of bruteforcing."
        )
    else:
        print(f"Recall is in the expected range - {recall}")


@requires_libdeeplake
def test_query_recall_cosine_similarity(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(2000):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})

        deeplake_ds.embedding.create_vdb_index(
            "hnsw_1", distance=DistanceType.COSINE_SIMILARITY
        )
    # Check if the index is recreated properly.
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "cosine_similarity"
    assert es[0]["type"] == "hnsw"

    correct = 0
    for i in range(len(deeplake_ds.embedding)):
        v = deeplake_ds.embedding[i].numpy()
        s = ",".join(str(c) for c in v)
        view = deeplake_ds.query(
            f"select *  order by cosine_similarity(embedding ,array[{s}]) DESC limit 1"
        )
        if view.index.values[0].value[0] == i:
            correct += 1

    recall = float(correct) / len(deeplake_ds)
    if recall < 0.7:
        warnings.warn(
            f"Recall is too low - {recall}. Make sure that indexing works properly"
        )
    elif recall >= 1:
        warnings.warn(
            f"Recall is too high - {recall}. Make sure that the query uses indexing instead of bruteforcing."
        )
    else:
        print(f"Recall is in the expected range - {recall}")


@requires_libdeeplake
def test_index_maintenance_append(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index("hnsw_1", distance="cosine_similarity")
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(5000, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds) - 5000)
        assert recall2 / recall > 0.98
        ds.embeddings.unload_vdb_index_cache()


@requires_libdeeplake
def test_load_nonexistent_vdb_index(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        with pytest.raises(ValueError):
            ds.embeddings.load_vdb_index("nonexistent_index")


@requires_libdeeplake
def test_load_vdb_index_on_empty_dataset(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        with pytest.raises(ValueError):
            ds.embeddings.load_vdb_index("hnsw_3")


@requires_libdeeplake
def test_load_vdb_index_rejects_path_parameter(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        with pytest.raises(TypeError):
            ds.embeddings.load_vdb_index("hnsw_test", path="/invalid/path")

        try:
            ds.embeddings.load_vdb_index("hnsw_test", path="/invalid/path")
            assert (
                False
            ), "TypeError not raised when 'path' parameter was incorrectly provided."
        except TypeError as e:
            assert "unexpected keyword argument 'path'" in str(
                e
            ), "Error raised, but not for the expected 'path' parameter issue."


@requires_libdeeplake
def test_index_maintenance_update(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index("hnsw_1", distance="cosine_similarity")
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[2000] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 2000
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_index_maintenance_delete(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index("hnsw_1", distance="cosine_similarity")
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = ds.embeddings[4999].numpy()
        ds.pop(4999)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] != 4999
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings.append(sample)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 4999
        with pytest.raises(EmbeddingTensorPopError):
            ds.embeddings.pop(2000)
        with pytest.raises(EmbeddingTensorPopError):
            ds.pop(2000)
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_partitioned_index(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        additional_params = {
            "efConstruction": 200,
            "M": 16,
            "partitions": 2,
        }
        ds.embeddings.create_vdb_index(
            "hnsw_1", distance="cosine_similarity", additional_params=additional_params
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 2
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = ds.embeddings[4999].numpy()
        ds.pop(4999)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] != 4999
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings.append(sample)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 4999
        with pytest.raises(EmbeddingTensorPopError):
            ds.embeddings.pop(2000)
        with pytest.raises(EmbeddingTensorPopError):
            ds.pop(2000)
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_partitioned_index_add(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)

        additional_params = {
            "efConstruction": 200,
            "M": 16,
            "partitions": 2,
        }

        ds.embeddings.create_vdb_index(
            "hnsw_1", distance="cosine_similarity", additional_params=additional_params
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 2
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 4
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_partitioned_index_uneven_partitions(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)

        additional_params = {
            "efConstruction": 200,
            "M": 16,
            "partitions": 2,
        }

        ds.embeddings.create_vdb_index(
            "hnsw_1", distance="cosine_similarity", additional_params=additional_params
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 2
        arr = np.random.uniform(-1, 1, (100, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 3

        arr = np.random.uniform(-1, 1, (700, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 6
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_partitioned_index_delete(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)

        additional_params = {
            "efConstruction": 200,
            "M": 16,
            "partitions": 2,
        }

        ds.embeddings.create_vdb_index(
            "hnsw_1", distance="cosine_similarity", additional_params=additional_params
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 2
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 4
        ds.pop(9999)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 4
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_partitioned_index_update(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)

        additional_params = {
            "efConstruction": 200,
            "M": 16,
            "partitions": 2,
        }
        ds.embeddings.create_vdb_index(
            "hnsw_1", distance="cosine_similarity", additional_params=additional_params
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 2
        arr = np.random.uniform(-1, 1, (5000, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 4
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[2000] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        ret = index.search_knn(ds.embeddings[2000].numpy(), 1)
        assert ret.indices[0] == 2000
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_index_partitioned_maintenance_delete(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (50, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 5},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        assert recall == 1.0
        sample = ds.embeddings[49].numpy()
        ds.pop(49)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] != 49
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings.append(sample)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 49
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_index_partitioned_maintenance_update(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 5},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[200] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.90
        ret = index.search_knn(sample, 1)
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_partitioned_add_multiple_new_part(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 10},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        arr = np.random.uniform(-1, 1, (100, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(500, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds) - 500)
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 12
        assert recall2 / recall > 0.98
        arr = np.random.uniform(-1, 1, (700, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        assert recall2 / recall > 0.98
        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_partitioned_add_multiple_old_part(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 10},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        # Add 10 rows to make new partition.
        arr = np.random.uniform(-1, 1, (10, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 11
        assert recall2 / recall > 0.98

        # Add 10 rows to the old partition.
        arr = np.random.uniform(-1, 1, (10, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 11
        assert recall2 / recall > 0.98

        # Add 10 rows to the old partition.
        arr = np.random.uniform(-1, 1, (10, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 11
        assert recall2 / recall > 0.98

        # Add 100 rows to the old partition and new partition.
        arr = np.random.uniform(-1, 1, (70, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 12
        assert recall2 / recall > 0.98

        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_index_maintenance_update_add(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (500, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 10},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[200] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 200
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 10

        # Add 100 rows to the old partition and new partition.
        arr = np.random.uniform(-1, 1, (10, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 11
        assert recall2 / recall > 0.98

        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[505] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / len(ds)
        assert recall2 / recall > 0.98

        # Add 100 rows to the old partition and new partition.
        arr = np.random.uniform(-1, 1, (100, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds))
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 13
        assert recall2 / recall > 0.98

        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[595] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(500, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall2 = count / (len(ds) - 500)
        assert recall2 == 1.0
        assert recall2 / recall > 0.98

        ds.embeddings.unload_vdb_index_cache()


@pytest.mark.slow
@requires_libdeeplake
def test_incremental_index_partitioned_maintenance_delete_multipart(
    local_auth_ds_generator,
):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype=np.float32,
            htype="embedding",
            sample_compression=None,
        )
        ds.embeddings.unload_vdb_index_cache()
        arr = np.random.uniform(-1, 1, (50, 48)).astype("float32")
        ds.embeddings.extend(arr)
        ds.embeddings.create_vdb_index(
            "hnsw_1",
            distance="cosine_similarity",
            additional_params={"M": 16, "efConstruction": 500, "partitions": 5},
        )
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall = count / len(ds)
        assert recall == 1.0

        # Pop Last Row 49.
        sample = ds.embeddings[49].numpy()
        ds.pop(49)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall21 = count / len(ds)
        assert recall21 == 1.0
        assert len(ds) == 49
        assert recall21 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] != 49

        # Add 1 row to the old partition.
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings.append(sample)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall22 = count / len(ds)
        assert recall22 == 1.0
        assert len(ds) == 50
        assert recall22 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] == 49

        # Add 10 rows to the old partition and new partition.
        arr = np.random.uniform(-1, 1, (10, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall23 = count / (len(ds))
        assert recall23 == 1.0
        assert len(ds) == 60
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 6
        assert recall23 / recall > 0.98

        # Update 1 row to the old partition.
        sample = np.random.uniform(-1, 1, (48)).astype("float32")
        ds.embeddings[55] = sample
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(50, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall24 = count / (len(ds) - 50)
        assert len(ds) == 60
        assert recall24 == 1.0
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 6
        assert recall24 / recall > 0.98

        # Add 2 rows to the new partition and new partition.
        arr = np.random.uniform(-1, 1, (2, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(0, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall25 = count / (len(ds))
        assert recall25 == 1.0
        assert len(ds) == 62
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 7
        assert recall25 / recall > 0.98

        # Pop Last Row 61.
        sample = ds.embeddings[61].numpy()
        ds.pop(61)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            if i == ret.indices[0]:
                count += 1
        recall26 = count / len(ds)
        assert recall26 == 1.0
        assert len(ds) == 61
        assert recall26 / recall > 0.98
        ret = index.search_knn(sample, 1)
        assert ret.indices[0] != 61

        # Add 20 rows to the new partition and new partition.
        arr = np.random.uniform(-1, 1, (20, 48)).astype("float32")
        ds.embeddings.extend(arr)
        index = ds.embeddings.load_vdb_index("hnsw_1")
        count = 0
        for i in range(61, len(ds)):
            ret = index.search_knn(ds.embeddings[i].numpy(fetch_chunks=True), 1)
            print(f"i: {i}, ret: {ret.indices[0]}")
            if i == ret.indices[0]:
                count += 1
        recall27 = count / (len(ds) - 61)
        assert len(ds) == 81
        assert recall27 == 1.0
        additional_params = ds.embeddings.get_vdb_indexes()[0]["additional_params"]
        assert additional_params["partitions"] == 9
        assert recall27 / recall > 0.98

        ds.embeddings.unload_vdb_index_cache()
