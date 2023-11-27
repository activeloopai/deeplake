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
    assert es[0]["distance"] == "l2_norm"
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
