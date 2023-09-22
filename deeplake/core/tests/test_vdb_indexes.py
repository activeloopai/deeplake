import deeplake
import numpy as np

from deeplake.core.distance_type import DistanceType
from deeplake.tests.common import requires_libdeeplake
from deeplake.tests.dataset_fixtures import local_auth_ds_generator
from deeplake.util.exceptions import ReadOnlyModeError
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
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(200):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})

    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    # Append rows and check query recall.
    random_embedding = np.random.random_sample(384).astype(np.float32)
    deeplake_ds.append({"embedding": random_embedding})

    # Check if the index is recreated properly.
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # check the query recall.
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
def test_index_maintenance_update(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        random_embedding = np.random.random_sample(384).astype(np.float32)
        deeplake_ds.append({"embedding": random_embedding})

    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    # Append rows and check query recall.
    deeplake_ds[0].update({"embedding": [2] * 384})

    # Check if the index is recreated properly.
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # check the query recall.
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
def test_index_maintenance_pop(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(200):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})

    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    # Pop few rows and check query recall.
    deeplake_ds.pop()

    # Check if the index is recreated properly.
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # check the query recall.
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
def test_index_maintenance_delete(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(200):
            random_embedding = np.random.random_sample(384).astype(np.float32)
            deeplake_ds.append({"embedding": random_embedding})

    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    # delete dataset and check if the index exists.
    deeplake_ds.delete()

    with pytest.raises(KeyError):
        deeplake_ds.embedding.delete_vdb_index("hnsw_1")
