import deeplake
import numpy as np
from deeplake.tests.common import requires_libdeeplake
from deeplake.util.exceptions import ReadOnlyModeError
import pytest
import warnings


@requires_libdeeplake
def test_index_management(local_ds_generator):
    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(200):
            deeplake_ds.embedding.append(
                np.random.random_sample(
                    384
                ).astype(np.float32)
            )
    deeplake_ds.embedding.create_vdb_index("hnsw_1")
    es = deeplake_ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]['id'] == 'hnsw_1'
    assert es[0]['distance'] == 'l2_norm'
    assert es[0]['type'] == 'hnsw'
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
def test_query_recall(local_ds_generator):
    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for _ in range(2000):
            deeplake_ds.embedding.append(
                np.random.random_sample(
                    384
                ).astype(np.float32)
            )
        deeplake_ds.embedding.create_vdb_index("hnsw_1")
    correct = 0
    for i in range(len(deeplake_ds.embedding)):
        v = deeplake_ds.embedding[i].numpy()
        s = ','.join(str(c) for c in v)
        view = deeplake_ds.query(f"select * order by l2_norm(embedding - array[{s}]) limit 1")
        if view.index.values[0].value[0] == i:
            correct += 1

    recall = float(correct) / len(deeplake_ds)
    if recall < 0.7:
        warnings.warn(f"Recall is too low - {recall}. Make sure that indexing works properly")
    elif recall >= 1:
        warnings.warn(f"Recall is too high - {recall}. Make sure that the query uses indexing instead of bruteforcing.")
    else:
        print(f"Recall is in the expected range - {recall}")
