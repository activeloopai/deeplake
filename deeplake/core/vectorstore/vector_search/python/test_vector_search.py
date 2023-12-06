import pytest

import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search.python import vector_search
from deeplake.core.dataset import Dataset as DeepLakeDataset


def test_vector_search():
    ds = deeplake.empty("mem://test_vector_search")
    ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
    ds.embedding.extend(np.zeros((10, 10), dtype=np.float32))

    query_embedding = np.zeros((10), dtype=np.float32)

    data = vector_search.vector_search(
        query=None,
        query_emb=query_embedding,
        exec_option="python",
        dataset=ds,
        logger=None,
        filter=None,
        embedding_tensor="embedding",
        distance_metric="l2",
        k=10,
        return_tensors=[],
        return_view=False,
        token=None,
        org_id=None,
        return_tql=False,
    )

    assert len(data["score"]) == 10

    with pytest.raises(ValueError):
        data = vector_search.vector_search(
            query=None,
            query_emb=query_embedding,
            exec_option="python",
            dataset=ds[0:0],
            logger=None,
            filter=None,
            embedding_tensor="embedding",
            distance_metric="l2",
            k=10,
            return_tensors=[],
            return_view=False,
            token=None,
            org_id=None,
            return_tql=False,
        )

    data = vector_search.vector_search(
        query=None,
        query_emb=query_embedding,
        exec_option="python",
        dataset=ds,
        logger=None,
        filter=None,
        embedding_tensor="embedding",
        distance_metric="l2",
        k=10,
        return_tensors=[],
        return_view=True,
        token=None,
        org_id=None,
        return_tql=False,
    )

    assert len(data) == 10
    assert isinstance(data, DeepLakeDataset)

    with pytest.raises(NotImplementedError):
        data = vector_search.vector_search(
            query="tql query",
            query_emb=query_embedding,
            exec_option="python",
            dataset=ds,
            logger=None,
            filter=None,
            embedding_tensor="embedding",
            distance_metric="l2",
            k=10,
            return_tensors=[],
            return_view=True,
            token=None,
            org_id=None,
            return_tql=False,
        )
