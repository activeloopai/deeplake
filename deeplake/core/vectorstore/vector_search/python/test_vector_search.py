import deeplake
from typing import Optional, Tuple, List
import numpy as np
from deeplake.core.vectorstore.vector_search.python import vector_search


def test_vector_search():
    ds = deeplake.empty("mem://test_vector_search")
    ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
    ds.embedding.extend(np.zeros((10, 10), dtype=np.float32))

    query_embedding = np.zeros((10), dtype=np.float32)
    embeddings = ds.embedding.numpy()

    data = vector_search.vector_search(ds, query_embedding, embeddings, k=10)

    assert len(data["score"]) == 10

    data = vector_search.vector_search(ds, query_embedding, embeddings[0:0], k=10)

    assert len(data["score"]) == 0
