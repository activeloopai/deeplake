from typing import Optional, Tuple, List

import numpy as np

from deeplake.core.vectorstore.vector_search.python import vector_search


def test_vector_search():
    query_embedding = np.zeros((1, 100), dtype=np.float32)
    embedding = np.array([])
    indices, scores = vector_search.vector_search(query_embedding, embedding)
    assert indices == []
    assert scores == []
