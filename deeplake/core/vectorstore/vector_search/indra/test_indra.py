import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.indra import query
from deeplake.core.vectorstore.indra.tql_distance_metrics import get_tql_distance_metric

array = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
METRIC_FUNC_TO_QUERY_STRING = {
    "l1": "L1_NORM(embedding-ARRAY[{array}])",
    "l2": "L2_NORM(embedding-ARRAY[{array}])",
    "cos": "COSINE_SIMILARITY(embedding, ARRAY[{array}])",
    "max": "LINF_NORM(embedding-ARRAY[{array}])",
}


@pytest.mark.parametrize(
    "metric",
    [
        "l1",
        "l2",
        "cos",
        "max",
    ],
)
def test_tql_distance_metric(metric):
    query_embedding = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    parsed_query = get_tql_distance_metric(metric, "embedding", query_embedding)
    assert parsed_query == METRIC_FUNC_TO_QUERY_STRING[metric]


@pytest.mark.parametrize(
    "metric",
    [
        "l1",
        "l2",
        "cos",
        "max",
    ],
)
def test_tql_distance_metric(metric, limit=10):
    query_embedding = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    embedding_tensor = "embedding"

    parsed_query = query.parse_query(metric, limit, query_embedding, embedding_tensor)
    assert parsed_query == METRIC_FUNC_TO_QUERY_STRING[metric]
