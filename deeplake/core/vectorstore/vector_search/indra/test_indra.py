import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search.indra.tql_distance_metrics import (
    get_tql_distance_metric,
)

array = "ARRAY[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"
METRIC_FUNC_TO_METRIC_STRING = {
    "l1": f"L1_NORM(embedding-{array})",
    "l2": f"L2_NORM(embedding-{array})",
    "cos": f"COSINE_SIMILARITY(embedding, {array})",
    "max": f"LINF_NORM(embedding-{array})",
}


def create_tql_string(metric_function, order="ASC"):
    return f"select * from (select *, {METRIC_FUNC_TO_METRIC_STRING[metric_function]} as score) order by score {order} limit 10"


METRIC_FUNC_TO_QUERY_STRING = {
    "l1": create_tql_string("l1", order="ASC"),
    "l2": create_tql_string("l2", order="ASC"),
    "cos": create_tql_string("cos", order="DESC"),
    "max": create_tql_string("max", order="ASC"),
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
def test_metric_to_tql_metric(metric):
    query_embedding = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    query_str = query.convert_tensor_to_str(query_embedding)
    parsed_query = get_tql_distance_metric(metric, "embedding", query_str)
    assert parsed_query == METRIC_FUNC_TO_METRIC_STRING[metric]


@pytest.mark.parametrize(
    "metric",
    [
        "l1",
        "l2",
        "cos",
        "max",
    ],
)
def test_tql_metric_to_tql_str(metric, limit=10):
    query_embedding = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    embedding_tensor = "embedding"

    parsed_query = query.parse_query(
        metric, 10, query_embedding, embedding_tensor, "", ["*"]
    )
    assert parsed_query == METRIC_FUNC_TO_QUERY_STRING[metric]
