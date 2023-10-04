import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search.indra.tql_distance_metrics import (
    get_tql_distance_metric,
)
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.tests.common import requires_libdeeplake

array = "ARRAY[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]"
METRIC_FUNC_TO_METRIC_STRING = {
    "l1": f"L1_NORM(embedding-{array})",
    "l2": f"L2_NORM(embedding-{array})",
    "cos": f"COSINE_SIMILARITY(embedding, {array})",
    "max": f"LINF_NORM(embedding-{array})",
}


def create_tql_string(metric_function, order="ASC"):
    # TODO: BRING THIS BACK AND DELETE IMPLEMENTATION BELOW AFTER TQL IS UPDATED
    # return f"select * from (select *, {METRIC_FUNC_TO_METRIC_STRING[metric_function]} as score) order by score {order} limit 10"

    return f"select *, score from (select *, {METRIC_FUNC_TO_METRIC_STRING[metric_function]} as score order by {METRIC_FUNC_TO_METRIC_STRING[metric_function]} {order} limit 10)"


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


@pytest.mark.slow
@requires_libdeeplake
def test_search_resulting_shapes(hub_cloud_dev_credentials, hub_cloud_dev_token):
    username, _ = hub_cloud_dev_credentials
    # index behavior fails on old datasets
    vector_store = VectorStore(
        f"hub://{username}/paul_graham_essay", read_only=True, token=hub_cloud_dev_token
    )
    search_text = "What I Worked On"

    def filter_fn(x):
        return search_text in x["text"].data()["value"]

    embedding = vector_store.dataset.embedding[0].numpy()
    embedding_str = "ARRAY[{}]".format(", ".join(map(str, embedding)))
    TQL_QUERY = f"select * from (select *, L2_NORM(embedding-{embedding_str}) as score where contains(text, '{search_text}')) order by score ASC limit 4"

    view = vector_store.dataset.filter(filter_fn)
    view_value = view.text.data(aslist=True)["value"]
    view_value_0 = view[0].text.data(aslist=True)["value"]

    view1 = vector_store.dataset.query(
        f"select * where contains(text, '{search_text}')"
    )
    view1_value = view1.text.data(aslist=True)["value"]
    view1_value_0 = view1[0].text.data(aslist=True)["value"]

    view2 = vector_store.dataset.query(TQL_QUERY)
    view2_value = view2.text.data(aslist=True)["value"]
    view2.text.summary()
    assert len(view2.text) == len(view2) == 1
    view2_value_0 = view2[0].text.data(aslist=True)["value"]

    assert view_value == view1_value == view2_value
    assert view_value_0 == view1_value_0 == view2_value_0
