import numpy as np
import pytest

import deeplake
from deeplake import VectorStore

deeplake.client.config.USE_STAGING_ENVIRONMENT = True


def test_deepmemory_init(hub_cloud_path, hub_cloud_dev_token):
    db = VectorStore(
        hub_cloud_path,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    assert db.deep_memory is not None


def embedding_fn(texts):
    return [np.random.rand(1536) for _ in range(len(texts))]


def test_deepmemory_search():
    pass


# @requires_libdeeplake
def test_deepmemory_evaluate(corpus_query_pair_path, hub_cloud_dev_token):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    db.deep_memory.evaluate(
        queries=["query1", "query2", "query3"],
        relevances=[["1"], ["2"], ["3"]],
        embedding_function=embedding_fn,
    )


@pytest.mark.slow
def test_deepmemory_train(corpus_query_pair_path, hub_cloud_dev_token):
    pass


hub_cloud_dev_token = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NzQyMDYwNiwiZXhwIjoxNzE5MDQyOTYwfQ.eyJpZCI6InRlc3RpbmdhY2MyIn0.OgeY0HeJQe9KMR3PIK-_2Ey-rAYlYlB80x1VnyqKkw1ufPFbIPLs5BmJT67_G0AKwH2u5nkhhRpA-aYWRebMyA"

# deeplake.deepcopy(
#     "hub://testingacc2/scifact_corpus",
#     "hub://testingacc2/corpus_test",
#     overwrite=True,
#     token=hub_cloud_dev_token,
# )
# # corpus, _ = corpus_query_pair_path
# corpus = "hub://testingacc2/corpus_test"
# queries = ["query1", "query2", "query3"]
# relevances = [["1"], ["2"], ["3"]]

# db = VectorStore(
#     corpus,
#     runtime={"tensor_db": True},
#     token=hub_cloud_dev_token,
# )

# # with pytest.raises(ValueError):
# #     job = db.deep_memory.train(
# #         queries=queries,
# #         relevances=relevances,
# #     )

# job = db.deep_memory.train(
#     queries=queries,
#     relevances=relevances,
#     embedding_function=embedding_fn,
# )
# db.deep_memory.cancel(job["job_id"])

# assert deeplake.exists(corpus + "_queries")
# qds = deeplake.load(corpus + "_queries")
# for i in range(len(qds.dataset.metata.data()["value"])):
#     assert qds[i].data()["value"] == {"relevance": [[f"{i}", 1]]}


corpus = "hub://testingacc2/scifact_corpus"

db = VectorStore(
    corpus,
    runtime={"tensor_db": True},
    token=hub_cloud_dev_token,
)

db.deep_memory.evaluate(
    queries=["query1", "query2", "query3"],
    relevances=[["1"], ["2"], ["3"]],
    embedding_function=embedding_fn,
)
