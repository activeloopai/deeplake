import deeplake
from vectorstore import VectorStore


def test_deepmemory_init(hub_cloud_path, hub_cloud_dev_token):
    db = VectorStore(
        hub_cloud_path,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    assert db.deep_memory is not None


def test_deepmemory_search():
    pass


def test_deepmemory_evaluate():
    pass


def test_deepmemory_train(corpus_query_pair_path, hub_cloud_dev_token):
    corpus, _ = corpus_query_pair_path
    queries = ["query1", "query2", "query3"]
    relevances = [["1"], ["2"], ["3"]]

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    job = db.deep_memory.train(
        queries=queries,
        relevances=relevances,
    )
    db.deep_memory.cancel(job["job_id"])

    assert deeplake.exists(corpus + "_queries")
    qds = deeplake.load(corpus + "_queries")
    for i in range(len(qds.dataset.metata.data()["value"])):
        assert db.deep_memory[i].data()["value"][i] == {"relevance": [f"{i}", 1]}
