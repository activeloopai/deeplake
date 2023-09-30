import numpy as np
import pytest
from time import sleep

import deeplake
from deeplake import VectorStore


@pytest.mark.slow
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


@pytest.mark.slow
def test_deepmemory_train_and_cancel(
    corpus_query_relevances_copy,
    hub_cloud_dev_token,
):
    corpus, queries, relevances = corpus_query_relevances_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    job_id = db.deep_memory.train(
        queries=queries,
        relevance=relevances,
        embedding_function=embedding_fn,
    )

    cancelled = db.deep_memory.cancel(job_id)
    assert cancelled == True

    deleted = db.deep_memory.delete(job_id)
    assert deleted == True


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_deepmemory_evaluate(
    corpus_query_pair_path,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, query_path = corpus_query_pair_path
    (
        questions_embeddings,
        question_relevances,
        queries,
    ) = questions_embeddings_and_relevances

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    recall = db.deep_memory.evaluate(
        queries=queries,
        embedding=questions_embeddings,
        relevance=question_relevances,
        qvs_params={
            "log_queries": True,
            "branch": "queries",
        },
    )

    assert recall["without model"] == {
        "recall@1": 0.4,
        "recall@3": 0.6,
        "recall@5": 0.6,
        "recall@10": 0.6,
        "recall@50": 0.7,
        "recall@100": 0.9,
    }

    assert recall["with model"] == {
        "recall@1": 0.9,
        "recall@3": 0.9,
        "recall@5": 0.9,
        "recall@10": 0.9,
        "recall@50": 0.9,
        "recall@100": 0.9,
    }

    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
        branch="queries",
    )
    assert len(queries_dataset) == len(question_relevances)

    recall = db.deep_memory.evaluate(
        queries=queries,
        embedding=questions_embeddings,
        relevance=question_relevances,
        qvs_params={
            "log_queries": True,
            "branch": "queries",
        },
    )
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
        branch="queires",
    )
    queries_dataset.checkout("queires")
    assert len(queries_dataset) == 2 * len(question_relevances)

    recall = db.deep_memory.evaluate(
        queries=queries,
        embedding=questions_embeddings,
        relevance=question_relevances,
        qvs_params={
            "log_queries": True,
        },
    )
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == len(question_relevances)
