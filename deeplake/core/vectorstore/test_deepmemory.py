import numpy as np
import pytest
from time import sleep

import deeplake
from deeplake import VectorStore


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


def test_deepmemory_train_and_cancel(
    corpus_query_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, queries = corpus_query_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    questions_embeddings, questions_relevances = questions_embeddings_and_relevances

    job = db.deep_memory.train(
        query_embeddings=questions_embeddings,
        relevances=questions_relevances,
    )

    cancelled = db.deep_memory.cancel(job["job_id"])
    assert cancelled == True

    job = db.deep_memory.train(
        queries=queries,
        relevances=questions_relevances,
        embedding_function=embedding_fn,
    )

    cancelled = db.deep_memory.cancel(job["job_id"])
    assert cancelled == True


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_deepmemory_evaluate(
    corpus_query_pair_path,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _ = corpus_query_pair_path
    questions_embeddings, question_relevances = questions_embeddings_and_relevances

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    recall = db.deep_memory.evaluate(
        embedding=questions_embeddings,
        relevances=question_relevances,
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
