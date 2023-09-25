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


@pytest.mark.slow
def test_deepmemory_train(corpus_query_pair_path, hub_cloud_dev_token):
    pass


@pytest.mark.slow
def test_deepmemory_evaluate(
    corpus_query_pair_path,
    questions_embeddings,
    question_relvences,
    hub_cloud_dev_token,
):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    recall = db.deep_memory.evaluate(
        embedding=questions_embeddings,
        relevances=question_relvences,
    )

    assert recall["with model"] = 