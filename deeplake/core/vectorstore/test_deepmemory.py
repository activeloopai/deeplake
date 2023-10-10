import logging
import numpy as np
import pytest
import sys
from time import sleep

import deeplake
from deeplake import VectorStore
from deeplake.tests.common import requires_libdeeplake


class DummyEmbedder:
    @staticmethod
    def embed_documents(texts):
        return [
            np.random.uniform(low=-10, high=10, size=(1536)).astype(np.float32)
            for _ in range(len(texts))
        ]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_init(hub_cloud_path, hub_cloud_dev_token):
    db = VectorStore(
        hub_cloud_path,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    assert db.deep_memory is not None


def embedding_fn(texts):
    return [
        np.random.uniform(low=-10, high=10, size=(1536)).astype(np.float32)
        for _ in range(len(texts))
    ]


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_train_and_cancel(
    capsys,
    corpus_query_relevances_copy,
    hub_cloud_dev_token,
):
    corpus, queries, relevances, _ = corpus_query_relevances_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    with pytest.raises(ValueError):
        # When embedding_function is provided neither in the constructor nor in the train method
        job_id = db.deep_memory.train(
            queries=queries,
            relevance=relevances,
        )

    job_id = db.deep_memory.train(
        queries=queries,
        relevance=relevances,
        embedding_function=embedding_fn,
    )

    # cancelling right after starting the job
    cancelled = db.deep_memory.cancel(job_id)
    assert cancelled == True

    # deleting the job
    deleted = db.deep_memory.delete(job_id)
    assert deleted == True

    # when embedding function is provided in the constructor
    deeplake.deepcopy(
        corpus,
        corpus + "_copy",
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    db = VectorStore(
        path=corpus + "_copy",
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=DummyEmbedder,
    )

    job_id = db.deep_memory.train(
        queries=queries,
        relevance=relevances,
        embedding_function=embedding_fn,
    )

    # TODO: Investigate why it is flaky
    # # cancelling right after starting the job
    # cancelled = db.deep_memory.cancel(job_id)
    # assert cancelled == True

    # # deleting the job
    # deleted = db.deep_memory.delete(job_id)
    # assert deleted == True

    # cancelled = db.deep_memory.cancel("non-existent-job-id")
    # out_str = capsys.readouterr()
    # error_str = (
    #     "Job with job_id='non-existent-job-id' was not cancelled!\n "
    #     "Error: Entity non-existent-job-id does not exist.\n"
    # )
    # assert cancelled == False
    # assert out_str.out == error_str

    deeplake.delete(
        corpus + "_copy", force=True, large_ok=True, token=hub_cloud_dev_token
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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

    # when qvs_params is wrong:
    with pytest.raises(ValueError):
        db.deep_memory.evaluate(
            queries=queries,
            embedding=questions_embeddings,
            relevance=question_relevances,
            qvs_params={
                "log_queries": True,
                "branch_name": "wrong_branch",
            },
        )

    # embedding_function is not provided in the constructor or in the eval method
    with pytest.raises(ValueError):
        db.deep_memory.evaluate(
            queries=queries,
            relevance=question_relevances,
            qvs_params={
                "log_queries": True,
                "branch_name": "wrong_branch",
            },
        )

    recall = db.deep_memory.evaluate(
        queries=queries,
        embedding=questions_embeddings,
        relevance=question_relevances,
        qvs_params={
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

    # TODO: add this back when the issue with the backend is resolved
    # assert recall["with model"] == {
    #     "recall@1": 0.8,
    #     "recall@3": 0.8,
    #     "recall@5": 0.9,
    #     "recall@10": 0.9,
    #     "recall@50": 0.9,
    #     "recall@100": 0.9,
    # }
    sleep(15)
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
        branch="queries",
    )
    assert len(queries_dataset) == len(question_relevances)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_log_queries(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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

    sleep(15)
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
        branch="queries",
    )
    queries_dataset.checkout("queries")
    assert len(queries_dataset) == len(question_relevances)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_without_branch_name_with_logging(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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
        },
    )

    sleep(15)
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == len(question_relevances)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_without_logging(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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
            "log_queries": False,
        },
    )
    sleep(15)

    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == 0


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_without_branch_name(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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
        qvs_params={"log_queries": True},
    )

    sleep(15)
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == len(question_relevances)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_without_qvs_params(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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
    # TODO: Improve these tests when lockup issues will be resolved

    recall = db.deep_memory.evaluate(
        queries=queries,
        relevance=question_relevances,
        embedding_function=embedding_fn,
    )

    sleep(15)

    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == 0


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@requires_libdeeplake
def test_deepmemory_evaluate_with_embedding_func_in_init(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=DummyEmbedder,
    )
    recall = db.deep_memory.evaluate(
        queries=queries,
        relevance=question_relevances,
        qvs_params={"log_queries": True},
    )

    sleep(15)
    queries_dataset = VectorStore(
        path=query_path,
        token=hub_cloud_dev_token,
        read_only=True,
    )
    assert len(queries_dataset) == len(question_relevances)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_evaluate_without_embedding_function(
    corpus_query_relevances_copy,
    questions_embeddings_and_relevances,
    hub_cloud_dev_token,
):
    corpus, _, _, query_path = corpus_query_relevances_copy
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

    sleep(15)
    # when embedding_function is not provided in the constructor and not in the eval method
    with pytest.raises(ValueError):
        db.deep_memory.evaluate(
            queries=queries,
            relevance=question_relevances,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_list_jobs(jobs_list, corpus_query_pair_path, hub_cloud_dev_token):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        read_only=True,
    )

    output_str = db.deep_memory.list_jobs(debug=True)
    # TODO: The reason why index is added is because sometimes backends returns request
    # parameters in different order need to address this issue either on a client side
    # or on a backend side
    assert output_str[:375] == jobs_list[:375]


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_status(capsys, job_id, corpus_query_pair_path, hub_cloud_dev_token):
    output_str = (
        "--------------------------------------------------------------\n"
        f"|                  {job_id}                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | completed                     |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 2.5 seconds              |\n"
        "|                            | recall@10: 0.62% (+0.62%)     |\n"
        "--------------------------------------------------------------\n"
        "| results                    | recall@10: 0.62% (+0.62%)     |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    corpus, query_path = corpus_query_pair_path

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    jobs_list = db.deep_memory.status(job_id)
    status = capsys.readouterr()
    # TODO: The reason why index is added is because sometimes backends returns request
    # parameters in different order need to address this issue either on a client side
    # or on a backend side
    assert status.out[511:] == output_str[511:]


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_search(
    corpus_query_pair_path,
    hub_cloud_dev_token,
):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    query_embedding = np.random.uniform(low=-10, high=10, size=(1536)).astype(
        np.float32
    )

    output = db.search(embedding=query_embedding)

    assert db.deep_memory is not None
    assert len(output) == 4

    output = db.search(embedding=query_embedding, exec_option="compute_engine")
    assert len(output) == 4
    # TODO: add some logging checks


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@pytest.mark.slow
@requires_libdeeplake
def test_deepmemory_search_on_local_datasets(
    deep_memory_local_dataset, hub_cloud_dev_token
):
    corpus_path, queries_path = deep_memory_local_dataset

    corpus = VectorStore(path=corpus_path, token=hub_cloud_dev_token)
    queries = VectorStore(path=queries_path, token=hub_cloud_dev_token)

    deep_memory_query_view = queries.search(
        query="SELECT * WHERE deep_memory_recall > vector_search_recall",
        return_view=True,
    )

    query_embedding = deep_memory_query_view[0].embedding.data()["value"]
    correct_id = deep_memory_query_view[0].metadata.data()["value"]["relvence"][0][0]

    output = corpus.search(embedding=query_embedding, deep_memory=True, k=10)

    assert correct_id in output["id"]
    assert "score" in output
