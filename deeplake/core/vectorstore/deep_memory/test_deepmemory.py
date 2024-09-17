import logging
import numpy as np
import pytest
import sys
from time import sleep
from unittest.mock import patch

import deeplake
from deeplake import VectorStore
from deeplake.core.vectorstore.deep_memory.deep_memory import (
    DeepMemory,
    _get_best_model,
)
from deeplake.tests.common import requires_libdeeplake
from deeplake.util.exceptions import (
    DeepMemoryAccessError,
    IncorrectQueriesTypeError,
    IncorrectRelevanceTypeError,
)
from deeplake.util.exceptions import DeepMemoryAccessError


logger = logging.getLogger(__name__)


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


def embedding_fn(texts, embedding_dim=1536):
    return [
        np.random.uniform(low=-10, high=10, size=(embedding_dim)).astype(np.float32)
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

    assert recall["with model"] == {
        "recall@1": 0.9,
        "recall@3": 0.9,
        "recall@5": 0.9,
        "recall@10": 0.9,
        "recall@50": 0.9,
        "recall@100": 0.9,
    }

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
        embedding_function=embedding_fn,
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
    corpus, _, _, _ = corpus_query_relevances_copy
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
    corpus, _ = corpus_query_pair_path
    dataset_id = corpus.split("//")[1].strip("/")

    class _response_mock:
        @property
        def status_code(self):
            return 200

        def json(self):
            return {
                "id": job_id,
                "status": "completed",
                "dataset_id": dataset_id,
                "progress": {
                    "eta": "2.5 seconds",
                    "best_recall@10": "50.00% (+25.00%)",
                },
            }

    output_str = (
        "--------------------------------------------------------------\n"
        f"|                  {job_id}                  |\n"
        "--------------------------------------------------------------\n"
        "| status                     | completed                     |\n"
        "--------------------------------------------------------------\n"
        "| progress                   | eta: 2.5 seconds              |\n"
        "|                            | recall@10: 50.00% (+25.00%)   |\n"
        "--------------------------------------------------------------\n"
        "| results                    | recall@10: 50.00% (+25.00%)   |\n"
        "--------------------------------------------------------------\n\n\n"
    )

    db = VectorStore(
        corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    with patch("deeplake.client.client.DeepMemoryBackendClient.request") as api_mock:
        api_mock.return_value = _response_mock()
        db.deep_memory.status(job_id)
        status = capsys.readouterr()
        assert status.out[511:] == output_str[511:]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_search(
    corpus_query_relevances_copy,
    testing_relevance_query_deepmemory,
    hub_cloud_dev_token,
):
    corpus, _, _, _ = corpus_query_relevances_copy
    relevance, query_embedding = testing_relevance_query_deepmemory

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    output = db.search(
        embedding=query_embedding, deep_memory=True, return_tensors=["id"]
    )

    assert len(output["id"]) == 4
    assert relevance in output["id"]

    output = db.search(embedding=query_embedding)
    assert len(output["id"]) == 4
    assert relevance not in output["id"]
    # TODO: add some logging checks


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@pytest.mark.slow
@requires_libdeeplake
def test_deepmemory_search_on_local_datasets(
    deep_memory_local_dataset,
    testing_relevance_query_deepmemory,
    hub_cloud_dev_token,
):
    corpus_path = deep_memory_local_dataset
    relevance, query_embedding = testing_relevance_query_deepmemory

    corpus = VectorStore(path=corpus_path, token=hub_cloud_dev_token)

    output = corpus.search(embedding=query_embedding, deep_memory=True, k=10)
    assert relevance in output["id"]
    assert "score" in output

    output = corpus.search(embedding=query_embedding, deep_memory=False, k=10)
    assert relevance not in output["id"]
    assert "score" in output


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@pytest.mark.slow
@requires_libdeeplake
def test_unsupported_deepmemory_users(local_ds):
    dm = DeepMemory(
        path=local_ds,
        dataset=None,
        logger=logger,
        embedding_function=DummyEmbedder,
    )
    with pytest.raises(DeepMemoryAccessError):
        dm.train(
            queries=[],
            relevance=[],
        )

    with pytest.raises(DeepMemoryAccessError):
        dm.status(job_id="123")

    with pytest.raises(DeepMemoryAccessError):
        dm.list_jobs()

    with pytest.raises(DeepMemoryAccessError):
        dm.evaluate(
            queries=[],
            relevance=[],
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@pytest.mark.slow
@requires_libdeeplake
def test_deepmemory_list_jobs_with_no_jobs(
    corpus_query_relevances_copy, hub_cloud_dev_token
):
    corpus, _, _, _ = corpus_query_relevances_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    output_str = db.deep_memory.list_jobs(debug=True)
    assert output_str == "No Deep Memory training jobs were found for this dataset"


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
@pytest.mark.slow
@requires_libdeeplake
def test_not_supported_training_args(corpus_query_relevances_copy, hub_cloud_dev_token):
    corpus, queries, relevances, _ = corpus_query_relevances_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    with pytest.raises(IncorrectQueriesTypeError):
        db.deep_memory.train(
            queries="queries",
            relevance=relevances,
            embedding_function=embedding_fn,
        )

    with pytest.raises(IncorrectRelevanceTypeError):
        db.deep_memory.train(
            queries=queries,
            relevance="relevances",
            embedding_function=embedding_fn,
        )

    with pytest.raises(IncorrectQueriesTypeError):
        db.deep_memory.evaluate(
            queries="queries",
            relevance=relevances,
        )

    with pytest.raises(IncorrectRelevanceTypeError):
        db.deep_memory.evaluate(
            queries=queries,
            relevance="relevances",
        )


@pytest.mark.slow
def test_deepmemory_v2_set_model_should_set_model_for_all_subsequent_loads(
    local_dmv2_dataset,
    hub_cloud_dev_token,
):
    # Setiing model should set model for all subsequent loads
    db = VectorStore(path=local_dmv2_dataset, token=hub_cloud_dev_token)
    assert db.deep_memory.get_model() == "655f86e8ab93e7fc5067a3ac_2"

    # ensure after setting model, get model returns specified model
    db.deep_memory.set_model("655f86e8ab93e7fc5067a3ac_1")

    assert (
        db.dataset.embedding.info["deepmemory"]["model.npy"]["job_id"]
        == "655f86e8ab93e7fc5067a3ac_1"
    )
    assert db.deep_memory.get_model() == "655f86e8ab93e7fc5067a3ac_1"

    # ensure after setting model, reloading the dataset returns the same model
    db = VectorStore(path=local_dmv2_dataset, token=hub_cloud_dev_token)
    assert db.deep_memory.get_model() == "655f86e8ab93e7fc5067a3ac_1"


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_search_should_contain_correct_answer(
    corpus_query_relevances_copy,
    testing_relevance_query_deepmemory,
    hub_cloud_dev_token,
):
    corpus, _, _, _ = corpus_query_relevances_copy
    relevance, query_embedding = testing_relevance_query_deepmemory

    db = VectorStore(
        path=corpus,
        token=hub_cloud_dev_token,
    )

    output = db.search(
        embedding=query_embedding, deep_memory=True, return_tensors=["id"]
    )
    assert len(output["id"]) == 4
    assert relevance in output["id"]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deeplake_search_should_not_contain_correct_answer(
    corpus_query_relevances_copy,
    testing_relevance_query_deepmemory,
    hub_cloud_dev_token,
):
    corpus, _, _, _ = corpus_query_relevances_copy
    relevance, query_embedding = testing_relevance_query_deepmemory

    db = VectorStore(
        path=corpus,
        token=hub_cloud_dev_token,
    )
    output = db.search(embedding=query_embedding)
    assert len(output["id"]) == 4
    assert relevance not in output["id"]


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_train_with_embedding_function_specified_in_constructor_should_not_throw_any_exception(
    deepmemory_small_dataset_copy,
    hub_cloud_dev_token,
):
    corpus, queries, relevances, _ = deepmemory_small_dataset_copy

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=embedding_fn,
    )

    _ = db.deep_memory.train(
        queries=queries,
        relevance=relevances,
    )


@pytest.mark.slow
@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_deepmemory_evaluate_with_embedding_function_specified_in_constructor_should_not_throw_any_exception(
    corpus_query_pair_path,
    hub_cloud_dev_token,
):
    corpus, queries = corpus_query_pair_path

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=embedding_fn,
    )

    queries_vs = VectorStore(
        path=queries,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=embedding_fn,
        read_only=True,
    )

    queries = queries_vs.dataset[:10].text.data()["value"]
    relevance = queries_vs.dataset[:10].metadata.data()["value"]
    relevance = [rel["relevance"] for rel in relevance]

    _ = db.deep_memory.evaluate(
        queries=queries,
        relevance=relevance,
    )


def test_db_deepmemory_status_should_show_best_model_with_deepmemory_v2_metadata_logic(
    capsys,
    corpus_query_pair_path,
    hub_cloud_dev_token,
):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=embedding_fn,
    )
    db.dataset.embedding.info = {
        "deepmemory": {
            "6581e3056a1162b64061a9a4_0.npy": {
                "base_recall@10": 0.25,
                "deep_memory_version": "0.2",
                "delta": 0.25,
                "job_id": "6581e3056a1162b64061a9a4_0",
                "model_type": "npy",
                "recall@10": 0.5,
            },
            "model.npy": {
                "base_recall@10": 0.25,
                "deep_memory_version": "0.2",
                "delta": 0.25,
                "job_id": "6581e3056a1162b64061a9a4_0",
                "model_type": "npy",
                "recall@10": 0.5,
            },
        }
    }

    recall, delta = _get_best_model(
        db.dataset.embedding,
        "6581e3056a1162b64061a9a4",
        latest_job=True,
    )
    assert recall == 0.5
    assert delta == 0.25


def test_db_deepmemory_status_should_show_best_model_with_deepmemory_v1_metadata_logic(
    capsys,
    corpus_query_pair_path,
    hub_cloud_dev_token,
):
    corpus, _ = corpus_query_pair_path

    db = VectorStore(
        path=corpus,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
        embedding_function=embedding_fn,
    )
    db.dataset.embedding.info = {
        "deepmemory": {
            "6581e3056a1162b64061a9a4_0.npy": {
                "base_recall@10": 0.25,
                "deep_memory_version": "0.2",
                "delta": 0.25,
                "job_id": "6581e3056a1162b64061a9a4_0",
                "model_type": "npy",
                "recall@10": 0.5,
            },
        },
        "deepmemory/model.npy": {
            "base_recall@10": 0.25,
            "deep_memory_version": "0.2",
            "delta": 0.25,
            "job_id": "6581e3056a1162b64061a9a4_0",
            "model_type": "npy",
            "recall@10": 0.5,
        },
    }

    recall, delta = _get_best_model(
        db.dataset.embedding,
        "6581e3056a1162b64061a9a4",
        latest_job=True,
    )
    assert recall == 0.5
    assert delta == 0.25
