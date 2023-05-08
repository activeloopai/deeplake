import pytest
import logging

import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils


logger = logging.getLogger(__name__)


def test_create_or_load_dataset():
    # dataset creation
    dataset = dataset_utils.create_or_load_dataset(
        dataset_path="./test-dataset",
        token=None,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="indra",
        overwrite=True,
    )
    assert len(dataset) == 0
    assert set(dataset.tensors.keys()) == {
        "embedding",
        "ids",
        "metadata",
        "text",
    }

    # dataset loading
    dataset = dataset_utils.create_or_load_dataset(
        dataset_path="hub://activeloop/mnist-train",
        token=None,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="python",
    )
    assert len(dataset) == 60000


def test_delete_and_commit():
    dataset = deeplake.empty("./test-dataset", overwrite=True)
    dataset.create_tensor("ids")
    dataset.ids.extend([1, 2, 3, 4, 5, 6, 7, 8, 9])

    dataset_utils.delete_and_commit(dataset, ids=[1, 2, 3])
    len(dataset) == 6


def test_delete_all():
    dataset = deeplake.empty("./test-dataset", overwrite=True)
    dataset.create_tensor("ids")
    dataset.ids.extend([1, 2, 3, 4, 5, 6, 7, 8, 9])

    dataset, deleted = dataset_utils.delete_all_samples_if_specified(
        dataset=dataset, delete_all=True
    )
    assert len(dataset) == 0
    assert deleted == True

    dataset = deeplake.empty("./test-dataset", overwrite=True)
    dataset.create_tensor("ids")
    dataset.ids.extend([1, 2, 3, 4, 5, 6, 7, 8, 9])
    dataset, deleted = dataset_utils.delete_all_samples_if_specified(
        dataset=dataset, delete_all=False
    )
    assert len(dataset) == 9
    assert deleted == False


def test_fetch_embeddings():
    dataset = deeplake.empty("./test-dataset", overwrite=True)
    dataset.create_tensor("embeddings")
    dataset.embeddings.extend([1, 2, 3, 4, 5, 6, 7, 8, 9])

    embedings = dataset_utils.fetch_embeddings("python", dataset)
    assert len(embedings) == 9

    embedings = dataset_utils.fetch_embeddings("indra", dataset)
    assert embedings is None

    embedings = dataset_utils.fetch_embeddings("db_engine", dataset)
    assert embedings is None


def test_get_embedding():
    class EmbeddingFunc:
        def embed_query(self, query):
            return np.array([0.5, 0.6, 4, 3, 5], dtype=np.float64)

    query = "tql query"
    with pytest.raises(Exception):
        embedding = dataset_utils.get_embedding(
            embedding=None, query=query, embedding_function=None
        )

    embedding_func = EmbeddingFunc()
    embedding = dataset_utils.get_embedding(
        embedding=None, query=query, embedding_function=embedding_func
    )
    assert embedding.dtype == np.float32
    assert len(embedding) == 5

    embedding_vector = np.zeros((1, 1538))
    embedding = dataset_utils.get_embedding(
        embedding=embedding_vector, query=query, embedding_function=None
    )
    assert embedding.dtype == np.float32
    assert embedding.shape == (1, 1538)


def test_preprocess_tensors():
    processed_tensors = dataset_utils.preprocess_tensors(
        ids=None, texts=None, metadatas=None, embeddings=None
    )

    assert processed_tensors["ids"] is None
    assert processed_tensors["texts"] is None
    assert processed_tensors["metadatas"] is None
    assert processed_tensors["embeddings"] is None

    ids = np.array([1, 2, 3, 4])
    texts = ["a", "b", "c", "d"]
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    embeddings = [np.array([0.1, 0.2, 0.3, 0.4])] * len(texts)
    processed_tensors = dataset_utils.preprocess_tensors(
        ids=ids, texts=texts, metadatas=metadatas, embeddings=embeddings
    )
    assert processed_tensors["ids"] == ids
    assert processed_tensors["texts"] == texts
    assert processed_tensors["metadatas"] == metadatas
    assert processed_tensors["embeddings"] == embeddings


def test_create_elements():
    ids = np.array([1, 2, 3, 4])
    texts = ["a", "b", "c", "d"]
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    embeddings = [np.array([0.1, 0.2, 0.3, 0.4])] * len(texts)

    targ_elements = [
        {
            "text": "a",
            "id": np.int64(1),
            "metadata": {"a": 1},
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        {
            "text": "b",
            "id": np.int64(2),
            "metadata": {"b": 2},
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        {
            "text": "c",
            "id": np.int64(3),
            "metadata": {"c": 3},
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        {
            "text": "d",
            "id": np.int64(4),
            "metadata": {"d": 4},
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
        },
    ]

    with pytest.raises(Exception):
        dataset_utils.create_elements(
            ids=ids, texts=texts[:2], embeddings=embeddings, metadatas=metadatas
        )
    elements = dataset_utils.create_elements(
        ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas
    )

    for i in range(len(elements)):
        assert np.array_equal(elements[i]["text"], targ_elements[i]["text"])
        assert np.array_equal(elements[i]["id"], targ_elements[i]["id"])
        assert np.array_equal(elements[i]["embedding"], targ_elements[i]["embedding"])
        assert np.array_equal(elements[i]["metadata"], targ_elements[i]["metadata"])
