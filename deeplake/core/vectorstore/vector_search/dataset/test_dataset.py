import pytest
import logging

import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore import DeepLakeVectorStore
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
    DEFAULT_VECTORSTORE_TENSORS,
)
from deeplake.tests.common import requires_libdeeplake


logger = logging.getLogger(__name__)


class Embedding:
    model = "random_model"
    deployment = "deployment"

    def embed_query(text, embedding_dim=100):
        return [0 for i in range(embedding_dim)]  # pragma: no cover


@requires_libdeeplake
def test_create(caplog, hub_cloud_dev_token):
    # dataset creation
    dataset = dataset_utils.create_or_load_dataset(
        tensor_params=DEFAULT_VECTORSTORE_TENSORS,
        dataset_path="./test-dataset",
        token=None,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="compute_engine",
        overwrite=True,
        embedding_function=Embedding,
    )
    assert len(dataset) == 0
    assert set(dataset.tensors.keys()) == {
        "embedding",
        "id",
        "metadata",
        "text",
    }
    assert dataset.embedding.info["embedding"] == {
        "model": "random_model",
        "deployment": "deployment",
        "embedding_ctx_length": None,
        "chunk_size": None,
        "max_retries": None,
    }

    dataset = dataset_utils.create_or_load_dataset(
        tensor_params=DEFAULT_VECTORSTORE_TENSORS,
        dataset_path="hub://testingacc2/vectorstore_dbengine",
        token=hub_cloud_dev_token,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="tensor_db",
        overwrite=True,
        embedding_function=Embedding,
    )
    assert len(dataset) == 0
    assert set(dataset.tensors.keys()) == {
        "embedding",
        "id",
        "metadata",
        "text",
    }
    assert dataset.embedding.info["embedding"] == {
        "model": "random_model",
        "deployment": "deployment",
        "embedding_ctx_length": None,
        "chunk_size": None,
        "max_retries": None,
    }


def test_load(caplog, hub_cloud_dev_token):
    # dataset loading
    dataset = dataset_utils.create_or_load_dataset(
        tensor_params=DEFAULT_VECTORSTORE_TENSORS,
        dataset_path="hub://testingacc2/vectorstore_test",
        creds={},
        logger=logger,
        exec_option="python",
        overwrite=False,
        read_only=True,
        token=hub_cloud_dev_token,
        embedding_function=None,
    )
    assert dataset.max_len == 10

    deeplake_vectorstore = DeepLakeVectorStore(
        path=DEFAULT_VECTORSTORE_DEEPLAKE_PATH, overwrite=True
    )

    test_logger = logging.getLogger("test_logger")
    with caplog.at_level(logging.WARNING, logger="test_logger"):
        # dataset loading
        dataset = dataset_utils.create_or_load_dataset(
            tensor_params=DEFAULT_VECTORSTORE_TENSORS,
            dataset_path=DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
            token=None,
            creds={},
            logger=test_logger,
            read_only=False,
            exec_option="python",
            embedding_function=None,
            overwrite=False,
        )
        assert (
            f"The default deeplake path location is used: {DEFAULT_VECTORSTORE_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset." in caplog.text
        )

    with pytest.raises(ValueError):
        dataset = dataset_utils.create_or_load_dataset(
            tensor_params={"name": "image", "htype": "image"},
            dataset_path=DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
            token=None,
            creds={},
            logger=test_logger,
            read_only=False,
            exec_option="python",
            embedding_function=None,
            overwrite=False,
        )

    with pytest.raises(ValueError):
        # embedding tensor doesn't exist
        dataset = deeplake.empty("local_ds", overwrite=True)
        dataset.create_tensor("id", htype="text")

        DeepLakeVectorStore(path="local_ds")

    with pytest.raises(ValueError):
        # id/ids tensor doesn't exist
        dataset = deeplake.empty("local_ds", overwrite=True)
        dataset.create_tensor("embedding", htype="embedding")

        DeepLakeVectorStore(path="local_ds")

    with pytest.raises(ValueError):
        # incorrect embedding name
        dataset = deeplake.empty("local_ds", overwrite=True)
        dataset.create_tensor("id", htype="embedding")
        dataset.create_tensor("ids", htype="text")

        DeepLakeVectorStore(path="local_ds")

    # TODO: Add back once old datasets without embedding htype are not in circulation
    # with pytest.raises(ValueError):
    #     # incorrect htype for tensor called `embedding`
    #     dataset = deeplake.empty("local_ds", overwrite=True)
    #     dataset.create_tensor("embedding", htype="text")
    #     dataset.create_tensor("ids", htype="text")

    #     DeepLakeVectorStore(path="local_ds")


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
    dataset.create_tensor("embedding")
    dataset.embedding.extend([1, 2, 3, 4, 5, 6, 7, 8, 9])

    embedings = dataset_utils.fetch_embeddings(dataset, "embedding")
    assert len(embedings) == 9


def test_embeding_data():
    query = "tql query"
    with pytest.raises(Exception):
        embedding = dataset_utils.get_embedding(
            embedding=None, query=query, embedding_function=None
        )

    embedding = dataset_utils.get_embedding(
        embedding=None,
        embedding_function=Embedding.embed_query,
        embedding_data=[query],
    )
    assert embedding.dtype == np.float32
    assert len(embedding) == 100

    embedding_vector = np.zeros((1, 1538))
    embedding = dataset_utils.get_embedding(
        embedding=embedding_vector,
        embedding_function=None,
        embedding_data=[query],
    )
    assert embedding.dtype == np.float32
    assert embedding.shape == (1, 1538)

    embedding_vector = np.zeros((1, 1538))
    embedding = dataset_utils.get_embedding(
        embedding=embedding_vector,
        embedding_function=None,
        embedding_data=[query],
    )
    assert embedding.dtype == np.float32
    assert embedding.shape == (1, 1538)

    embedding_vector = np.zeros((1, 1538))
    embedding = dataset_utils.get_embedding(
        embedding=embedding_vector,
        embedding_function=Embedding.embed_query,
        embedding_data=[query],
    )
    assert embedding.dtype == np.float32
    assert embedding.shape == (1, 1538)


def test_preprocess_tensors(local_path):
    dataset = deeplake.empty(local_path, overwrite=True)

    dataset.create_tensor("ids", htype="text")
    dataset.create_tensor("metadata", htype="json")
    dataset.create_tensor("embedding", htype="embedding")
    dataset.create_tensor("text", htype="text")

    texts = ["a", "b", "c", "d"]
    processed_tensors, ids = dataset_utils.preprocess_tensors(
        text=texts, dataset=dataset
    )

    assert len(processed_tensors["ids"]) == 4
    assert processed_tensors["text"] == texts

    texts = ("a", "b", "c", "d")
    ids = np.array([1, 2, 3, 4])
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    embeddings = [np.array([0.1, 0.2, 0.3, 0.4])] * len(texts)
    processed_tensors, _ = dataset_utils.preprocess_tensors(
        id=ids,
        text=texts,
        metadata=metadatas,
        embedding=embeddings,
        dataset=dataset,
    )
    assert np.array_equal(processed_tensors["ids"], ids)
    assert processed_tensors["text"] == list(texts)
    assert processed_tensors["metadata"] == metadatas
    assert processed_tensors["embedding"] == embeddings


def test_create_elements(local_path):
    dataset = deeplake.empty(local_path, overwrite=True)

    dataset.create_tensor("ids", htype="text")
    dataset.create_tensor("metadata", htype="json")
    dataset.create_tensor("embedding", htype="embedding")
    dataset.create_tensor("text", htype="text")

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

    processed_tensors, ids = dataset_utils.preprocess_tensors(
        dataset=dataset, id=ids, text=texts, embedding=embeddings, metadata=metadatas
    )
    elements = dataset_utils.create_elements(processed_tensors)

    for i in range(len(elements)):
        assert np.array_equal(elements[i]["text"], targ_elements[i]["text"])
        assert np.array_equal(elements[i]["id"], targ_elements[i]["id"])
        assert np.array_equal(elements[i]["embedding"], targ_elements[i]["embedding"])
        assert np.array_equal(elements[i]["metadata"], targ_elements[i]["metadata"])
