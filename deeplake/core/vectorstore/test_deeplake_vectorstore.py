import logging
import pickle
import uuid
import os
import sys
from math import isclose
from functools import partial
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.deeplake_vectorstore import (
    DeepLakeVectorStore,
    VectorStore,
)
from deeplake.core.vectorstore.embeddings.embedder import DeepLakeEmbedder
from deeplake.core.vectorstore import utils
from deeplake.tests.common import requires_libdeeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    HUB_CLOUD_DEV_USERNAME,
)
from deeplake.constants import MB
from deeplake.util.exceptions import (
    IncorrectEmbeddingShapeError,
    TensorDoesNotExistError,
    DatasetHandlerError,
    EmbeddingTensorPopError,
)
from deeplake.core.index_maintenance import (
    METRIC_TO_INDEX_METRIC,
)
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils

EMBEDDING_DIM = 100
NUMBER_OF_DATA = 10
# create data
texts, embeddings, ids, metadatas, images = utils.create_data(
    number_of_data=NUMBER_OF_DATA, embedding_dim=EMBEDDING_DIM
)

query_embedding = np.random.uniform(low=-10, high=10, size=(EMBEDDING_DIM)).astype(
    np.float32
)


class OpenAILikeEmbedder:
    def embed_documents(self, docs: List[str]):
        return [np.ones(EMBEDDING_DIM) for _ in range(len(docs))]

    def embed_query(self, query: str):
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        return np.ones(EMBEDDING_DIM)


def embedding_fn(text, embedding_dim=EMBEDDING_DIM):
    return np.zeros((len(text), EMBEDDING_DIM))  # pragma: no cover


def embedding_fn2(text, embedding_dim=EMBEDDING_DIM):
    return []  # pragma: no cover


def embedding_fn3(text, embedding_dim=EMBEDDING_DIM):
    """Returns embedding in List[np.ndarray] format"""
    return [np.zeros(embedding_dim) for i in range(len(text))]


def embedding_fn4(text, embedding_dim=EMBEDDING_DIM):
    return np.zeros((1, EMBEDDING_DIM))  # pragma: no cover


def embedding_fn5(text, embedding_dim=EMBEDDING_DIM):
    """Returns embedding in List[np.ndarray] format"""
    return [np.zeros(i) for i in range(len(text))]


def embedding_function(embedding_value, text):
    """Embedding function with custom embedding values"""
    return [np.ones(EMBEDDING_DIM) * embedding_value for _ in range(len(text))]


def get_embedding_function(embedding_value=100):
    """Function for creation embedding function with given embedding value"""
    return partial(embedding_function, embedding_value)


def get_multiple_embedding_function(embedding_value, num_of_funcs=2):
    return [
        partial(embedding_function, embedding_value[i]) for i in range(num_of_funcs)
    ]


def filter_udf(x):
    return x["metadata"].data()["value"] in [f"{i}" for i in range(5)]


def test_id_backward_compatibility(local_path):
    num_of_items = 10
    embedding_dim = 100

    ids = [f"{i}" for i in range(num_of_items)]
    # Creating embeddings of float32 as dtype of embedding tensor is float32.
    embedding = [np.zeros(embedding_dim, dtype=np.float32) for i in range(num_of_items)]
    text = ["aadfv" for i in range(num_of_items)]
    metadata = [{"key": i} for i in range(num_of_items)]

    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("ids", htype="text")
    ds.create_tensor("embedding", htype="embedding")
    ds.create_tensor("text", htype="text")
    ds.create_tensor("metadata", htype="json")

    ds.extend(
        {
            "ids": ids,
            "embedding": embedding,
            "text": text,
            "metadata": metadata,
        }
    )

    vectorstore = VectorStore(path=local_path)
    vectorstore.add(
        text=text,
        embedding=embedding,
        metadata=metadata,
    )

    assert len(vectorstore) == 2 * num_of_items


def test_custom_tensors(local_path):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {"name": "texts_custom", "htype": "text"},
            {"name": "emb_custom", "htype": "embedding"},
        ],
    )

    with pytest.raises(ValueError):
        vector_store.add(
            bad_tensor_1=texts,
            bad_tensor_2=embeddings,
            text=texts,
        )

    vector_store.add(
        texts_custom=texts,
        emb_custom=embeddings,
    )

    data = vector_store.search(
        embedding=query_embedding, exec_option="python", embedding_tensor="emb_custom"
    )
    assert len(data.keys()) == 3
    assert "texts_custom" in data.keys() and "id" in data.keys()

    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {"name": "texts_custom", "htype": "text"},
            {"name": "emb_custom", "htype": "embedding"},
        ],
        embedding_function=embedding_fn5,
    )

    with pytest.raises(IncorrectEmbeddingShapeError):
        vector_store.add(
            embedding_data=texts,
            embedding_tensor="emb_custom",
            texts_custom=texts,
        )

    texts_extended = texts * 2500
    with pytest.raises(IncorrectEmbeddingShapeError):
        vector_store.add(
            embedding_data=texts_extended,
            embedding_tensor="emb_custom",
            texts_custom=texts_extended,
        )


@pytest.mark.parametrize(
    ("path", "hub_token"),
    [
        ("local_path", "hub_cloud_dev_token"),
        ("s3_path", "hub_cloud_dev_token"),
        ("gcs_path", "hub_cloud_dev_token"),
        ("azure_path", "hub_cloud_dev_token"),
        ("hub_cloud_path", "hub_cloud_dev_token"),
    ],
    indirect=True,
)
@pytest.mark.slow
def test_providers(path, hub_token):
    vector_store = DeepLakeVectorStore(
        path=path,
        overwrite=True,
        tensor_params=[
            {"name": "texts_custom", "htype": "text"},
            {"name": "emb_custom", "htype": "embedding"},
        ],
        token=hub_token,
    )

    vector_store.add(
        texts_custom=texts,
        emb_custom=embeddings,
    )
    assert len(vector_store) == 10


@pytest.mark.slow
def test_creds(gcs_path, gcs_creds):
    # testing create dataset with creds
    vector_store = DeepLakeVectorStore(
        path=gcs_path,
        overwrite=True,
        tensor_params=[
            {"name": "texts_custom", "htype": "text"},
            {"name": "emb_custom", "htype": "embedding"},
        ],
        creds=gcs_creds,
    )

    vector_store.add(
        texts_custom=texts,
        emb_custom=embeddings,
    )
    assert len(vector_store) == 10

    # testing dataset loading with creds
    vector_store = DeepLakeVectorStore(
        path=gcs_path,
        overwrite=False,
        creds=gcs_creds,
    )
    assert len(vector_store) == 10


@pytest.mark.slow
@requires_libdeeplake
def test_search_basic(local_path, hub_cloud_dev_token, caplog):
    logging.getLogger("deeplake").propagate = True

    openai_embeddings = OpenAILikeEmbedder()
    """Test basic search features"""
    # Initialize vector store object and add data
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    assert vector_store.dataset_handler.exec_option == "compute_engine"

    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    with pytest.raises(IncorrectEmbeddingShapeError):
        vector_store.add(
            embedding_function=embedding_fn2,
            embedding_data=texts,
            text=texts,
            metadata=metadatas,
        )
    # Check that default option works
    data_default = vector_store.search(
        embedding=query_embedding,
    )
    assert (len(data_default.keys())) > 0
    # Use python implementation to search the data
    data_p = vector_store.search(
        embedding=query_embedding,
        k=2,
        return_tensors=["id", "text"],
        filter={"metadata": {"abc": 1}},
    )

    assert len(data_p["text"]) == 1
    assert (
        sum(
            [
                tensor in data_p.keys()
                for tensor in vector_store.dataset_handler.dataset.tensors
            ]
        )
        == 2
    )  # One for each return_tensors
    assert len(data_p.keys()) == 3  # One for each return_tensors + score

    # Load a vector store object from the cloud for indra testing
    vector_store_cloud = DeepLakeVectorStore(
        path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )
    assert vector_store_cloud.dataset_handler.exec_option == "compute_engine"

    # Use indra implementation to search the data
    data_ce = vector_store_cloud.search(
        embedding=query_embedding,
        k=2,
        return_tensors=["id", "text"],
    )
    assert len(data_ce["text"]) == 2
    assert (
        sum(
            [
                tensor in data_ce.keys()
                for tensor in vector_store_cloud.dataset_handler.dataset.tensors
            ]
        )
        == 2
    )  # One for each return_tensors
    assert len(data_ce.keys()) == 3  # One for each return_tensors + score

    with pytest.raises(ValueError):
        vector_store_cloud.search(
            query=f"SELECT * WHERE id=='{vector_store_cloud.dataset_handler.dataset.id[0].numpy()[0]}'",
            embedding=query_embedding,
            k=2,
            return_tensors=["id", "text"],
        )

    # Run a full custom query
    test_text = vector_store_cloud.dataset_handler.dataset.text[0].data()["value"]
    data_q = vector_store_cloud.search(
        query=f"select * where text == '{test_text}'",
    )

    assert len(data_q["text"]) == 1
    assert data_q["text"][0] == test_text
    assert sum(
        [
            tensor in data_q.keys()
            for tensor in vector_store_cloud.dataset_handler.dataset.tensors
        ]
    ) == len(
        vector_store_cloud.dataset_handler.dataset.tensors
    )  # One for each tensor - embedding + score

    # Run a filter query using a json
    data_e_j = vector_store.search(
        k=2,
        return_tensors=["id", "text"],
        filter={"metadata": metadatas[2], "text": texts[2]},
    )
    assert len(data_e_j["text"]) == 1
    assert (
        sum(
            [
                tensor in data_e_j.keys()
                for tensor in vector_store.dataset_handler.dataset.tensors
            ]
        )
        == 2
    )  # One for each return_tensors
    assert len(data_e_j.keys()) == 2

    # Run the same filter as above using a function
    def filter_fn(x):
        return x["metadata"].data()["value"]["abc"] == 1

    data_e_f = vector_store.search(
        k=2,
        return_tensors=["id", "text"],
        filter=filter_fn,
    )
    assert len(data_e_f["text"]) == 1
    assert (
        sum(
            [
                tensor in data_e_f.keys()
                for tensor in vector_store.dataset_handler.dataset.tensors
            ]
        )
        == 2
    )  # One for each return_tensors
    assert len(data_e_f.keys()) == 2

    # Run a filter query using a list
    data_e_j = vector_store.search(
        k=2,
        return_tensors=["id", "text"],
        filter={"text": texts[0:2]},
    )
    assert len(data_e_j["text"]) == 2

    # Run a filter query using a json with indra. Wrap text as list to make sure it works
    data_ce_f = vector_store_cloud.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        k=2,
        return_tensors=["id", "text"],
        filter={
            "metadata": vector_store_cloud.dataset_handler.dataset.metadata[0].data()[
                "value"
            ],
            "text": [
                vector_store_cloud.dataset_handler.dataset.text[0].data()["value"]
            ],
        },
    )
    assert len(data_ce_f["text"]) == 1
    assert (
        sum(
            [
                tensor in data_ce_f.keys()
                for tensor in vector_store_cloud.dataset_handler.dataset.tensors
            ]
        )
        == 2
    )  # One for each return_tensors
    assert len(data_ce_f.keys()) == 3  # One for each return_tensors + score

    # Check returning views
    data_p_v = vector_store.search(
        embedding=query_embedding,
        k=2,
        filter={"metadata": {"abc": 1}},
        return_view=True,
    )
    assert len(data_p_v) == 1
    assert isinstance(data_p_v.text[0].data()["value"], str)
    assert data_p_v.embedding[0].numpy().size > 0

    # Check that specifying exec option during search works, by specifying an invalid option
    with pytest.raises(ValueError):
        vector_store.search(
            embedding=query_embedding,
            exec_option="tensor_db",
            k=2,
            filter={"metadata": {"abc": 1}},
            return_view=True,
        )

    data_ce_v = vector_store_cloud.search(
        embedding=query_embedding, k=2, return_view=True
    )
    assert len(data_ce_v) == 2
    assert isinstance(data_ce_v.text[0].data()["value"], str)
    assert data_ce_v.embedding[0].numpy().size > 0

    # Check that None option works
    vector_store_none_exec = DeepLakeVectorStore(
        path=local_path, overwrite=True, token=hub_cloud_dev_token, exec_option=None
    )

    assert vector_store_none_exec.dataset_handler.exec_option == "compute_engine"

    # Check that filter_fn with cloud dataset (and therefore "compute_engine" exec option) switches to "python" automatically.
    _ = vector_store_cloud.search(
        filter=filter_fn,
    )
    assert (
        'Switching exec_option to "python" (runs on client) because filter is specified as a function.'
        in caplog.text
    )

    # Check exceptions
    # Invalid exec option
    with pytest.raises(ValueError):
        vector_store.search(
            embedding=query_embedding, exec_option="invalid_exec_option"
        )
    # Search without parameters
    with pytest.raises(ValueError):
        vector_store.search()
    # Query with python exec_option
    with pytest.raises(ValueError):
        vector_store.search(query="dummy", exec_option="python")
    # Returning a tensor that does not exist
    with pytest.raises(TensorDoesNotExistError):
        vector_store.search(
            embedding=query_embedding,
            return_tensors=["non_existant_tensor"],
        )
    # Specifying return tensors is not valid when also specifying a query
    with pytest.raises(ValueError):
        vector_store_cloud.search(query="dummy", return_tensors=["id"])
    # Specifying a filter function is not valid when also specifying a query
    with pytest.raises(ValueError):
        vector_store_cloud.search(query="dummy", filter=filter_fn)
    # Specifying a filter function is not valid when exec_option is "compute_engine"
    with pytest.raises(ValueError):
        vector_store_cloud.search(
            embedding=query_embedding, filter=filter_fn, exec_option="compute_engine"
        )
    # Not specifying a query or data that should be embedded
    with pytest.raises(ValueError):
        vector_store.search(
            embedding_function=embedding_fn,
        )
    # Empty dataset cannot be queried
    with pytest.raises(ValueError):
        vector_store_empty = DeepLakeVectorStore(path="mem://xyz")
        vector_store_empty.search(
            embedding=query_embedding,
            k=2,
            filter={"metadata": {"abc": 1}},
            return_view=True,
        )

    vector_store = DeepLakeVectorStore(path="mem://xyz")
    assert vector_store.dataset_handler.exec_option == "python"
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    data = vector_store.search(
        embedding_function=openai_embeddings.embed_query,
        embedding_data=["dummy"],
        return_view=True,
        k=2,
    )
    assert len(data) == 2
    assert isinstance(data.text[0].data()["value"], str)
    assert data.embedding[0].numpy().size > 0

    data = vector_store.search(
        embedding_function=openai_embeddings.embed_query,
        embedding_data="dummy",
        return_view=True,
        k=2,
    )
    assert len(data) == 2
    assert isinstance(data.text[0].data()["value"], str)
    assert data.embedding[0].numpy().size > 0

    with pytest.raises(NotImplementedError):
        data = vector_store.search(
            embedding_function=embedding_fn3,
            embedding_data=["dummy", "dummy2"],
            return_view=True,
            k=2,
        )

    data = vector_store.search(
        filter={"metadata": {"abcdefh": 1}},
        embedding=None,
        return_view=True,
        k=2,
    )
    assert len(data) == 0

    data = vector_store.search(
        filter={"metadata": {"abcdefh": 1}},
        embedding=query_embedding,
        k=2,
    )
    assert len(data) == 4
    assert len(data["id"]) == 0
    assert len(data["metadata"]) == 0
    assert len(data["text"]) == 0
    assert len(data["score"]) == 0

    # Test that the embedding function during initalization works
    vector_store = DeepLakeVectorStore(
        path="mem://xyz", embedding_function=openai_embeddings
    )
    assert vector_store.dataset_handler.exec_option == "python"
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)
    result = vector_store.search(embedding_data=["dummy"])
    assert len(result) == 4


@pytest.mark.slow
@requires_libdeeplake
def test_index_basic(local_path, hub_cloud_dev_token, caplog):
    logging.getLogger("deeplake").propagate = True
    # Start by testing behavior without an index
    vector_store = VectorStore(
        path=local_path,
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    assert vector_store.dataset_handler.distance_metric_index is None

    # Then test behavior when index is added
    vector_store = VectorStore(
        path=local_path, token=hub_cloud_dev_token, index_params={"threshold": 1}
    )

    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)
    es = vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()

    assert (
        es[0]["distance"] == METRIC_TO_INDEX_METRIC[DEFAULT_VECTORSTORE_DISTANCE_METRIC]
    )

    # Then test behavior when index is added previously and the dataset is reloaded
    vector_store = VectorStore(path=local_path, token=hub_cloud_dev_token)
    es = vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()

    assert (
        es[0]["distance"] == METRIC_TO_INDEX_METRIC[DEFAULT_VECTORSTORE_DISTANCE_METRIC]
    )

    # Test index with sample updates
    pre_update_index = vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[
        0
    ]
    vector_store.add(
        embedding=[embeddings[0]], text=[texts[0]], metadata=[metadatas[0]]
    )
    post_update_index = (
        vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[0]
    )

    assert pre_update_index == post_update_index

    # Test index with sample deletion
    pre_delete_index = vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[
        0
    ]
    vector_store.delete(row_ids=[len(vector_store) - 1])
    post_delete_index = (
        vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[0]
    )

    assert pre_delete_index == post_delete_index

    # Test index with sample updating
    pre_update_index = vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[
        0
    ]
    vector_store.update_embedding(row_ids=[0], embedding_function=embedding_fn)
    post_update_index = (
        vector_store.dataset_handler.dataset.embedding.get_vdb_indexes()[0]
    )

    assert pre_update_index == post_update_index

    # Check that distance metric throws a warning when there is an index
    vector_store.search(embedding=query_embedding, distance_metric="l1")
    assert (
        "The specified `distance_metric': `l1` does not match the distance metric in the index:"
        in caplog.text
    )


@pytest.mark.slow
@requires_libdeeplake
@pytest.mark.parametrize("distance_metric", ["L1", "L2", "COS", "MAX"])
def test_search_quantitative(distance_metric, hub_cloud_dev_token):
    """Test whether TQL and Python return the same results"""
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )

    # use python implementation to search the data
    data_p = vector_store.search(
        embedding=query_embedding, exec_option="python", distance_metric=distance_metric
    )

    # use indra implementation to search the data
    data_ce = vector_store.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        distance_metric=distance_metric,
    )

    assert len(data_p["score"]) == len(data_ce["score"])
    assert all(
        [
            isclose(
                data_p["score"][i],
                data_ce["score"][i],
                abs_tol=0.00001
                * (abs(data_p["score"][i]) + abs(data_ce["score"][i]))
                / 2,
            )
            for i in range(len(data_p["score"]))
        ]
    )
    assert data_p["text"] == data_ce["text"]
    assert data_p["id"] == data_ce["id"]
    assert data_p["metadata"] == data_ce["metadata"]

    # use indra implementation to search the data
    data_ce_f = vector_store.search(
        embedding=None,
        exec_option="compute_engine",
        distance_metric=distance_metric,
        filter={"metadata": {"abc": 100}},
    )

    # All medatata are the same to this should return k (k) results
    assert len(data_ce_f["id"]) == 4

    with pytest.raises(ValueError):
        # use indra implementation to search the data
        vector_store.search(
            query="select * where metadata == {'abcdefg': 28}",
            exec_option="compute_engine",
            distance_metric=distance_metric,
            filter={"metadata": {"abcdefg": 28}},
        )

    test_id = vector_store.dataset_handler.dataset.id[0].data()["value"]

    data_ce_q = vector_store.search(
        query=f"select * where id == '{test_id}'",
        exec_option="compute_engine",
    )
    assert data_ce_q["id"][0] == test_id


@requires_libdeeplake
@pytest.mark.slow
def test_search_managed(hub_cloud_dev_token):
    """Test whether managed TQL and client-side TQL return the same results"""
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path="hub://testingacc2/vectorstore_test_managed",
        read_only=True,
        token=hub_cloud_dev_token,
    )

    # use indra implementation to search the data
    data_ce = vector_store.search(
        embedding=query_embedding,
        exec_option="compute_engine",
    )

    data_db = vector_store.search(
        embedding=query_embedding,
        exec_option="tensor_db",
    )

    assert "vectordb/" in vector_store.dataset_handler.dataset.base_storage.path

    assert len(data_ce["score"]) == len(data_db["score"])
    assert all(
        [
            isclose(
                data_ce["score"][i],
                data_db["score"][i],
                abs_tol=0.00001
                * (abs(data_ce["score"][i]) + abs(data_db["score"][i]))
                / 2,
            )
            for i in range(len(data_ce["score"]))
        ]
    )
    assert data_ce["text"] == data_db["text"]
    assert data_ce["id"] == data_db["id"]


@requires_libdeeplake
def test_delete(local_path, hub_cloud_dev_token):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=False,
    )

    # add data to the dataset:
    vector_store.add(id=ids, embedding=embeddings, text=texts, metadata=metadatas)
    assert_vectorstore_structure(vector_store, 10)

    # delete the data in the dataset by id:
    vector_store.delete(row_ids=[4, 8, 9])
    assert len(vector_store.dataset_handler.dataset) == NUMBER_OF_DATA - 3

    vector_store.delete(filter={"metadata": {"abc": 1}})
    assert len(vector_store.dataset_handler.dataset) == NUMBER_OF_DATA - 4

    vector_store.delete(ids=["7"])
    assert len(vector_store.dataset_handler.dataset) == NUMBER_OF_DATA - 5

    with pytest.raises(ValueError):
        vector_store.delete()

    tensors_before_delete = vector_store.dataset_handler.dataset.tensors
    vector_store.delete(delete_all=True)
    assert len(vector_store.dataset_handler.dataset) == 0
    assert (
        vector_store.dataset_handler.dataset.tensors.keys()
        == tensors_before_delete.keys()
    )

    vector_store.delete_by_path(local_path)
    dirs = os.listdir("./")
    assert local_path not in dirs

    # backwards compatibility test:
    vector_store_b = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        exec_option="compute_engine",
        tensor_params=[
            {
                "name": "ids",
                "htype": "text",
            },
            {
                "name": "docs",
                "htype": "text",
            },
        ],
        token=hub_cloud_dev_token,
    )
    # add data to the dataset:
    vector_store_b.add(ids=ids, docs=texts)

    # delete the data in the dataset by id:
    vector_store_b.delete(row_ids=[0])
    assert len(vector_store_b.dataset_handler.dataset) == NUMBER_OF_DATA - 1

    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("id", htype="text")
    ds.create_tensor("embedding", htype="embedding")
    ds.extend(
        {
            "id": ids,
            "embedding": embeddings,
        }
    )

    vector_store = DeepLakeVectorStore(
        path=local_path,
        exec_option="compute_engine",
        token=hub_cloud_dev_token,
    )
    vector_store.delete(ids=ids[:3])
    assert len(vector_store) == NUMBER_OF_DATA - 3


def assert_updated_vector_store(
    new_embedding_value,
    vector_store,
    ids,
    row_ids,
    filters,
    query,
    embedding_function,
    embedding_source_tensor,
    embedding_tensor,
    exec_option,
    num_changed_samples=3,
):
    if isinstance(embedding_tensor, str):
        new_embeddings = [
            np.ones(EMBEDDING_DIM) * new_embedding_value
        ] * num_changed_samples
    else:
        new_embeddings = []
        for i in range(len(embedding_tensor)):
            new_embedding = [
                np.ones(EMBEDDING_DIM) * new_embedding_value[i]
            ] * num_changed_samples
            new_embeddings.append(new_embedding)

    if not row_ids:
        row_ids = dataset_utils.search_row_ids(
            dataset=vector_store.dataset,
            search_fn=vector_store.search,
            ids=ids,
            filter=filters,
            query=query,
            exec_option=exec_option,
        )

    if callable(embedding_function) and isinstance(embedding_tensor, str):
        np.testing.assert_array_equal(
            vector_store.dataset_handler.dataset[embedding_tensor][row_ids].numpy(),
            new_embeddings,
        )

    if callable(embedding_function) and isinstance(embedding_tensor, list):
        for i in range(len(embedding_tensor)):
            np.testing.assert_array_equal(
                vector_store.dataset_handler.dataset[embedding_tensor[i]][
                    row_ids
                ].numpy(),
                new_embeddings[i],
            )

    if isinstance(embedding_function, list) and isinstance(embedding_tensor, list):
        for i in range(len(embedding_tensor)):
            np.testing.assert_array_equal(
                vector_store.dataset_handler.dataset[embedding_tensor[i]][
                    row_ids
                ].numpy(),
                new_embeddings[i],
            )


# TODO: refactor this method:
# 1. Split this method into multiple methods (1 test per 1 behavior)
# 2. use create_and_populate_vs to make these tests more readable
# 3. create one fixture for these nested fixtures
@requires_libdeeplake
@pytest.mark.parametrize(
    "ds, vector_store_hash_ids, vector_store_row_ids, vector_store_filters, vector_store_filter_udf, vector_store_query, hub_cloud_dev_token",
    [
        (
            "local_auth_ds",
            "vector_store_hash_ids",
            None,
            None,
            None,
            None,
            "hub_cloud_dev_token",
        ),
        (
            "local_auth_ds",
            None,
            "vector_store_row_ids",
            None,
            None,
            None,
            "hub_cloud_dev_token",
        ),
        (
            "local_auth_ds",
            None,
            None,
            None,
            "vector_store_filter_udf",
            None,
            "hub_cloud_dev_token",
        ),
        (
            "local_auth_ds",
            None,
            None,
            "vector_store_filters",
            None,
            None,
            "hub_cloud_dev_token",
        ),
        (
            "hub_cloud_ds",
            None,
            None,
            None,
            None,
            "vector_store_query",
            "hub_cloud_dev_token",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("init_embedding_function", [embedding_fn3, None])
@pytest.mark.slow
@requires_libdeeplake
def test_update_embedding(
    ds,
    vector_store_hash_ids,
    vector_store_row_ids,
    vector_store_filters,
    vector_store_filter_udf,
    vector_store_query,
    init_embedding_function,
    hub_cloud_dev_token,
):
    vector_store_filters = vector_store_filters or vector_store_filter_udf

    exec_option = "compute_engine"
    if vector_store_filter_udf:
        exec_option = "python"

    embedding_tensor = "embedding"
    embedding_source_tensor = "text"
    # dataset has a single embedding_tensor:

    path = ds.path
    vector_store = DeepLakeVectorStore(
        path=path,
        overwrite=True,
        verbose=False,
        exec_option=exec_option,
        embedding_function=init_embedding_function,
        index_params={"threshold": 10},
        token=hub_cloud_dev_token,
    )

    # add data to the dataset:
    metadatas[1:6] = [{"a": 1} for _ in range(5)]
    vector_store.add(id=ids, embedding=embeddings, text=texts, metadata=metadatas)

    # case 1: single embedding_source_tensor, single embedding_tensor, single embedding_function
    new_embedding_value = 100
    embedding_fn = get_embedding_function(embedding_value=new_embedding_value)
    vector_store.update_embedding(
        ids=vector_store_hash_ids,
        row_ids=vector_store_row_ids,
        filter=vector_store_filters,
        query=vector_store_query,
        embedding_function=embedding_fn,
        embedding_source_tensor=embedding_source_tensor,
        embedding_tensor=embedding_tensor,
    )
    assert_updated_vector_store(
        new_embedding_value,
        vector_store,
        vector_store_hash_ids,
        vector_store_row_ids,
        vector_store_filters,
        vector_store_query,
        embedding_fn,
        embedding_source_tensor,
        embedding_tensor,
        exec_option,
        num_changed_samples=5,
    )

    # case 2: single embedding_source_tensor, single embedding_tensor not specified, single embedding_function
    new_embedding_value = 100
    embedding_fn = get_embedding_function(embedding_value=new_embedding_value)
    vector_store.update_embedding(
        ids=vector_store_hash_ids,
        row_ids=vector_store_row_ids,
        filter=vector_store_filters,
        query=vector_store_query,
        embedding_function=embedding_fn,
        embedding_source_tensor=embedding_source_tensor,
    )
    assert_updated_vector_store(
        new_embedding_value,
        vector_store,
        vector_store_hash_ids,
        vector_store_row_ids,
        vector_store_filters,
        vector_store_query,
        embedding_fn,
        embedding_source_tensor,
        embedding_tensor,
        exec_option,
        num_changed_samples=5,
    )

    # case 3-4: single embedding_source_tensor, single embedding_tensor, single init_embedding_function
    if init_embedding_function is None:
        # case 3: errors out when init_embedding_function is not specified
        with pytest.raises(ValueError):
            vector_store.update_embedding(
                ids=vector_store_hash_ids,
                row_ids=vector_store_row_ids,
                filter=vector_store_filters,
                query=vector_store_query,
                embedding_source_tensor=embedding_source_tensor,
            )
    else:
        # case 4
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_source_tensor=embedding_source_tensor,
        )
        assert_updated_vector_store(
            0,
            vector_store,
            vector_store_hash_ids,
            vector_store_row_ids,
            vector_store_filters,
            vector_store_query,
            init_embedding_function,
            embedding_source_tensor,
            embedding_tensor,
            exec_option,
            num_changed_samples=5,
        )

    vector_store.delete_by_path(path, token=ds.token)

    # dataset has a multiple embedding_tensor:
    tensors = [
        {
            "name": "text",
            "htype": "text",
            "create_id_tensor": False,
            "create_sample_info_tensor": False,
            "create_shape_tensor": False,
        },
        {
            "name": "metadata",
            "htype": "json",
            "create_id_tensor": False,
            "create_sample_info_tensor": False,
            "create_shape_tensor": False,
        },
        {
            "name": "embedding",
            "htype": "embedding",
            "dtype": np.float32,
            "create_id_tensor": False,
            "create_sample_info_tensor": False,
            "create_shape_tensor": True,
            "max_chunk_size": 64 * MB,
        },
        {
            "name": "embedding_md",
            "htype": "embedding",
            "dtype": np.float32,
            "create_id_tensor": False,
            "create_sample_info_tensor": False,
            "create_shape_tensor": True,
            "max_chunk_size": 64 * MB,
        },
        {
            "name": "id",
            "htype": "text",
            "create_id_tensor": False,
            "create_sample_info_tensor": False,
            "create_shape_tensor": False,
        },
    ]
    multiple_embedding_tensor = ["embedding", "embedding_md"]
    multiple_embedding_source_tensor = ["embedding", "metadata"]
    vector_store = DeepLakeVectorStore(
        path=path + "_multi",
        overwrite=True,
        verbose=False,
        embedding_function=init_embedding_function,
        tensor_params=tensors,
        token=ds.token,
        exec_option=exec_option,
    )

    vector_store.add(
        id=ids,
        text=texts,
        embedding=embeddings,
        embedding_md=embeddings,
        metadata=metadatas,
    )

    # case 1: multiple embedding_source_tensor, single embedding_tensor, single embedding_function
    new_embedding_value = [100, 200]
    embedding_fn = get_multiple_embedding_function(new_embedding_value)
    with pytest.raises(ValueError):
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_function=embedding_function,
            embedding_source_tensor=multiple_embedding_source_tensor,
            embedding_tensor=embedding_tensor,
        )

    # case 2: multiple embedding_source_tensor, single embedding_tensor, multiple embedding_function -> error out?
    with pytest.raises(ValueError):
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_function=embedding_fn,
            embedding_source_tensor=multiple_embedding_source_tensor,
            embedding_tensor=embedding_tensor,
        )

    # case 3: 4 embedding_source_tensor, 2 embedding_tensor, 2 embedding_function
    with pytest.raises(ValueError):
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_function=embedding_fn,
            embedding_source_tensor=multiple_embedding_source_tensor * 2,
            embedding_tensor=embedding_tensor,
        )

    # case 4: multiple embedding_source_tensor, multiple embedding_tensor, multiple embedding_function
    new_embedding_value = [100, 200]
    embedding_fn = get_multiple_embedding_function(new_embedding_value)
    vector_store.update_embedding(
        ids=vector_store_hash_ids,
        row_ids=vector_store_row_ids,
        filter=vector_store_filters,
        query=vector_store_query,
        embedding_function=embedding_fn,
        embedding_source_tensor=multiple_embedding_source_tensor,
        embedding_tensor=multiple_embedding_tensor,
    )

    assert_updated_vector_store(
        new_embedding_value,
        vector_store,
        vector_store_hash_ids,
        vector_store_row_ids,
        vector_store_filters,
        vector_store_query,
        embedding_fn,
        multiple_embedding_source_tensor,
        multiple_embedding_tensor,
        exec_option,
        num_changed_samples=5,
    )

    # case 5-6: multiple embedding_source_tensor, multiple embedding_tensor, single init_embedding_function
    new_embedding_value = [0, 0]

    if init_embedding_function is None:
        with pytest.raises(ValueError):
            # case 5: error out because no embedding function was specified
            vector_store.update_embedding(
                ids=vector_store_hash_ids,
                row_ids=vector_store_row_ids,
                filter=vector_store_filters,
                query=vector_store_query,
                embedding_source_tensor=multiple_embedding_source_tensor,
                embedding_tensor=multiple_embedding_tensor,
            )
    else:
        # case 6
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_source_tensor=multiple_embedding_source_tensor,
            embedding_tensor=multiple_embedding_tensor,
        )
        assert_updated_vector_store(
            new_embedding_value,
            vector_store,
            vector_store_hash_ids,
            vector_store_row_ids,
            vector_store_filters,
            vector_store_query,
            embedding_fn3,
            multiple_embedding_source_tensor,
            multiple_embedding_tensor,
            exec_option,
            num_changed_samples=5,
        )

    # case 7: multiple embedding_source_tensor, not specified embedding_tensor, multiple embedding_function -> error out?
    with pytest.raises(ValueError):
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_source_tensor=multiple_embedding_source_tensor,
            embedding_function=embedding_fn,
        )

    # case 8-9: single embedding_source_tensor, multiple embedding_tensor, single init_embedding_function
    with pytest.raises(ValueError):
        # case 8: error out because embedding_function is not specified during init call and update call
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_source_tensor=embedding_source_tensor,
            embedding_function=embedding_fn,
        )

    # case 10: single embedding_source_tensor, multiple embedding_tensor,  multiple embedding_function -> error out?
    with pytest.raises(ValueError):
        # error out because single embedding_source_tensor is specified
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=multiple_embedding_tensor,
            embedding_function=embedding_fn,
        )

    # case 11: single embedding_source_tensor, single embedding_tensor, single embedding_function, single init_embedding_function
    new_embedding_value = 300
    embedding_fn = get_embedding_function(new_embedding_value)
    vector_store.update_embedding(
        ids=vector_store_hash_ids,
        row_ids=vector_store_row_ids,
        filter=vector_store_filters,
        query=vector_store_query,
        embedding_source_tensor=embedding_source_tensor,
        embedding_tensor=embedding_tensor,
        embedding_function=embedding_fn,
    )

    assert_updated_vector_store(
        new_embedding_value,
        vector_store,
        vector_store_hash_ids,
        vector_store_row_ids,
        vector_store_filters,
        vector_store_query,
        embedding_function,
        embedding_source_tensor,
        embedding_tensor,
        exec_option,
        num_changed_samples=5,
    )
    vector_store.delete_by_path(path + "_multi", token=ds.token)


def create_and_populate_vs(
    path,
    token=None,
    overwrite=True,
    verbose=False,
    exec_option="compute_engine",
    index_params={"threshold": -1},
    number_of_data=NUMBER_OF_DATA,
    runtime=None,
):
    # if runtime specified and tensor_db is enabled, then set exec_option to None
    if runtime and runtime.get("tensor_db", False):
        exec_option = None

    vector_store = DeepLakeVectorStore(
        path=path,
        overwrite=overwrite,
        verbose=verbose,
        exec_option=exec_option,
        index_params=index_params,
        token=token,
        runtime=runtime,
    )

    utils.create_data(number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM)

    # add data to the dataset:
    metadatas[1:6] = [{"a": 1} for _ in range(5)]
    vector_store.add(id=ids, embedding=embeddings, text=texts, metadata=metadatas)
    return vector_store


@requires_libdeeplake
def test_update_embedding_row_ids_and_ids_specified_should_throw_exception(
    local_path,
    vector_store_hash_ids,
    vector_store_row_ids,
    hub_cloud_dev_token,
):
    # specifying both row_ids and ids during update embedding should throw an exception
    # initializing vectorstore and populating it:
    vector_store = create_and_populate_vs(
        local_path,
        token=hub_cloud_dev_token,
    )
    embedding_fn = get_embedding_function()

    # calling update_embedding with both ids and row_ids being specified
    with pytest.raises(ValueError):
        vector_store.update_embedding(
            ids=vector_store_hash_ids,
            row_ids=vector_store_row_ids,
            embedding_function=embedding_fn,
        )


@requires_libdeeplake
def test_update_embedding_row_ids_and_filter_specified_should_throw_exception(
    local_path,
    vector_store_filters,
    vector_store_row_ids,
    hub_cloud_dev_token,
):
    # specifying both row_ids and filter during update embedding should throw an exception
    # initializing vectorstore and populating it:
    vector_store = create_and_populate_vs(
        local_path,
        token=hub_cloud_dev_token,
    )
    embedding_fn = get_embedding_function()

    with pytest.raises(ValueError):
        vector_store.update_embedding(
            row_ids=vector_store_row_ids,
            filter=vector_store_filters,
            embedding_function=embedding_fn,
        )


@requires_libdeeplake
def test_update_embedding_query_and_filter_specified_should_throw_exception(
    local_path,
    vector_store_filters,
    vector_store_query,
    hub_cloud_dev_token,
):
    # initializing vectorstore and populating it:
    vector_store = create_and_populate_vs(
        local_path,
        token=hub_cloud_dev_token,
    )
    embedding_fn = get_embedding_function()

    # calling update_embedding with both query and filter being specified

    with pytest.raises(ValueError):
        vector_store.update_embedding(
            filter=vector_store_filters,
            query=vector_store_query,
            embedding_function=embedding_fn,
        )


@requires_libdeeplake
def test_vdb_index_creation(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 1000
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 200, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=embeddings, text=texts, id=ids, metadata=metadatas)

    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    ds = vector_store.dataset_handler.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_incr_maint(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 1000
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    txt1 = texts[:250]
    md1 = metadatas[:250]
    ids1 = ids[:250]
    emb1 = embeddings[:250]

    txt2 = texts[250:500]
    md2 = metadatas[250:500]
    ids2 = ids[250:500]
    emb2 = embeddings[250:500]

    txt3 = texts[500:750]
    md3 = metadatas[500:750]
    ids3 = ids[500:750]
    emb3 = embeddings[500:750]

    txt4 = texts[750:1000]
    md4 = metadatas[750:1000]
    ids4 = ids[750:1000]
    emb4 = embeddings[750:1000]

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 200, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=emb1, text=txt1, id=ids1, metadata=md1)
    vector_store.add(embedding=emb2, text=txt2, id=ids2, metadata=md2)
    vector_store.add(embedding=emb3, text=txt3, id=ids3, metadata=md3)
    vector_store.add(embedding=emb4, text=txt4, id=ids4, metadata=md4)

    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    ds = vector_store.dataset_handler.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # search the embeddings.
    query1 = ds.embedding[1].numpy()
    query300 = ds.embedding[300].numpy()
    query700 = ds.embedding[700].numpy()

    s1 = ",".join(str(c) for c in query1)
    view1 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s1}]) DESC limit 1"
    )
    res1 = list(view1.sample_indices)
    assert res1[0] == 1

    s300 = ",".join(str(c) for c in query300)
    view300 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s300}]) DESC limit 1"
    )
    res300 = list(view300.sample_indices)
    assert res300[0] == 300

    s700 = ",".join(str(c) for c in query700)
    view700 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s700}]) DESC limit 1"
    )
    res700 = list(view700.sample_indices)
    assert res700[0] == 700

    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_incr_maint_extend(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 103
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    txt1 = texts[:100]
    md1 = metadatas[:100]
    ids1 = ids[:100]
    emb1 = embeddings[:100]

    txt2 = texts[100:101]
    md2 = metadatas[100:101]
    ids2 = ids[100:101]
    emb2 = embeddings[100:101]

    txt3 = texts[101:102]
    md3 = metadatas[101:102]
    ids3 = ids[101:102]
    emb3 = embeddings[101:102]

    txt4 = texts[102:103]
    md4 = metadatas[102:103]
    ids4 = ids[102:103]
    emb4 = embeddings[102:103]

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 50, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=emb1, text=txt1, id=ids1, metadata=md1)
    ds = vector_store.dataset_handler.dataset
    ds.extend({"embedding": emb2, "text": txt2, "id": ids2, "metadata": md2})
    ds.extend({"embedding": emb3, "text": txt3, "id": ids3, "metadata": md3})
    ds.extend({"embedding": emb4, "text": txt4, "id": ids4, "metadata": md4})

    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    ds = vector_store.dataset_handler.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # search the embeddings.
    query1 = ds.embedding[1].numpy()
    query101 = ds.embedding[101].numpy()
    query102 = ds.embedding[102].numpy()

    s1 = ",".join(str(c) for c in query1)
    view1 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s1}]) DESC limit 1"
    )
    res1 = list(view1.sample_indices)
    assert res1[0] == 1

    s101 = ",".join(str(c) for c in query101)
    view101 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s101}]) DESC limit 1"
    )
    res101 = list(view101.sample_indices)
    assert res101[0] == 101

    s102 = ",".join(str(c) for c in query102)
    view102 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s102}]) DESC limit 1"
    )
    res102 = list(view102.sample_indices)
    assert res102[0] == 102

    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_incr_maint_append_pop(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 103
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    txt1 = texts[99]
    md1 = metadatas[99]
    ids1 = ids[99]
    emb1 = embeddings[99]

    txt2 = texts[100]
    md2 = metadatas[100]
    ids2 = ids[100]
    emb2 = embeddings[100]

    txt3 = texts[101]
    md3 = metadatas[101]
    ids3 = ids[101]
    emb3 = embeddings[101]

    txt4 = texts[102]
    md4 = metadatas[102]
    ids4 = ids[102]
    emb4 = embeddings[102]

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 2, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    ds = vector_store.dataset_handler.dataset
    ds.append({"embedding": emb1, "text": txt1, "id": ids1, "metadata": md1})
    ds.append({"embedding": emb2, "text": txt2, "id": ids2, "metadata": md2})
    ds.append({"embedding": emb3, "text": txt3, "id": ids3, "metadata": md3})
    ds.append({"embedding": emb4, "text": txt4, "id": ids4, "metadata": md4})

    # assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    # ds = vector_store.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # search the embeddings.
    query1 = ds.embedding[1].numpy()
    query2 = ds.embedding[2].numpy()
    query3 = ds.embedding[3].numpy()

    s1 = ",".join(str(c) for c in query1)
    view1 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s1}]) DESC limit 1"
    )
    res1 = list(view1.sample_indices)
    assert res1[0] == 1

    s2 = ",".join(str(c) for c in query2)
    view2 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s2}]) DESC limit 1"
    )
    res2 = list(view2.sample_indices)
    assert res2[0] == 2

    s3 = ",".join(str(c) for c in query3)
    view3 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s3}]) DESC limit 1"
    )
    res3 = list(view3.sample_indices)
    assert res3[0] == 3

    with pytest.raises(EmbeddingTensorPopError):
        vector_store.dataset.embedding.pop(2)
        vector_store.dataset.id.pop(2)
        vector_store.dataset.metadata.pop(2)
        vector_store.dataset.text.pop(2)
    with pytest.raises(EmbeddingTensorPopError):
        vector_store.dataset.pop(2)
    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_incr_maint_update(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 105
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    txt1 = texts[:100]
    md1 = metadatas[:100]
    ids1 = ids[:100]
    emb1 = embeddings[:100]

    txt2 = texts[100]
    md2 = metadatas[100]
    ids2 = ids[100]
    emb2 = embeddings[100]

    txt3 = texts[101]
    md3 = metadatas[101]
    ids3 = ids[101]
    emb3 = embeddings[101]

    txt4 = texts[102]
    md4 = metadatas[102]
    ids4 = ids[102]
    emb4 = embeddings[102]

    emb5 = embeddings[103]
    emb6 = embeddings[104]

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 2, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=emb1, text=txt1, id=ids1, metadata=md1)
    ds = vector_store.dataset_handler.dataset
    ds.append({"embedding": emb2, "text": txt2, "id": ids2, "metadata": md2})
    ds.append({"embedding": emb3, "text": txt3, "id": ids3, "metadata": md3})
    ds.append({"embedding": emb4, "text": txt4, "id": ids4, "metadata": md4})

    # assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    # ds = vector_store.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # search the embeddings.
    query1 = ds.embedding[1].numpy()
    query2 = ds.embedding[2].numpy()
    query3 = ds.embedding[3].numpy()

    s1 = ",".join(str(c) for c in query1)
    view1 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s1}]) DESC limit 1"
    )
    res1 = list(view1.sample_indices)
    assert res1[0] == 1

    s2 = ",".join(str(c) for c in query2)
    view2 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s2}]) DESC limit 1"
    )
    res2 = list(view2.sample_indices)
    assert res2[0] == 2

    s3 = ",".join(str(c) for c in query3)
    view3 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s3}]) DESC limit 1"
    )
    res3 = list(view3.sample_indices)
    assert res3[0] == 3

    ds[3].update({"embedding": emb5})
    query3 = ds.embedding[3].numpy()
    s3 = ",".join(str(c) for c in query3)
    view3 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s3}]) DESC limit 1"
    )
    res3 = list(view3.sample_indices)
    assert res3[0] == 3

    ds.embedding[4] = emb6
    query4 = ds.embedding[4].numpy()
    s4 = ",".join(str(c) for c in query4)
    view4 = ds.query(
        f"select *  order by cosine_similarity(embedding ,array[{s4}]) DESC limit 1"
    )
    res4 = list(view4.sample_indices)
    assert res4[0] == 4

    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_incr_maint_tensor_append(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 105
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    txt1 = texts[:100]
    md1 = metadatas[:100]
    ids1 = ids[:100]
    emb1 = embeddings[:100]

    txt2 = texts[100]
    md2 = metadatas[100]
    ids2 = ids[100]
    emb2 = embeddings[100]

    txt3 = texts[101]
    md3 = metadatas[101]
    ids3 = ids[101]
    emb3 = embeddings[101]

    txt4 = texts[102]
    md4 = metadatas[102]
    ids4 = ids[102]
    emb4 = embeddings[102]
    emb5 = embeddings[104]

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 2, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=emb1, text=txt1, id=ids1, metadata=md1)
    ds = vector_store.dataset_handler.dataset

    ds.embedding.append(emb2)
    ds.embedding.append(emb3)
    ds.embedding.append(emb4)
    # ds.embedding[104] = emb5

    # assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    # ds = vector_store.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    # # search the embeddings.
    # query1 = ds.embedding[100].numpy()
    # query2 = ds.embedding[101].numpy()
    # query3 = ds.embedding[102].numpy()
    #
    # s1 = ",".join(str(c) for c in query1)
    # view1 = ds.query(
    #     f"select *  order by cosine_similarity(embedding ,array[{s1}]) DESC limit 1"
    # )
    # res1 = list(view1.sample_indices)
    # assert res1[0] == 100
    #
    # s2 = ",".join(str(c) for c in query2)
    # view2 = ds.query(
    #     f"select *  order by cosine_similarity(embedding ,array[{s2}]) DESC limit 1"
    # )
    # res2 = list(view2.sample_indices)
    # assert res2[0] == 101
    #
    # s3 = ",".join(str(c) for c in query3)
    # view3 = ds.query(
    #     f"select *  order by cosine_similarity(embedding ,array[{s3}]) DESC limit 1"
    # )
    # res3 = list(view3.sample_indices)
    # assert res3[0] == 102

    vector_store.delete_by_path(local_path, token=ds.token)


@requires_libdeeplake
def test_vdb_index_like(local_path, capsys, hub_cloud_dev_token):
    number_of_data = 1000
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    # initialize vector store object with vdb index threshold as 200.
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
        exec_option="compute_engine",
        index_params={"threshold": 200, "distance_metric": "L2"},
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=embeddings, text=texts, id=ids, metadata=metadatas)

    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset.tensors) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )
    assert set(vector_store.tensors()) == set(
        [
            "embedding",
            "id",
            "metadata",
            "text",
        ]
    )

    # Check if the index is recreated properly.
    ds = vector_store.dataset
    es = ds.embedding.get_vdb_indexes()
    assert len(es) == 1
    assert es[0]["id"] == "hnsw_1"
    assert es[0]["distance"] == "l2_norm"
    assert es[0]["type"] == "hnsw"

    ds = deeplake.load(path=local_path, read_only=True)

    ds2 = deeplake.like("mem://dummy", ds, overwrite=True)

    for tensor in ds2.tensors:
        ds2[tensor].extend(ds[tensor].data()["value"])

    vector_store.delete_by_path(local_path, token=hub_cloud_dev_token)


def assert_vectorstore_structure(vector_store, number_of_data):
    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset_handler.dataset.tensors) == {
        "embedding",
        "id",
        "metadata",
        "text",
    }
    assert set(vector_store.tensors()) == {
        "embedding",
        "id",
        "metadata",
        "text",
    }
    assert vector_store.dataset_handler.dataset.embedding.htype == "embedding"
    assert vector_store.dataset_handler.dataset.id.htype == "text"
    assert vector_store.dataset_handler.dataset.metadata.htype == "json"
    assert vector_store.dataset_handler.dataset.text.htype == "text"
    assert vector_store.dataset_handler.dataset.embedding.dtype == "float32"
    assert vector_store.dataset_handler.dataset.id.dtype == "str"
    assert vector_store.dataset_handler.dataset.metadata.dtype == "str"
    assert vector_store.dataset_handler.dataset.text.dtype == "str"


@pytest.mark.slow
def test_ingestion(local_path):
    # create data
    number_of_data = 1000
    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=True,
    )

    with pytest.raises(Exception):
        # add data to the dataset:
        vector_store.add(
            embedding=embeddings,
            text=texts[: number_of_data - 2],
            id=ids,
            metadata=metadatas,
        )

    with pytest.raises(ValueError):
        # add data to the dataset:
        vector_store.add(
            embedding=embeddings,
            text=texts[: number_of_data - 2],
            id=ids,
            metadata=metadatas,
            something=texts[: number_of_data - 2],
        )

    vector_store.add(embedding=embeddings, text=texts, id=ids, metadata=metadatas)
    assert_vectorstore_structure(vector_store, number_of_data)

    vector_store.add(
        embedding_function=embedding_fn3,
        embedding_data=texts,
        text=texts,
        id=ids,
        metadata=metadatas,
    )
    assert_vectorstore_structure(vector_store, 2 * number_of_data)

    vector_store.add(
        embedding_function=embedding_fn3,
        embedding_data=25 * texts,
        text=25 * texts,
        id=25 * ids,
        metadata=25 * metadatas,
    )
    assert_vectorstore_structure(vector_store, 27000)


def test_ingestion_images(local_path):
    tensor_params = [
        {"name": "image", "htype": "image", "sample_compression": "jpg"},
        {"name": "embedding", "htype": "embedding"},
    ]

    append_images = images
    append_images[0] = np.random.randint(0, 255, (100, 100, 3)).astype(
        np.uint8
    )  # Mix paths and images

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path=local_path,
        tensor_params=tensor_params,
        overwrite=True,
        verbose=True,
    )

    ids = vector_store.add(image=images, embedding=embeddings, return_ids=True)

    assert "image" in vector_store.dataset_handler.dataset.tensors
    assert "embedding" in vector_store.dataset_handler.dataset.tensors
    assert len(vector_store.dataset_handler.dataset.image[0].numpy().shape) == 3
    assert len(vector_store.dataset_handler.dataset.image[1].numpy().shape) == 3
    assert len(ids) == 10


def test_parse_add_arguments(local_path):
    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn,
    )
    embedding_fn_dp = DeepLakeEmbedder(embedding_function=embedding_fn)
    embedding_fn2_dp = DeepLakeEmbedder(embedding_function=embedding_fn2)

    with pytest.raises(ValueError):
        # Throw error because embedding_function requires embed_data_from
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            initial_embedding_function=embedding_fn,
            embedding_function=embedding_fn,
            embeding_tensor="embedding",
            text=texts,
            id=ids,
            metadata=metadatas,
        )

    with pytest.raises(ValueError):
        # Throw error because embedding function is not specified anywhere
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            embedding_data=texts,
            embeding_tensor="embedding",
            text=texts,
            id=ids,
            metadata=metadatas,
        )

    with pytest.raises(ValueError):
        # Throw error because data is not specified for all tensors
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            text=texts,
            id=ids,
            metadata=metadatas,
        )

    with pytest.raises(ValueError):
        # initial embedding function specified and embeding_tensor is specified
        (
            embedding_function,
            embeding_tensor,
            embed_data_from,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            initial_embedding_function=embedding_fn,
            embedding_tensor="embedding",
            text=texts,
            id=ids,
            metadata=metadatas,
        )

    # initial embedding function is specified and embeding_tensor, embed_data_from are not specified
    (
        embedding_function,
        embeding_tensor,
        embed_data_from,
        tensors,
    ) = utils.parse_add_arguments(
        dataset=deeplake_vector_store.dataset_handler.dataset,
        initial_embedding_function=embedding_fn_dp,
        text=texts,
        id=ids,
        embedding=embeddings,
        metadata=metadatas,
    )
    assert embedding_function is None
    assert embeding_tensor == None
    assert embed_data_from is None
    assert tensors == {
        "id": ids,
        "text": texts,
        "metadata": metadatas,
        "embedding": embeddings,
    }

    with pytest.raises(ValueError):
        # initial embedding function specified and embeding_tensor is not specified
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            initial_embedding_function=embedding_fn_dp,
            embedding_data=texts,
            text=texts,
            id=ids,
            embedding=embeddings,
            metadata=metadatas,
        )  # 2

    with pytest.raises(ValueError):
        # Throw error because embedding_function and embedding are specified
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            initial_embedding_function=embedding_fn_dp,
            embedding_function=embedding_fn_dp,
            embedding_data=texts,
            embedding_tensor="embedding",
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding=embeddings,
        )

    with pytest.raises(ValueError):
        # initial_embedding_function is specified and embeding_tensor, embed_data_from and embedding is specified.
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            initial_embedding_function=embedding_fn_dp,
            embedding_tensor="embedding",
            embedding_data=texts,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding=embeddings,
        )

    with pytest.raises(ValueError):
        # initial_embedding_function is not specified and embeding_tensor, embed_data_from and embedding is specified.
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            embeding_tensor="embedding",
            embedding_data=texts,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding=embeddings,
        )

    with pytest.raises(ValueError):
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            embedding_function=embedding_fn_dp,
            initial_embedding_function=embedding_fn_dp,
            embedding_data=texts,
            embedding_tensor="embedding",
            text=texts,
            id=ids,
            embedding=embeddings,
            metadata=metadatas,
        )

    (
        embedding_function,
        embedding_data,
        embedding_tensors,
        tensors,
    ) = utils.parse_add_arguments(
        dataset=deeplake_vector_store.dataset_handler.dataset,
        embedding_function=embedding_fn2_dp,
        embedding_data=texts,
        embedding_tensor="embedding",
        text=texts,
        id=ids,
        metadata=metadatas,
    )
    assert embedding_tensors == ["embedding"]
    assert tensors == {
        "id": ids,
        "text": texts,
        "metadata": metadatas,
    }

    (
        embedding_function,
        embedding_data,
        embedding_tensors,
        tensors,
    ) = utils.parse_add_arguments(
        dataset=deeplake_vector_store.dataset_handler.dataset,
        embedding_function=embedding_fn2_dp,
        embedding_data="text",
        embedding_tensor="embedding",
        text=texts,
        metadata=metadatas,
    )
    assert embedding_tensors == ["embedding"]
    assert len(tensors) == 2

    # Creating a vector store with two embedding tensors
    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn,
        tensor_params=[
            {
                "name": "embedding_1",
                "htype": "embedding",
            },
            {
                "name": "embedding_2",
                "htype": "embedding",
            },
            {
                "name": "texts",
                "htype": "text",
            },
        ],
    )

    # There are two embedding but an embedding_tensor is not specified, so it's not clear where to add the embedding data
    with pytest.raises(ValueError):
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            embedding_function=embedding_fn2_dp,
            embedding_data="text",
            text=texts,
            metadata=metadatas,
        )

    # Creating a vector store without embedding htype or tensor name
    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn,
        tensor_params=[
            {
                "name": "text",
                "htype": "text",
            },
        ],
    )

    # There is no embedding tensor, so it's not clear where to add the embedding data
    with pytest.raises(ValueError):
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            embedding_function=embedding_fn2_dp,
            embedding_data=texts,
            text=texts,
        )

    # Creating a vector store with embedding tensor with a custom name
    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn2,
        tensor_params=[
            {
                "name": "embedding_1",
                "htype": "embedding",
            },
            {
                "name": "text",
                "htype": "text",
            },
        ],
    )

    (
        embedding_function,
        embedding_data,
        embedding_tensors,
        tensors,
    ) = utils.parse_add_arguments(
        dataset=deeplake_vector_store.dataset_handler.dataset,
        embedding_function=embedding_fn2_dp,
        embedding_data=texts,
        text=texts,
    )

    assert embedding_tensors == ["embedding_1"]
    assert len(tensors) == 1

    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn,
    )

    (
        embedding_function,
        embedding_data,
        embedding_tensor,
        tensors,
    ) = utils.parse_add_arguments(
        dataset=deeplake_vector_store.dataset_handler.dataset,
        initial_embedding_function=embedding_fn_dp,
        text=texts,
        id=ids,
        metadata=metadatas,
        embedding_data=texts,
        embedding_tensor="embedding",
    )
    assert embedding_tensor == ["embedding"]
    assert embedding_data == [texts]
    assert tensors == {
        "id": ids,
        "text": texts,
        "metadata": metadatas,
    }

    with pytest.raises(ValueError):
        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding_tensor="embedding",
        )

    with pytest.raises(ValueError):
        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding_data=texts,
        )

    with pytest.raises(ValueError):
        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset_handler.dataset,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding=embeddings,
            new_tensor=texts,
        )

    with pytest.raises(ValueError):
        dataset = deeplake.empty(local_path, overwrite=True)
        dataset.create_tensor("embedding_1", htype="embedding")
        dataset.create_tensor("embedding_2", htype="embedding")
        dataset.create_tensor("id", htype="text")
        dataset.create_tensor("text", htype="text")
        dataset.create_tensor("metadata", htype="json")

        dataset.extend(
            {
                "embedding_1": embeddings,
                "embedding_2": embeddings,
                "id": ids,
                "text": texts,
                "metadata": metadatas,
            }
        )

        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=dataset,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding_data=texts,
            embedding_function=DeepLakeEmbedder(embedding_function=embedding_fn3),
            embedding_2=embeddings,
        )


def test_parse_tensors_kwargs():
    tensors = {
        "embedding_1": (embedding_fn, texts),
        "embedding_2": (embedding_fn2, texts),
        "custom_text": texts,
    }
    func, data, emb_tensor, new_tensors = utils.parse_tensors_kwargs(
        tensors, None, None, None
    )

    assert isinstance(func[0], DeepLakeEmbedder)
    assert isinstance(func[1], DeepLakeEmbedder)
    assert data == [texts, texts]
    assert emb_tensor == ["embedding_1", "embedding_2"]
    assert new_tensors == {"custom_text": texts}

    with pytest.raises(ValueError):
        utils.parse_tensors_kwargs(tensors, embedding_fn, None, None)

    with pytest.raises(ValueError):
        utils.parse_tensors_kwargs(tensors, None, texts, None)

    with pytest.raises(ValueError):
        utils.parse_tensors_kwargs(tensors, None, None, "embedding_1")


@pytest.mark.slow
@requires_libdeeplake
def test_multiple_embeddings(local_path):
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {
                "name": "text",
                "htype": "text",
            },
            {
                "name": "embedding_1",
                "htype": "embedding",
            },
            {
                "name": "embedding_2",
                "htype": "embedding",
            },
        ],
    )

    with pytest.raises(AssertionError):
        vector_store.add(
            text=texts,
            embedding_function=[embedding_fn, embedding_fn],
            embedding_data=[texts],
            embedding_tensor=["embedding_1", "embedding_2"],
        )

    with pytest.raises(AssertionError):
        vector_store.add(
            text=texts,
            embedding_function=[embedding_fn, embedding_fn],
            embedding_data=[texts, texts],
            embedding_tensor=["embedding_1"],
        )

    with pytest.raises(AssertionError):
        vector_store.add(
            text=texts,
            embedding_function=[embedding_fn],
            embedding_data=[texts, texts],
            embedding_tensor=["embedding_1", "embedding_2"],
        )

    vector_store.add(
        text=texts,
        embedding_function=[embedding_fn, embedding_fn],
        embedding_data=[texts, texts],
        embedding_tensor=["embedding_1", "embedding_2"],
    )

    vector_store.add(
        text=texts, embedding_1=(embedding_fn, texts), embedding_2=(embedding_fn, texts)
    )

    vector_store.add(
        text=texts,
        embedding_function=embedding_fn,
        embedding_data=[texts, texts],
        embedding_tensor=["embedding_1", "embedding_2"],
    )

    # test with initial embedding function
    vector_store.dataset_handler.embedding_function = DeepLakeEmbedder(
        embedding_function=embedding_fn
    )
    vector_store.add(
        text=texts,
        embedding_data=[texts, texts],
        embedding_tensor=["embedding_1", "embedding_2"],
    )

    number_of_data = 1000
    _texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=number_of_data, embedding_dim=EMBEDDING_DIM
    )
    vector_store.add(
        text=25 * _texts,
        embedding_function=[embedding_fn3, embedding_fn3],
        embedding_data=[25 * _texts, 25 * _texts],
        embedding_tensor=["embedding_1", "embedding_2"],
    )
    vector_store.add(
        text=25 * _texts,
        embedding_1=(embedding_fn3, 25 * _texts),
        embedding_2=(embedding_fn3, 25 * _texts),
    )

    assert len(vector_store.dataset_handler.dataset) == 50040
    assert len(vector_store.dataset_handler.dataset.embedding_1) == 50040
    assert len(vector_store.dataset_handler.dataset.embedding_2) == 50040
    assert len(vector_store.dataset_handler.dataset.id) == 50040
    assert len(vector_store.dataset_handler.dataset.text) == 50040


def test_extend_none(local_path):
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {"name": "text", "htype": "text"},
            {"name": "embedding", "htype": "embedding"},
            {
                "name": "id",
                "htype": "text",
            },
            {"name": "metadata", "htype": "json"},
        ],
    )

    vector_store.add(text=texts, embedding=None, id=ids, metadata=None)
    assert len(vector_store.dataset_handler.dataset) == 10
    assert len(vector_store.dataset_handler.dataset.text) == 10
    assert len(vector_store.dataset_handler.dataset.embedding) == 10
    assert len(vector_store.dataset_handler.dataset.id) == 10
    assert len(vector_store.dataset_handler.dataset.metadata) == 10


def test_query_dim(local_path):
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {"name": "text", "htype": "text"},
            {"name": "embedding", "htype": "embedding"},
        ],
    )

    vector_store.add(text=texts, embedding=embeddings)
    with pytest.raises(NotImplementedError):
        vector_store.search([texts[0], texts[0]], embedding_fn3, k=1)

    vector_store.search([texts[0]], embedding_fn4, k=1)


def test_embeddings_only(local_path):
    vector_store = VectorStore(
        path=local_path,
        overwrite=True,
        tensor_params=[
            {"name": "embedding_1", "htype": "embedding"},
            {"name": "embedding_2", "htype": "embedding"},
        ],
    )

    vector_store.add(
        embedding_1=(embedding_fn, texts), embedding_2=(embedding_fn3, texts)
    )

    assert len(vector_store.dataset_handler.dataset) == 10
    assert len(vector_store.dataset_handler.dataset.embedding_1) == 10
    assert len(vector_store.dataset_handler.dataset.embedding_2) == 10


def test_uuid_fix(local_path):
    vector_store = VectorStore(local_path, overwrite=True)

    ids = [uuid.uuid4() for _ in range(NUMBER_OF_DATA)]

    vector_store.add(text=texts, id=ids, embedding=embeddings, metadata=metadatas)

    assert vector_store.dataset_handler.dataset.id.data()["value"] == list(
        map(str, ids)
    )


@pytest.mark.slow
def test_read_only():
    db = VectorStore("hub://davitbun/twitter-algorithm")
    assert db.dataset_handler.dataset.read_only == True


def test_delete_by_path_wrong_path():
    with pytest.raises(DatasetHandlerError):
        VectorStore.delete_by_path("some_path")


@pytest.mark.slow
@requires_libdeeplake
def test_exec_option_with_auth(local_path, hub_cloud_path, hub_cloud_dev_token):
    db = VectorStore(path=local_path)
    assert db.dataset_handler.exec_option == "python"

    db = VectorStore(
        path=local_path,
        token=hub_cloud_dev_token,
    )
    assert db.dataset_handler.exec_option == "compute_engine"

    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
    )
    assert db.dataset_handler.exec_option == "compute_engine"

    db = VectorStore(
        path=hub_cloud_path + "_tensor_db",
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )
    assert db.dataset_handler.exec_option == "tensor_db"


@requires_libdeeplake
@pytest.mark.parametrize(
    "path, creds",
    [
        ("s3_path", "s3_creds"),
        ("gcs_path", "gcs_creds"),
        ("azure_path", "azure_creds"),
    ],
    indirect=True,
)
def test_exec_option_with_connected_datasets(
    hub_cloud_path,
    path,
    creds,
):
    db = VectorStore(path, overwrite=True, creds=creds)

    db.dataset_handler.dataset.connect(
        creds_key=creds,
        dest_path=hub_cloud_path,
    )
    db.dataset_handler.dataset.add_creds_key(creds, managed=True)
    assert db.exec_option == "compute_engine"


def test_dataset_init_param(local_ds):
    local_ds.create_tensor("text", htype="text")
    local_ds.create_tensor("embedding", htype="embedding")
    local_ds.create_tensor("id", htype="text")
    local_ds.create_tensor("metadata", htype="json")

    db = VectorStore(
        dataset=local_ds,
    )

    db.add(text=texts, embedding=embeddings, id=ids, metadata=metadatas)
    assert len(db) == 10


@requires_libdeeplake
def test_vs_commit(local_path):
    # TODO: add index params, when index will support commit
    db = create_and_populate_vs(
        local_path, number_of_data=NUMBER_OF_DATA, index_params=None
    )
    db.checkout("branch_1", create=True)
    db.commit("commit_1")
    db.add(text=texts, embedding=embeddings, id=ids, metadata=metadatas)
    assert len(db) == 2 * NUMBER_OF_DATA

    db.checkout("main")
    assert len(db) == NUMBER_OF_DATA


def test_vs_init_when_both_dataset_and_path_is_specified(local_path):
    with pytest.raises(ValueError):
        VectorStore(
            path=local_path,
            dataset=deeplake.empty(local_path, overwrite=True),
        )


def test_vs_init_when_both_dataset_and_path_are_not_specified():
    with pytest.raises(ValueError):
        VectorStore()


def test_vs_init_with_emptyt_token(local_path):
    with patch("deeplake.client.config.DEEPLAKE_AUTH_TOKEN", ""):
        db = VectorStore(
            path=local_path,
        )

    assert db.dataset_handler.username == "public"


@pytest.fixture
def mock_search_managed(mocker):
    # Replace SearchManaged with a mock
    mock_class = mocker.patch(
        "deeplake.core.vectorstore.vector_search.indra.search_algorithm.SearchManaged"
    )
    return mock_class


@pytest.fixture
def mock_search_indra(mocker):
    # Replace SearchIndra with a mock
    mock_class = mocker.patch(
        "deeplake.core.vectorstore.vector_search.indra.search_algorithm.SearchIndra"
    )
    return mock_class


def test_vs_init_when_both_dataset_and_path_is_specified_should_throw_exception(
    local_path,
):
    with pytest.raises(ValueError):
        VectorStore(
            path=local_path,
            dataset=deeplake.empty(local_path, overwrite=True),
        )


def test_specifying_row_ids_and_filter_should_throw_excrption(local_path):
    db = VectorStore(
        path=local_path,
    )
    db.add(text=texts, embedding=embeddings, id=ids, metadata=metadatas)


def test_vs_init_when_both_dataset_and_path_are_not_specified_should_throw_exception():
    with pytest.raises(ValueError):
        VectorStore()


def test_vs_init_with_emptyt_token_should_not_throw_exception(local_path):
    with patch("deeplake.client.config.DEEPLAKE_AUTH_TOKEN", ""):
        db = VectorStore(
            path=local_path,
        )

    assert db.dataset_handler.username == "public"


@pytest.mark.slow
def test_db_search_with_managed_db_should_instantiate_SearchManaged_class(
    mock_search_managed, hub_cloud_path, hub_cloud_dev_token
):
    # using interaction test to ensure that the search managed class is executed
    db = create_and_populate_vs(
        hub_cloud_path,
        runtime={"tensor_db": True},
        token=hub_cloud_dev_token,
    )

    # Perform the search
    db.search(embedding=query_embedding)

    # Assert that SearchManaged was instantiated
    mock_search_managed.assert_called()


@pytest.mark.slow
@requires_libdeeplake
def test_db_search_should_instantiate_SearchIndra_class(
    mock_search_indra, hub_cloud_path, hub_cloud_dev_token
):
    # using interaction test to ensure that the search indra class is executed
    db = create_and_populate_vs(
        hub_cloud_path,
        token=hub_cloud_dev_token,
    )

    # Perform the search
    db.search(embedding=query_embedding)

    # Assert that SearchIndra was instantiated
    mock_search_indra.assert_called()


def returning_tql_for_exec_option_python_should_throw_exception(local_path):
    db = VectorStore(
        path=local_path,
    )
    db.add(text=texts, embedding=embeddings, id=ids, metadata=metadatas)

    with pytest.raises(NotImplementedError):
        db.search(embedding=query_embedding, return_tql=True)


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_returning_tql_for_exec_option_compute_engine_should_return_correct_tql(
    local_path,
    hub_cloud_dev_token,
):
    db = VectorStore(
        path=local_path,
        token=hub_cloud_dev_token,
    )

    texts, embeddings, ids, metadatas, _ = utils.create_data(
        number_of_data=10, embedding_dim=3
    )

    db.add(text=texts, embedding=embeddings, id=ids, metadata=metadatas)

    query_embedding = np.zeros(3, dtype=np.float32)
    output = db.search(embedding=query_embedding, return_tql=True)

    assert output["tql"] == (
        "select text, metadata, id, score from "
        "(select *, COSINE_SIMILARITY(embedding, ARRAY[0.0, 0.0, 0.0]) as score "
        "order by COSINE_SIMILARITY(embedding, ARRAY[0.0, 0.0, 0.0]) DESC limit 4)"
    )


def test_delete_all_bug(local_path):
    vs = VectorStore("local_path", overwrite=True)
    ids = vs.add(
        text=["a", "b"],
        metadata=[{}, {}],
        embedding=[[1, 2, 3], [2, 3, 4]],
        return_ids=True,
    )

    ds = vs.dataset
    pickled = pickle.dumps(ds)
    unpickled = pickle.loads(pickled)

    vs = VectorStore(dataset=unpickled)

    assert len(vs) == 2
    vs.delete(delete_all=True)

    assert len(vs) == 0
