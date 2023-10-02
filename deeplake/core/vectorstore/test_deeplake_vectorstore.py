import uuid
import os
import sys
from math import isclose
from functools import partial

import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.deeplake_vectorstore import (
    DeepLakeVectorStore,
    VectorStore,
)
from deeplake.core.vectorstore.deepmemory_vectorstore import DeepMemoryVectorStore
from deeplake.core.vectorstore.vectorstore_factory import vectorstore_factory
from deeplake.core.vectorstore import utils
from deeplake.tests.common import requires_libdeeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
)
from deeplake.constants import MB
from deeplake.util.exceptions import (
    IncorrectEmbeddingShapeError,
    TensorDoesNotExistError,
    DatasetHandlerError,
)
from deeplake.core.vectorstore.vector_search.indra.index import METRIC_TO_INDEX_METRIC
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.cli.auth import login, logout
from click.testing import CliRunner


EMBEDDING_DIM = 100
NUMBER_OF_DATA = 10
# create data
texts, embeddings, ids, metadatas, images = utils.create_data(
    number_of_data=NUMBER_OF_DATA, embedding_dim=EMBEDDING_DIM
)

query_embedding = np.random.uniform(low=-10, high=10, size=(EMBEDDING_DIM)).astype(
    np.float32
)


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


def get_embedding_function(embedding_value):
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

    assert len(vectorstore) == 20


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
def test_search_basic(local_path, hub_cloud_dev_token):
    """Test basic search features"""
    # Initialize vector store object and add data
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    assert vector_store.exec_option == "compute_engine"

    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    with pytest.raises(ValueError):
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
        sum([tensor in data_p.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_p.keys()) == 3  # One for each return_tensors + score

    # Load a vector store object from the cloud for indra testing
    vector_store_cloud = DeepLakeVectorStore(
        path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )
    assert vector_store_cloud.exec_option == "compute_engine"

    # Use indra implementation to search the data
    data_ce = vector_store_cloud.search(
        embedding=query_embedding,
        k=2,
        return_tensors=["id", "text"],
    )
    assert len(data_ce["text"]) == 2
    assert (
        sum([tensor in data_ce.keys() for tensor in vector_store_cloud.dataset.tensors])
        == 2
    )  # One for each return_tensors
    assert len(data_ce.keys()) == 3  # One for each return_tensors + score

    with pytest.raises(ValueError):
        vector_store_cloud.search(
            query=f"SELECT * WHERE id=='{vector_store_cloud.dataset.id[0].numpy()[0]}'",
            embedding=query_embedding,
            k=2,
            return_tensors=["id", "text"],
        )

    # Run a full custom query
    test_text = vector_store_cloud.dataset.text[0].data()["value"]
    data_q = vector_store_cloud.search(
        query=f"select * where text == '{test_text}'",
    )

    assert len(data_q["text"]) == 1
    assert data_q["text"][0] == test_text
    assert sum(
        [tensor in data_q.keys() for tensor in vector_store_cloud.dataset.tensors]
    ) == len(
        vector_store_cloud.dataset.tensors
    )  # One for each tensor - embedding + score

    # Run a filter query using a json
    data_e_j = vector_store.search(
        k=2,
        return_tensors=["id", "text"],
        filter={"metadata": metadatas[2], "text": texts[2]},
    )
    assert len(data_e_j["text"]) == 1
    assert (
        sum([tensor in data_e_j.keys() for tensor in vector_store.dataset.tensors]) == 2
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
        sum([tensor in data_e_f.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_e_f.keys()) == 2

    # Run a filter query using a json with indra
    data_ce_f = vector_store_cloud.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        k=2,
        return_tensors=["id", "text"],
        filter={
            "metadata": vector_store_cloud.dataset.metadata[0].data()["value"],
            "text": vector_store_cloud.dataset.text[0].data()["value"],
        },
    )
    assert len(data_ce_f["text"]) == 1
    assert (
        sum(
            [
                tensor in data_ce_f.keys()
                for tensor in vector_store_cloud.dataset.tensors
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

    assert vector_store_none_exec.exec_option == "compute_engine"

    # Check that filter_fn with cloud dataset (and therefore "compute_engine" exec option) switches to "python" automatically.
    with pytest.warns(None):
        _ = vector_store_cloud.search(
            filter=filter_fn,
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
    assert vector_store.exec_option == "python"
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    data = vector_store.search(
        embedding_function=embedding_fn3,
        embedding_data=["dummy"],
        return_view=True,
        k=2,
    )
    assert len(data) == 2
    assert isinstance(data.text[0].data()["value"], str)
    assert data.embedding[0].numpy().size > 0

    data = vector_store.search(
        embedding_function=embedding_fn3,
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
        path="mem://xyz", embedding_function=embedding_fn3
    )
    assert vector_store.exec_option == "python"
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)
    result = vector_store.search(embedding_data=["dummy"])
    assert len(result) == 4


@requires_libdeeplake
def test_index_basic(local_path, hub_cloud_dev_token):
    # Start by testing behavior without an index
    vector_store = VectorStore(
        path=local_path,
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    assert vector_store.distance_metric_index is None

    # Then test behavior when index is added
    vector_store = VectorStore(
        path=local_path, token=hub_cloud_dev_token, index_params={"threshold": 1}
    )

    assert (
        vector_store.distance_metric_index
        == METRIC_TO_INDEX_METRIC[DEFAULT_VECTORSTORE_DISTANCE_METRIC]
    )

    # Then test behavior when index is added previously and the dataset is reloaded
    vector_store = VectorStore(path=local_path, token=hub_cloud_dev_token)

    assert (
        vector_store.distance_metric_index
        == METRIC_TO_INDEX_METRIC[DEFAULT_VECTORSTORE_DISTANCE_METRIC]
    )

    # Check that distance metric throws a warning when there is an index
    with pytest.warns(None):
        vector_store.search(embedding=query_embedding, distance_metric="l1")


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

    test_id = vector_store.dataset.id[0].data()["value"]

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

    assert "vectordb/" in vector_store.dataset.base_storage.path

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
    assert len(vector_store.dataset) == NUMBER_OF_DATA - 3

    vector_store.delete(filter={"metadata": {"abc": 1}})
    assert len(vector_store.dataset) == NUMBER_OF_DATA - 4

    vector_store.delete(ids=["7"])
    assert len(vector_store.dataset) == NUMBER_OF_DATA - 5

    with pytest.raises(ValueError):
        vector_store.delete()

    tensors_before_delete = vector_store.dataset.tensors
    vector_store.delete(delete_all=True)
    assert len(vector_store.dataset) == 0
    assert vector_store.dataset.tensors.keys() == tensors_before_delete.keys()

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
    assert len(vector_store_b.dataset) == NUMBER_OF_DATA - 1

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
            vector_store.dataset[embedding_tensor][row_ids].numpy(),
            new_embeddings,
        )

    if callable(embedding_function) and isinstance(embedding_tensor, list):
        for i in range(len(embedding_tensor)):
            np.testing.assert_array_equal(
                vector_store.dataset[embedding_tensor[i]][row_ids].numpy(),
                new_embeddings[i],
            )

    if isinstance(embedding_function, list) and isinstance(embedding_tensor, list):
        for i in range(len(embedding_tensor)):
            np.testing.assert_array_equal(
                vector_store.dataset[embedding_tensor[i]][row_ids].numpy(),
                new_embeddings[i],
            )


@requires_libdeeplake
@pytest.mark.parametrize(
    "ds, vector_store_hash_ids, vector_store_row_ids, vector_store_filters, vector_store_query",
    [
        ("local_auth_ds", "vector_store_hash_ids", None, None, None),
        ("local_auth_ds", None, "vector_store_row_ids", None, None),
        ("local_auth_ds", None, None, "vector_store_filter_udf", None),
        ("local_auth_ds", None, None, "vector_store_filters", None),
        ("hub_cloud_ds", None, None, None, "vector_store_query"),
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
    vector_store_query,
    init_embedding_function,
):
    if vector_store_filters == "filter_udf":
        vector_store_filters = filter_udf

    embedding_tensor = "embedding"
    embedding_source_tensor = "text"
    # dataset has a single embedding_tensor:

    path = ds.path
    vector_store = DeepLakeVectorStore(
        path=path,
        overwrite=True,
        verbose=False,
        exec_option="compute_engine",
        embedding_function=init_embedding_function,
        index_params={"threshold": 10},
        token=ds.token,
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
        "compute_engine",
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
        "compute_engine",
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
            "compute_engine",
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
        "compute_engine",
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
            "compute_engine",
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
        "compute_engine",
        num_changed_samples=5,
    )
    vector_store.delete_by_path(path + "_multi", token=ds.token)


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

    vector_store.delete_by_path(local_path, token=ds.token)


def assert_vectorstore_structure(vector_store, number_of_data):
    assert len(vector_store) == number_of_data
    assert set(vector_store.dataset.tensors) == {
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
    assert vector_store.dataset.embedding.htype == "embedding"
    assert vector_store.dataset.id.htype == "text"
    assert vector_store.dataset.metadata.htype == "json"
    assert vector_store.dataset.text.htype == "text"
    assert vector_store.dataset.embedding.dtype == "float32"
    assert vector_store.dataset.id.dtype == "str"
    assert vector_store.dataset.metadata.dtype == "str"
    assert vector_store.dataset.text.dtype == "str"


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

    assert "image" in vector_store.dataset.tensors
    assert "embedding" in vector_store.dataset.tensors
    assert len(vector_store.dataset.image[0].numpy().shape) == 3
    assert len(vector_store.dataset.image[1].numpy().shape) == 3
    assert len(ids) == 10


def test_parse_add_arguments(local_path):
    deeplake_vector_store = DeepLakeVectorStore(
        path="mem://dummy",
        overwrite=True,
        embedding_function=embedding_fn,
    )

    with pytest.raises(ValueError):
        # Throw error because embedding_function requires embed_data_from
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset,
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
            dataset=deeplake_vector_store.dataset,
            embedding_data=texts,
            embeding_tensor="embedding",
            text=texts,
            id=ids,
            metadata=metadatas,
        )

    with pytest.raises(ValueError):
        # Throw error because data is not specified for all tensors
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset,
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
            dataset=deeplake_vector_store.dataset,
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
        dataset=deeplake_vector_store.dataset,
        initial_embedding_function=embedding_fn,
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
            dataset=deeplake_vector_store.dataset,
            initial_embedding_function=embedding_fn,
            embedding_data=texts,
            text=texts,
            id=ids,
            embedding=embeddings,
            metadata=metadatas,
        )  # 2

    with pytest.raises(ValueError):
        # Throw error because embedding_function and embedding are specified
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset,
            initial_embedding_function=embedding_fn,
            embedding_function=embedding_fn,
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
            dataset=deeplake_vector_store.dataset,
            initial_embedding_function=embedding_fn,
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
            dataset=deeplake_vector_store.dataset,
            embeding_tensor="embedding",
            embedding_data=texts,
            text=texts,
            id=ids,
            metadata=metadatas,
            embedding=embeddings,
        )

    with pytest.raises(ValueError):
        utils.parse_add_arguments(
            dataset=deeplake_vector_store.dataset,
            embedding_function=embedding_fn,
            initial_embedding_function=embedding_fn,
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
        dataset=deeplake_vector_store.dataset,
        embedding_function=embedding_fn2,
        embedding_data=texts,
        embedding_tensor="embedding",
        text=texts,
        id=ids,
        metadata=metadatas,
    )
    assert embedding_function[0] is embedding_fn2
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
        dataset=deeplake_vector_store.dataset,
        embedding_function=embedding_fn2,
        embedding_data="text",
        embedding_tensor="embedding",
        text=texts,
        metadata=metadatas,
    )
    assert embedding_function[0] is embedding_fn2
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
            dataset=deeplake_vector_store.dataset,
            embedding_function=embedding_fn2,
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
            dataset=deeplake_vector_store.dataset,
            embedding_function=embedding_fn2,
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
        dataset=deeplake_vector_store.dataset,
        embedding_function=embedding_fn2,
        embedding_data=texts,
        text=texts,
    )

    assert embedding_function[0] is embedding_fn2
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
        dataset=deeplake_vector_store.dataset,
        initial_embedding_function=embedding_fn,
        text=texts,
        id=ids,
        metadata=metadatas,
        embedding_data=texts,
        embedding_tensor="embedding",
    )
    assert embedding_function[0] is embedding_fn
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
            dataset=deeplake_vector_store.dataset,
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
            dataset=deeplake_vector_store.dataset,
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
            dataset=deeplake_vector_store.dataset,
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
            embedding_function=embedding_fn3,
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

    assert func == [embedding_fn, embedding_fn2]
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
    vector_store.embedding_function = embedding_fn
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

    assert len(vector_store.dataset) == 50040
    assert len(vector_store.dataset.embedding_1) == 50040
    assert len(vector_store.dataset.embedding_2) == 50040
    assert len(vector_store.dataset.id) == 50040
    assert len(vector_store.dataset.text) == 50040


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
    assert len(vector_store.dataset) == 10
    assert len(vector_store.dataset.text) == 10
    assert len(vector_store.dataset.embedding) == 10
    assert len(vector_store.dataset.id) == 10
    assert len(vector_store.dataset.metadata) == 10


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

    assert len(vector_store.dataset) == 10
    assert len(vector_store.dataset.embedding_1) == 10
    assert len(vector_store.dataset.embedding_2) == 10


def test_uuid_fix(local_path):
    vector_store = VectorStore(local_path, overwrite=True)

    ids = [uuid.uuid4() for _ in range(NUMBER_OF_DATA)]

    vector_store.add(text=texts, id=ids, embedding=embeddings, metadata=metadatas)

    assert vector_store.dataset.id.data()["value"] == list(map(str, ids))


def test_read_only():
    db = VectorStore("hub://davitbun/twitter-algorithm")
    assert db.dataset.read_only == True


def test_delete_by_path_wrong_path():
    with pytest.raises(DatasetHandlerError):
        VectorStore.delete_by_path("some_path")


@requires_libdeeplake
def test_exec_option_with_auth(local_path, hub_cloud_path, hub_cloud_dev_token):
    db = VectorStore(path=local_path)
    assert db.exec_option == "python"

    db = VectorStore(
        path=local_path,
        token=hub_cloud_dev_token,
    )
    assert db.exec_option == "compute_engine"

    db = VectorStore(
        path=hub_cloud_path,
        token=hub_cloud_dev_token,
    )
    assert db.exec_option == "compute_engine"

    db = VectorStore(
        path=hub_cloud_path + "_tensor_db",
        token=hub_cloud_dev_token,
        runtime={"tensor_db": True},
    )
    assert db.exec_option == "tensor_db"


@requires_libdeeplake
def test_exec_option_cli(
    local_path,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_credentials,
):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials
    # Testing exec_option with cli login and logout commands are executed
    runner.invoke(login, f"-u {username} -p {password}")

    # local dataset and logged in with cli
    db = VectorStore(
        path=local_path,
    )
    assert db.exec_option == "compute_engine"

    # hub cloud dataset and logged in with cli
    db = VectorStore(
        path=hub_cloud_path,
    )
    assert db.exec_option == "compute_engine"

    # hub cloud dataset and logged in with cli
    db = VectorStore(
        path="mem://abc",
    )
    assert db.exec_option == "python"

    # logging out with cli
    runner.invoke(logout)

    # local dataset and logged out with cli
    db = VectorStore(
        path=local_path,
    )
    assert db.exec_option == "python"

    # Check whether after logging out exec_option changes to python
    # logging in with cli token
    runner.invoke(login, f"-t {hub_cloud_dev_token}")
    db = VectorStore(
        path=local_path,
    )
    assert db.exec_option == "compute_engine"
    # logging out with cli
    runner.invoke(logout)
    assert db.exec_option == "python"

    # Check whether after logging out when token specified exec_option doesn't change
    # logging in with cli token
    runner.invoke(login, f"-t {hub_cloud_dev_token}")
    db = VectorStore(
        path=local_path,
        token=hub_cloud_dev_token,
    )
    assert db.exec_option == "compute_engine"
    # logging out with cli
    runner.invoke(logout)
    assert db.exec_option == "compute_engine"


@requires_libdeeplake
@pytest.mark.parametrize(
    "path",
    [
        "s3_path",
        "gcs_path",
        "azure_path",
    ],
    indirect=True,
)
def test_exec_option_with_connected_datasets(
    hub_cloud_dev_token,
    hub_cloud_path,
    hub_cloud_dev_managed_creds_key,
    path,
):
    runner = CliRunner()

    db = VectorStore(path, overwrite=True)
    assert db.exec_option == "python"

    runner.invoke(login, f"-t {hub_cloud_dev_token}")
    assert db.exec_option == "python"

    db.dataset.connect(
        creds_key=hub_cloud_dev_managed_creds_key,
        dest_path=hub_cloud_path,
        token=hub_cloud_dev_token,
    )
    db.dataset.add_creds_key(hub_cloud_dev_managed_creds_key, managed=True)
    assert db.exec_option == "compute_engine"


@pytest.mark.slow
@pytest.mark.parametrize(
    "runtime",
    ["runtime", None],
    indirect=True,
)
@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_vectorstore_factory(hub_cloud_dev_token, hub_cloud_path, runtime):
    db = vectorstore_factory(
        path=hub_cloud_path,
        runtime=runtime,
        token=hub_cloud_dev_token,
    )

    if runtime is not None:
        assert isinstance(db, DeepMemoryVectorStore)
    else:
        assert isinstance(db, DeepLakeVectorStore)
