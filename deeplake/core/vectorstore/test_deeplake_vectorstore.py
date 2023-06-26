import numpy as np
import pytest

import deeplake
from deeplake.core.vectorstore.deeplake_vectorstore import (
    DeepLakeVectorStore,
    VectorStore,
)
from deeplake.core.vectorstore import utils
from deeplake.tests.common import requires_libdeeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
)

from math import isclose
import os

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


def test_id_backward_compatibility(local_path):
    num_of_items = 10
    embedding_dim = 100

    ids = [f"{i}" for i in range(num_of_items)]
    embedding = [np.zeros(embedding_dim) for i in range(num_of_items)]
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


@requires_libdeeplake
def test_search_basic(local_path, hub_cloud_dev_token):
    """Test basic search features"""
    # Initialize vector store object and add data
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        token=hub_cloud_dev_token,
    )
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
        exec_option="python",
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
    # Use indra implementation to search the data
    data_ce = vector_store_cloud.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        k=2,
        return_tensors=["ids", "text"],
    )
    assert len(data_ce["text"]) == 2
    assert (
        sum([tensor in data_ce.keys() for tensor in vector_store_cloud.dataset.tensors])
        == 2
    )  # One for each return_tensors
    assert len(data_ce.keys()) == 3  # One for each return_tensors + score

    with pytest.raises(ValueError):
        data_ce = vector_store_cloud.search(
            query=f"SELECT * WHERE id=='{vector_store_cloud.dataset.ids[0].numpy()[0]}'",
            embedding=query_embedding,
            exec_option="compute_engine",
            k=2,
            return_tensors=["ids", "text"],
        )

    # Run a full custom query
    test_text = vector_store_cloud.dataset.text[0].data()["value"]
    data_q = vector_store_cloud.search(
        query=f"select * where text == '{test_text}'", exec_option="compute_engine"
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
        exec_option="python",
        k=2,
        return_tensors=["id", "text"],
        filter={"metadata": {"abc": 1}},
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
        exec_option="python",
        k=2,
        return_tensors=["id", "text"],
        filter=filter_fn,
    )
    assert len(data_e_f["text"]) == 1
    assert (
        sum([tensor in data_e_f.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_e_f.keys()) == 2

    # Check returning views
    data_p_v = vector_store.search(
        embedding=query_embedding,
        exec_option="python",
        k=2,
        filter={"metadata": {"abc": 1}},
        return_view=True,
    )
    assert len(data_p_v) == 1
    assert isinstance(data_p_v.text[0].data()["value"], str)
    assert data_p_v.embedding[0].numpy().size > 0

    data_ce_v = vector_store_cloud.search(
        embedding=query_embedding, exec_option="compute_engine", k=2, return_view=True
    )
    assert len(data_ce_v) == 2
    assert isinstance(data_ce_v.text[0].data()["value"], str)
    assert data_ce_v.embedding[0].numpy().size > 0

    # Check exceptions
    with pytest.raises(ValueError):
        vector_store.search(embedding=query_embedding, exec_option="remote_tensor_db")
    with pytest.raises(ValueError):
        vector_store.search()
    with pytest.raises(ValueError):
        vector_store.search(query="dummy", exec_option="python")
    with pytest.raises(ValueError):
        vector_store.search(
            query="dummy",
            return_tensors=["non_existant_tensor"],
            exec_option="compute_engine",
        )
    with pytest.raises(ValueError):
        vector_store.search(query="dummy", return_tensors=["ids"], exec_option="python")
    with pytest.raises(ValueError):
        vector_store.search(
            query="dummy", filter=filter_fn, exec_option="compute_engine"
        )
    with pytest.raises(ValueError):
        vector_store.search(
            embedding_function=embedding_fn,
        )

    with pytest.raises(ValueError):
        vector_store = DeepLakeVectorStore(path="mem://xyz")

        vector_store.search(
            embedding=query_embedding,
            exec_option="python",
            k=2,
            filter={"metadata": {"abc": 1}},
            return_view=True,
        )

    vector_store = DeepLakeVectorStore(path="mem://xyz")
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    data = vector_store.search(
        exec_option="python",
        embedding_function=embedding_fn3,
        embedding_data=["dummy"],
        return_view=True,
        k=2,
    )
    assert len(data) == 2
    assert isinstance(data.text[0].data()["value"], str)
    assert data.embedding[0].numpy().size > 0

    data = vector_store.search(
        exec_option="python",
        filter={"metadata": {"abcdefh": 1}},
        embedding=None,
        return_view=True,
        k=2,
    )
    assert len(data) == 0

    data = vector_store.search(
        exec_option="python",
        filter={"metadata": {"abcdefh": 1}},
        embedding=query_embedding,
        k=2,
    )
    assert len(data) == 4
    assert len(data["id"]) == 0
    assert len(data["metadata"]) == 0
    assert len(data["text"]) == 0
    assert len(data["score"]) == 0

    with pytest.raises(ValueError):
        data = vector_store.search(
            exec_option="compute_engine",
            filter=filter_fn,
            k=2,
        )

    with pytest.raises(ValueError):
        data = vector_store.search(
            exec_option="compute_engine",
            query="select * where metadata == {'abcdefg': 28}",
            return_tensors=["metadata", "id"],
        )

    vector_store = DeepLakeVectorStore(
        path="mem://xyz", embedding_function=embedding_fn
    )
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)
    result = vector_store.search(embedding=np.zeros((1, EMBEDDING_DIM)))
    assert len(result) == 4


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
    assert data_p["ids"] == data_ce["ids"]
    assert data_p["metadata"] == data_ce["metadata"]

    # use indra implementation to search the data
    data_ce = vector_store.search(
        embedding=None,
        exec_option="compute_engine",
        distance_metric=distance_metric,
        filter={"metadata": {"abcdefg": 28}},
    )

    assert data_ce["ids"] == "0"

    with pytest.raises(ValueError):
        # use indra implementation to search the data
        data_ce = vector_store.search(
            query="select * where metadata == {'abcdefg': 28}",
            exec_option="compute_engine",
            distance_metric=distance_metric,
            filter={"metadata": {"abcdefg": 28}},
        )

    data_ce = vector_store.search(
        query="select * where ids == '0'",
        exec_option="compute_engine",
    )
    assert data_ce["ids"] == ["0"]


@requires_libdeeplake
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
    assert data_ce["ids"] == data_db["ids"]


def test_delete(local_path, capsys):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
        verbose=False,
    )

    # add data to the dataset:
    vector_store.add(id=ids, embedding=embeddings, text=texts, metadata=metadatas)

    output = (
        f"Dataset(path='{local_path}', tensors=['embedding', 'id', 'metadata', 'text'])\n\n"
        "  tensor      htype      shape     dtype  compression\n"
        "  -------    -------    -------   -------  ------- \n"
        " embedding  embedding  (10, 100)  float32   None   \n"
        "    id        text      (10, 1)     str     None   \n"
        " metadata     json      (10, 1)     str     None   \n"
        "   text       text      (10, 1)     str     None   \n"
    )

    vector_store.summary()
    captured = capsys.readouterr()
    assert output in captured.out

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
    vector_store = DeepLakeVectorStore(
        path=local_path,
        overwrite=True,
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
    )
    # add data to the dataset:
    vector_store.add(ids=ids, docs=texts)

    # delete the data in the dataset by id:
    vector_store.delete(row_ids=[0])
    assert len(vector_store.dataset) == NUMBER_OF_DATA - 1

    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("ids", htype="text")
    ds.create_tensor("embedding", htype="embedding")
    ds.extend(
        {
            "ids": ids,
            "embedding": embeddings,
        }
    )

    vector_store = DeepLakeVectorStore(
        path=local_path,
    )
    vector_store.delete(ids=ids[:3])
    assert len(vector_store) == NUMBER_OF_DATA - 3

    with pytest.raises(ValueError):
        vector_store.delete(ids=ids[5:7], exec_option="remote_tensor_db")


def test_ingestion(local_path, capsys):
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

    with pytest.raises(Exception):
        # add data to the dataset:
        vector_store.add(
            embedding=embeddings,
            text=texts[: number_of_data - 2],
            id=ids,
            metadata=metadatas,
            something=texts[: number_of_data - 2],
        )

    vector_store.add(embedding=embeddings, text=texts, id=ids, metadata=metadatas)
    captured = capsys.readouterr()

    output = (
        f"Dataset(path='{local_path}', tensors=['embedding', 'id', 'metadata', 'text'])\n\n"
        "  tensor      htype       shape      dtype  compression\n"
        "  -------    -------     -------    -------  ------- \n"
        " embedding  embedding  (1000, 100)  float32   None   \n"
        "    id        text      (1000, 1)     str     None   \n"
        " metadata     json      (1000, 1)     str     None   \n"
        "   text       text      (1000, 1)     str     None   \n"
    )
    assert output in captured.out

    assert len(vector_store) == number_of_data
    assert list(vector_store.dataset.tensors) == [
        "embedding",
        "id",
        "metadata",
        "text",
    ]
    assert list(vector_store.tensors()) == [
        "embedding",
        "id",
        "metadata",
        "text",
    ]

    vector_store.add(
        embedding_function=embedding_fn3,
        embedding_data=texts,
        text=texts,
        id=ids,
        metadata=metadatas,
    )
    captured = capsys.readouterr()

    output = (
        f"Dataset(path='{local_path}', tensors=['embedding', 'id', 'metadata', 'text'])\n\n"
        "  tensor      htype       shape      dtype  compression\n"
        "  -------    -------     -------    -------  ------- \n"
        " embedding  embedding  (2000, 100)  float32   None   \n"
        "    id        text      (2000, 1)     str     None   \n"
        " metadata     json      (2000, 1)     str     None   \n"
        "   text       text      (2000, 1)     str     None   \n"
    )
    assert output in captured.out
    assert len(vector_store) == 2 * number_of_data
    assert list(vector_store.tensors()) == [
        "embedding",
        "id",
        "metadata",
        "text",
    ]

    vector_store.add(
        embedding_function=embedding_fn3,
        embedding_data=25 * texts,
        text=25 * texts,
        id=25 * ids,
        metadata=25 * metadatas,
    )
    captured = capsys.readouterr()

    output = (
        f"Dataset(path='{local_path}', tensors=['embedding', 'id', 'metadata', 'text'])\n\n"
        "  tensor      htype       shape       dtype  compression\n"
        "  -------    -------     -------     -------  ------- \n"
        " embedding  embedding  (27000, 100)  float32   None   \n"
        "    id        text      (27000, 1)     str     None   \n"
        " metadata     json      (27000, 1)     str     None   \n"
        "   text       text      (27000, 1)     str     None   \n"
    )
    assert output in captured.out
    assert len(vector_store) == 27000
    assert list(vector_store.tensors()) == [
        "embedding",
        "id",
        "metadata",
        "text",
    ]


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


def test_multiple_embeddings(local_path, capsys):
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
    with pytest.raises(AssertionError):
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
