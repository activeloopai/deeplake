import numpy as np
import pytest

from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore
from deeplake.core.vectorstore import utils
from deeplake.tests.common import requires_libdeeplake


embedding_dim = 100
# create data
texts, embeddings, ids, metadatas = utils.create_data(
    number_of_data=10, embedding_dim=embedding_dim
)

query_embedding = np.random.uniform(low=-10, high=10, size=(embedding_dim)).astype(
    np.float32
)


def embedding_fn(text, embedding_dim=100):
    return np.zeros((embedding_dim,)).astype(np.float32)


def test_tensor_dict():
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
        tensors_dict=[
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
    assert "texts_custom" in data.keys() and "ids" in data.keys()


@requires_libdeeplake
def test_search_basic(hub_cloud_dev_token):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    # add data to the dataset:
    vector_store.add(embedding=embeddings, text=texts, metadata=metadatas)

    # check that default works
    data_default = vector_store.search(
        embedding=query_embedding,
    )
    assert (len(data_default.keys())) > 0

    # use python implementation to search the data
    data_p = vector_store.search(
        embedding=query_embedding,
        exec_option="python",
        k=2,
        return_tensors=["ids", "text"],
        filter={"metadata": {"abc": "value"}},
    )

    assert len(data_p["text"]) == 2
    assert (
        sum([tensor in data_p.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_p.keys()) == 3  # One for each return_tensors + score

    # initialize vector store object in the cloud for indra testing:
    vector_store_cloud = DeepLakeVectorStore(
        dataset_path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )
    # use indra implementation to search the data
    data_ce = vector_store_cloud.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        k=2,
        return_tensors=["ids", "text"],
    )
    assert len(data_ce["text"]) == 2
    assert (
        sum([tensor in data_ce.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_ce.keys()) == 3  # One for each return_tensors + score

    # run a full custom query
    test_text = vector_store.dataset.text[0].data()["value"]
    data_q = vector_store.search(
        query=f"select * where text == '{test_text}'", exec_option="compute_engine"
    )

    assert len(data_q["text"]) == 1
    assert data_q["text"][0] == test_text
    assert sum(
        [tensor in data_q.keys() for tensor in vector_store.dataset.tensors]
    ) == len(
        vector_store.dataset.tensors
    )  # One for each tensor - embedding + score

    # Run a filter query
    data_e = vector_store.search(
        prompt="dummy",
        embedding_function=embedding_fn,
        exec_option="python",
        k=2,
        return_tensors=["ids", "text"],
        filter={"metadata": {"abc": "value"}},
    )
    assert len(data_e["text"]) == 2
    assert (
        sum([tensor in data_e.keys() for tensor in vector_store.dataset.tensors]) == 2
    )  # One for each return_tensors
    assert len(data_e.keys()) == 3  # One for each return_tensors + score

    with pytest.raises(ValueError):
        vector_store.search(embedding=query_embedding, exec_option="remote_tensor_db")
    with pytest.raises(ValueError):
        vector_store.search()
    with pytest.raises(ValueError):
        vector_store.search(query="dummy", exec_option="python")
    with pytest.raises(ValueError):
        vector_store.search(
            query="dummy", return_tensors=["dummy"], exec_option="compute_engine"
        )
    with pytest.raises(ValueError):
        vector_store.search(query="dummy", return_tensors=["ids"], exec_option="python")


@requires_libdeeplake
@pytest.mark.parametrize("distance_metric", ["L1", "L2", "COS", "MAX", "DOT"])
def test_search_quantitative(distance_metric, hub_cloud_dev_token):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
        token=hub_cloud_dev_token,
    )

    # add data to the dataset:
    vector_store.add(embeddings=embeddings, texts=texts)

    # use python implementation to search the data
    data_p = vector_store.search(
        embedding=query_embedding, exec_option="python", distance_metric=distance_metric
    )

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )

    # use indra implementation to search the data
    data_ce = vector_store.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        distance_metric=distance_metric,
    )
    np.testing.assert_almost_equal(data_p["score"], data_ce["score"])
    np.testing.assert_almost_equal(data_p["text"], data_ce["text"])
    np.testing.assert_almost_equal(data_p["ids"], data_ce["ids"])
    np.testing.assert_almost_equal(data_p["metadata"], data_ce["metadata"])


@requires_libdeeplake
def test_search_managed(hub_cloud_dev_token):
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="hub://testingacc2/vectorstore_test",
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

    assert len(data_ce.keys) == len(data_ce.data_db)
    np.testing.assert_almost_equal(data_ce["score"], data_db["score"])
    np.testing.assert_almost_equal(data_ce["text"], data_db["text"])
    np.testing.assert_almost_equal(data_ce["ids"], data_db["ids"])
    np.testing.assert_almost_equal(data_ce["metadata"], data_db["metadata"])


def test_delete():
    embedding_dim = 1536
    # create data
    texts, embeddings, ids, metadatas = utils.create_data(
        number_of_data=1000, embedding_dim=embedding_dim
    )

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
    )

    # add data to the dataset:
    vector_store.add(embeddings=embeddings, texts=texts, ids=ids, metadatas=metadatas)

    # delete the data in the dataset by id:
    vector_store.delete(ids=["1", "2", "3"])
    assert len(vector_store) == 997

    vector_store.delete(filter={"abcdefg": 113})
    assert len(vector_store) == 996

    vector_store.delete(delete_all=True)
    assert len(vector_store) == 0

    vector_store.force_delete_by_path("./deeplake_vector_store")
    dirs = os.listdir("./")
    assert "./deeplake_vector_store" not in dirs


def test_ingestion(capsys):
    embedding_dim = 1536
    # create data
    texts, embeddings, ids, metadatas = utils.create_data(
        number_of_data=1000, embedding_dim=embedding_dim
    )

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
        verbose=True,
    )

    with pytest.raises(Exception):
        # add data to the dataset:
        vector_store.add(
            embeddings=embeddings, texts=texts[:100], ids=ids, metadatas=metadatas
        )

    vector_store.add(embeddings=embeddings, texts=texts, ids=ids, metadatas=metadatas)
    captured = capsys.readouterr()
    output = (
        "Dataset(path='./deeplake_vector_store', tensors=['embedding', 'ids', 'metadata', 'text'])\n\n"
        "  tensor      htype       shape       dtype  compression\n"
        "  -------    -------     -------     -------  ------- \n"
        " embedding  embedding  (1000, 1536)  float32   None   \n"
        "    ids       text      (1000, 1)      str     None   \n"
        " metadata     json      (1000, 1)      str     None   \n"
        "   text       text      (1000, 1)      str     None   \n"
    )

    assert output in captured.out

    assert len(vector_store) == 1000
    assert list(vector_store.dataset.tensors.keys()) == [
        "embedding",
        "ids",
        "metadata",
        "text",
    ]

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
        verbose=True,
        embedding_function=embedding_fn,
    )

    vector_store.add(texts=texts, ids=ids, metadatas=metadatas)

    np.testing.assert_array_equal(
        vector_store.dataset.embedding.numpy(), np.zeros((1000, 100), dtype=np.float32)
    )
