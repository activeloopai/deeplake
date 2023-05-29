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


def embedding_fn(text, embedding_dim=100):
    return np.zeros((len(text), embedding_dim))

    
@requires_libdeeplake
@pytest.mark.parametrize("distance_metric", ["L1", "L2", "COS", "MAX", "DOT"])
def test_search(distance_metric, hub_cloud_dev_token):
    query_embedding = np.random.uniform(low=-10, high=10, size=(embedding_dim)).astype(
        np.float32
    )

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="/Users/istranic/ActiveloopCode/Datasets/Debug/vs_tests",
        overwrite=True,
    )

    # add data to the dataset:
    vector_store.add(embeddings=embeddings, texts=texts)

    # use python implementation to search the data
    data = vector_store.search(
        embedding=query_embedding,
        exec_option="python",
        distance_metric=distance_metric,
        k=3,
    )
    assert len(data["text"]) == 3

    # use python implementation to search the data
    data_p = vector_store.search(
        embedding=query_embedding, exec_option="python", distance_metric=distance_metric
    )

    # Add assertion about keys

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="hub://testingacc2/vectorstore_test",
        read_only=True,
        token=hub_cloud_dev_token,
    )

    vector_store.add(embeddings=embeddings, texts=texts)

    # use indra implementation to search the data
    data_ce = vector_store.search(
        embedding=query_embedding,
        exec_option="compute_engine",
        distance_metric=distance_metric,
    )
    np.testing.assert_almost_equal(data_p["score"], data_ce["score"])
    np.testing.assert_almost_equal(data_p["text"], data_ce["text"])

    data_db = vector_store.search(
        embedding=query_embedding,
        exec_option="tensor_db",
        distance_metric=distance_metric,
    )

    np.testing.assert_almost_equal(data_ce["score"], data_db["score"])
    np.testing.assert_almost_equal(data_ce["text"], data_db["text"])

    data_q = vector_store.search(
        query=f"select * where text == {texts[0]}", exec_option="compute_engine"
    )
    assert len(data_q["text"]) == 1
    assert data_q["text"] == texts[0]

    with pytest.raises(ValueError):
        vector_store.search(embedding=query_embedding, exec_option="remote_tensor_db")
    with pytest.raises(ValueError):
        vector_store.search()

    with pytest.raises(ValueError):
        vector_store.search(query="dummy", exec_option="python")


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
