import numpy as np
import pytest
import random
import string

from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore

random.seed(0)
np.random.seed(0)


def generate_random_string(length):
    # Define the character set to include letters (both lowercase and uppercase) and digits
    characters = string.ascii_letters + string.digits
    # Generate a random string of the specified length
    random_string = "".join(random.choice(characters) for _ in range(length))

    return random_string


def generate_random_json(length):
    string = "abcdefg"
    integer = np.random.randint(0, length)
    return {string: integer}


def create_data(number_of_data, embedding_dim=100):
    embeddings = np.random.randint(0, 255, (number_of_data, embedding_dim))
    texts = [generate_random_string(1000) for i in range(number_of_data)]
    ids = [f"{i}" for i in range(number_of_data)]
    metadata = [generate_random_json(100) for i in range(number_of_data)]
    return texts, embeddings, ids, metadata


# create data
texts, embeddings, ids, metadatas = create_data(number_of_data=1000)


@pytest.mark.parametrize("distance_metric", ["L1", "L2", "COS", "MAX", "DOT"])
def test_search(distance_metric):
    k = 4
    embedding_dim = 1536
    query_embedding = np.random.randint(0, 255, (1, embedding_dim))

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
    )

    # add data to the dataset:
    vector_store.add(embeddings=embeddings, texts=texts)

    # use python implementation to search the data
    python_view, python_indices, python_scores = vector_store.search(
        embedding=query_embedding, exec_option="python"
    )

    # use indra implementation to search the data
    indra_view, indra_indices, indra_scores = vector_store.search(
        embedding=query_embedding, exec_option="indra"
    )

    np.testing.assert_almost_equal(python_indices, indra_indices)
    np.testing.assert_almost_equal(python_scores, indra_scores)


def test_delete():
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

    vector_store.delete(filter={"abcdefg": 0})
