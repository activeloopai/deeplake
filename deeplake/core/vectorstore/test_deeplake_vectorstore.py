import numpy as np
import os
import pytest
import random
import string

from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore
from deeplake.core.vectorstore.deeplake_vectorstore import utils
from deeplake.tests.common import requires_libdeeplake


embedding_dim = 1536
# create data
texts, embeddings, ids, metadatas = utils.create_data(
    number_of_data=1000, embedding_dim=embedding_dim
)


@requires_libdeeplake
@pytest.mark.parametrize("distance_metric", ["L1", "L2", "COS", "MAX", "DOT"])
def test_search(distance_metric):
    k = 4
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

    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="hub://activeloop-test/deeplake_vector_store-test", read_only=True
    )
    db_engine_view, db_engine_indices, db_engine_scores = vector_store.search(
        embedding=query_embedding, exec_option="db_engine"
    )
    np.testing.assert_almost_equal(python_scores, db_engine_scores)

    view, indices, scores = vector_store.search(query=texts[0])
    assert len(view) == 1
    assert indices == [0]


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
    assert len(vector_store) == 985

    vector_store.delete(delete_all=True)
    assert len(vector_store) == 0

    vector_store.force_delete_by_path("./deeplake_vector_store")
    dirs = os.listdir("./")
    assert "./deeplake_vector_store" not in dirs


def test_ingestion():
    # initialize vector store object:
    vector_store = DeepLakeVectorStore(
        dataset_path="./deeplake_vector_store",
        overwrite=True,
    )

    with pytest.raises(Exception):
        # add data to the dataset:
        vector_store.add(
            embeddings=embeddings, texts=texts[:100], ids=ids, metadatas=metadatas
        )

    vector_store.add(embeddings=embeddings, texts=texts, ids=ids, metadatas=metadatas)
    assert len(vector_store) == 1000
    assert list(vector_store.dataset.tensors.keys()) == [
        "embeddings",
        "ids",
        "metadatas",
        "texts",
    ]
