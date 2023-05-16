import pytest
import logging

import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.constants import DEFAULT_VECTORSTORE_DEEPLAKE_PATH
from deeplake.tests.common import requires_libdeeplake


logger = logging.getLogger(__name__)


@requires_libdeeplake
def test_create(caplog, hub_cloud_dev_token):
    # dataset creation
    dataset = dataset_utils.create_or_load_dataset(
        dataset_path="./test-dataset",
        token=None,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="compute_engine",
        overwrite=True,
    )
    assert len(dataset) == 0
    assert set(dataset.tensors.keys()) == {
        "embedding",
        "ids",
        "metadata",
        "text",
    }

    dataset = dataset_utils.create_or_load_dataset(
        dataset_path="hub://testingacc2/vectorstore_dbengine",
        token=hub_cloud_dev_token,
        creds={},
        logger=logger,
        read_only=False,
        exec_option="tensor_db",
        overwrite=True,
    )
    assert len(dataset) == 0
    assert set(dataset.tensors.keys()) == {
        "embedding",
        "ids",
        "metadata",
        "text",
    }


def test_load(caplog, hub_cloud_dev_token):
    # dataset loading
    dataset = dataset_utils.create_or_load_dataset(
        dataset_path="hub://testingacc2/vectorstore_test",
        creds={},
        logger=logger,
        exec_option="python",
        overwrite=False,
        read_only=True,
        token=hub_cloud_dev_token,
    )
    assert dataset.max_len == 10

    ds = deeplake.empty(DEFAULT_VECTORSTORE_DEEPLAKE_PATH, overwrite=True)

    test_logger = logging.getLogger("test_logger")
    with caplog.at_level(logging.WARNING, logger="test_logger"):
        # dataset loading
        dataset = dataset_utils.create_or_load_dataset(
            dataset_path=DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
            token=None,
            creds={},
            logger=test_logger,
            read_only=False,
            exec_option="python",
        )
        assert (
            f"The default deeplake path location is used: {DEFAULT_VECTORSTORE_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset." in caplog.text
        )
        tensors = ["text", "embedding", "ids", "metadata"]
        for tensor in tensors:
            assert (
                f"`{tensor}` tensor does not exist in the dataset. If you created dataset manually "
                "and stored text data in another tensor, consider copying the contents of that "
                f"tensor into `{tensor}` tensor and deleting if afterwards. To view dataset content "
                "run ds.summary()" in caplog.text
            )


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

    embedings = dataset_utils.fetch_embeddings("python", dataset, logger)
    assert len(embedings) == 9

    embedings = dataset_utils.fetch_embeddings("compute_engine", dataset, logger)
    assert embedings is None

    embedings = dataset_utils.fetch_embeddings("tensor_db", dataset, logger)
    assert embedings is None


def test_get_embedding():
    def embedding_function(arr):
        return np.array([0.5, 0.6, 4, 3, 5], dtype=np.float64)

    query = "tql query"
    with pytest.raises(Exception):
        embedding = dataset_utils.get_embedding(
            embedding=None, query=query, embedding_function=None
        )

    embedding_func = embedding_function
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

    with pytest.warns(UserWarning):
        embedding = dataset_utils.get_embedding(
            embedding=embedding_vector, query=query, embedding_function=embedding_func
        )


def test_preprocess_tensors():
    texts = ["a", "b", "c", "d"]
    processed_tensors, ids = dataset_utils.preprocess_tensors(
        ids=None, texts=texts, metadatas=None, embeddings=None
    )

    assert len(processed_tensors["ids"]) == 4
    assert processed_tensors["texts"] == texts
    assert processed_tensors["metadatas"] == [{}, {}, {}, {}]
    assert processed_tensors["embeddings"] == [None, None, None, None]

    texts = ("a", "b", "c", "d")
    ids = np.array([1, 2, 3, 4])
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    embeddings = [np.array([0.1, 0.2, 0.3, 0.4])] * len(texts)
    processed_tensors, _ = dataset_utils.preprocess_tensors(
        ids=ids, texts=texts, metadatas=metadatas, embeddings=embeddings
    )
    assert np.array_equal(processed_tensors["ids"], ids)
    assert processed_tensors["texts"] == list(texts)
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
    elements, ids = dataset_utils.create_elements(
        ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas
    )

    for i in range(len(elements)):
        assert np.array_equal(elements[i]["text"], targ_elements[i]["text"])
        assert np.array_equal(elements[i]["id"], targ_elements[i]["id"])
        assert np.array_equal(elements[i]["embedding"], targ_elements[i]["embedding"])
        assert np.array_equal(elements[i]["metadata"], targ_elements[i]["metadata"])
