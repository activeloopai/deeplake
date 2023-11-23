import json
import pytest
import sys
from time import time

import numpy as np

from deeplake.constants import MAX_BYTES_PER_MINUTE, TARGET_BYTE_SIZE
from deeplake.core.vectorstore.embeddings.embedder import (
    DeepLakeEmbedder,
    chunk_by_bytes,
)


EMBEDDING_DIM = 15


def test_chunk_by_bytes():
    data = ["a" * 10000] * 10  # 10 chunks of 10000 bytes

    batched_data = chunk_by_bytes(data, target_byte_size=10)
    serialized_data = json.dumps(batched_data)
    byte_size = len(serialized_data.encode("utf-8"))
    list_wieght = 100
    assert (
        byte_size <= 100000 + list_wieght
    ), "Chunking by bytes did not work as expected!"


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Sometimes MacOS fails this test due to speed issues",
)
def test_embedder_with_func():
    def embed_documents(documents):
        return [np.random.rand(EMBEDDING_DIM) for doc in documents]

    embedder = DeepLakeEmbedder(embedding_function=embed_documents)
    documents = ["a" * 10000] * 10  # 10 chunks of 10000 bytes
    embeddings = embedder.embed_documents(documents)
    assert len(embeddings) == 10, "Embedding function did not work as expected!"

    embedder = DeepLakeEmbedder(embedding_function=embed_documents)
    documents = ["a" * 10000] * 10000  # 10 chunks of 10000 bytes
    embeddings = embedder.embed_documents(documents)
    assert len(embeddings) == 10000, "Embedding function did not work as expected!"

    documents = ["a" * 10000] * 10
    start_time = time()
    embeddings = embedder.embed_documents(
        documents,
        rate_limiter={
            "enabled": True,
            "bytes_per_minute": MAX_BYTES_PER_MINUTE,
            "batch_byte_size": TARGET_BYTE_SIZE,
        },
    )
    end_time = time()
    elapsed_minutes = end_time - start_time
    expected_time = 60 * (
        len(documents) * 10000 / MAX_BYTES_PER_MINUTE
    )  # each data chunk has 10 bytes
    tolerance = 0.1

    assert len(embeddings) == 10, "Embedding function did not work as expected!"
    assert (
        abs(elapsed_minutes - expected_time) <= tolerance
    ), "Rate limiting did not work as expected!"


def test_embedder_with_class():
    class Embedder:
        def embed_documents(self, documents):
            return [np.random.rand(EMBEDDING_DIM) for doc in documents]

        def embed_query(self, query):
            return np.random.rand(EMBEDDING_DIM)

    embedder_obj = Embedder()
    embedder = DeepLakeEmbedder(embedding_function=embedder_obj)
    documents = ["a" * 10000] * 10  # 10 chunks of 10000 bytes
    embeddings = embedder.embed_documents(documents)
    assert len(embeddings) == 10, "Embedding function did not work as expected!"

    embedder = DeepLakeEmbedder(embedding_function=embedder_obj)
    documents = ["a" * 10000] * 10000  # 10 chunks of 10000 bytes
    embeddings = embedder.embed_documents(documents)
    assert len(embeddings) == 10000, "Embedding function did not work as expected!"

    embeddings = embedder.embed_query(documents[0])
    assert (
        len(embeddings) == EMBEDDING_DIM
    ), "Embedding function did not work as expected!"
