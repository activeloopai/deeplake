import numpy as np

import pytest
import random
from functools import partial


import deeplake
from deeplake.constants import MB
from deeplake.core.vectorstore.vector_search.ingestion import ingest_data
from deeplake.util.exceptions import FailedIngestionError


random.seed(1)


def corrupted_embedding_function(emb, threshold):
    p = random.uniform(0, 1)
    if p > threshold:
        raise Exception("CorruptedEmbeddingFunction")
    return np.zeros((len(emb), 1536), dtype=np.float32)


def test_ingest_data():
    data = [
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

    dataset = deeplake.empty("./xyzabc", overwrite=True)
    dataset.create_tensor(
        "text",
        htype="text",
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        chunk_compression="lz4",
    )
    dataset.create_tensor(
        "metadata",
        htype="json",
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        chunk_compression="lz4",
    )
    dataset.create_tensor(
        "embedding",
        htype="embedding",
        dtype=np.float32,
        create_id_tensor=False,
        create_sample_info_tensor=False,
        max_chunk_size=64 * MB,
        create_shape_tensor=True,
    )
    dataset.create_tensor(
        "ids",
        htype="text",
        create_id_tensor=False,
        create_sample_info_tensor=False,
        create_shape_tensor=False,
        chunk_compression="lz4",
    )

    ingest_data.run_data_ingestion(
        dataset=dataset,
        elements=data,
        embedding_function=None,
        ingestion_batch_size=1024,
        num_workers=2,
    )

    assert len(dataset) == 4
    extended_data = data * 5000
    embedding_function = partial(corrupted_embedding_function, threshold=0.9)

    ingest_data.run_data_ingestion(
        dataset=dataset,
        elements=extended_data,
        embedding_function=embedding_function,
        ingestion_batch_size=1024,
        num_workers=2,
    )
    assert len(dataset) == 20004

    extended_data = extended_data * 10
    embedding_function = partial(corrupted_embedding_function, threshold=0.95)
    with pytest.raises(FailedIngestionError):
        ingest_data.run_data_ingestion(
            dataset=dataset,
            elements=extended_data,
            embedding_function=embedding_function,
            ingestion_batch_size=1024,
            num_workers=2,
        )

    extended_data = extended_data * 10
    with pytest.raises(FailedIngestionError):
        ingest_data.run_data_ingestion(
            dataset=dataset,
            elements=extended_data,
            embedding_function=embedding_function,
            ingestion_batch_size=1024,
            num_workers=2,
        )

    with pytest.raises(ValueError):
        ingest_data.run_data_ingestion(
            dataset=dataset,
            elements=extended_data,
            embedding_function=corrupted_embedding_function,
            ingestion_batch_size=0,
            num_workers=2,
        )
