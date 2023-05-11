import numpy as np

import pytest
import random

import deeplake
from deeplake.constants import MB
from deeplake.core.vectorstore.vector_search.ingestion import data_ingestion

random.seed(1)


def corrupted_embedding_function(emb):
    p = random.uniform(0, 1)
    if p > 0.9:
        raise Exception("CorruptedEmbeddingFunction")
    return np.zeros((1, 1536), dtype=np.float32)


def test_data_ingestion():
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

    dataset = deeplake.empty("mem://xyz")
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

    data_ingestion.run_data_ingestion(
        dataset=dataset,
        elements=data,
        embedding_function=None,
        ingestion_batch_size=1024,
        num_workers=2,
    )

    assert len(dataset) == 4
    extended_data = data * 10
    with pytest.raises(Exception):
        data_ingestion.run_data_ingestion(
            dataset=dataset,
            elements=extended_data,
            embedding_function=corrupted_embedding_function,
            ingestion_batch_size=1,
            num_workers=2,
        )

    with pytest.raises(ValueError):
        data_ingestion.run_data_ingestion(
            dataset=dataset,
            elements=extended_data,
            embedding_function=corrupted_embedding_function,
            ingestion_batch_size=0,
            num_workers=2,
        )
