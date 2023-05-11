from typing import Dict, List, Any, Callable

import numpy as np

import deeplake
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import utils
from deeplake.util.exceptions import TransformError


def run_data_ingestion(
    elements: List[Dict[str, Any]],
    dataset: DeepLakeDataset,
    embedding_function: Callable,
    ingestion_batch_size: int,
    num_workers: int,
):
    """Running data ingestion into deeplake dataset.

    Args:
        elements (List[Dict[str, Any]]): List of dictionaries. Each dictionary contains mapping of
            names of 4 tensors (i.e. "embedding", "metadata", "ids", "text") to their corresponding values.
        dataset (DeepLakeDataset): deeplake dataset object.
        embedding_function (Callable): function used to convert query into an embedding.
        ingestion_batch_size (int): The batch size to use during ingestion.
        num_workers (int): The number of workers to use for ingesting data in parallel.
    """
    batch_size = min(ingestion_batch_size, len(elements))
    if batch_size == 0:
        raise ValueError("batch_size must be a positive number greater than zero.")

    batched = [
        elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
    ]

    num_workers = min(num_workers, len(batched) // max(num_workers, 1))
    checkpoint_interval = int(
        (0.1 * len(batched) // max(num_workers, 1)) * max(num_workers, 1)
    )

    ingest(embedding_function=embedding_function).eval(
        batched,
        dataset,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
    )


@deeplake.compute
def ingest(sample_in: list, sample_out: list, embedding_function) -> None:
    text_list = [s["text"] for s in sample_in]

    embeds = [None] * len(text_list)
    if embedding_function is not None:
        try:
            embeddings = embedding_function(text_list)
        except Exception as e:
            raise Exception(
                "Could not use embedding function. Please try again with a different embedding function."
            )
        embeds = [np.array(e, dtype=np.float32) for e in embeddings]

    for s, e in zip(sample_in, embeds):
        embedding = e if embedding_function else s["embedding"]
        sample_out.append(
            {
                "text": s["text"],
                "metadata": s["metadata"],
                "ids": s["id"],
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )
