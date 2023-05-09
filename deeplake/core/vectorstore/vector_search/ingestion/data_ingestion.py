import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search import utils
from deeplake.util.exceptions import TransformError


def run_data_ingestion(
    elements,
    dataset,
    embedding_function,
    ingestion_batch_size,
    num_workers,
):
    batch_size = min(ingestion_batch_size, len(elements))
    if batch_size == 0:
        return []

    batched = [
        elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
    ]

    ingest(_embedding_function=embedding_function).eval(
        batched,
        dataset,
        num_workers=min(num_workers, len(batched) // max(num_workers, 1)),
    )


@deeplake.compute
def ingest(sample_in: list, sample_out: list, _embedding_function) -> None:
    text_list = [s["text"] for s in sample_in]

    embeds = [None] * len(text_list)
    if _embedding_function is not None:
        embeddings = _embedding_function.embed_documents(text_list)
        embeds = [np.array(e, dtype=np.float32) for e in embeddings]

    for s, e in zip(sample_in, embeds):
        embedding = e if _embedding_function else s["embedding"]
        sample_out.append(
            {
                "text": s["text"],
                "metadata": s["metadata"],
                "ids": s["id"],
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )
