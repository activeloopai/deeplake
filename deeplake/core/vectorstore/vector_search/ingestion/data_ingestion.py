from typing import Dict, List, Any, Callable, Optional

import numpy as np

import deeplake
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import utils
from deeplake.util.exceptions import TransformError, FailedIngestionError
from deeplake.constants import (
    MAX_VECTORSTORE_INGESTION_RETRY_ATTEMPTS,
    MAX_CHECKPOINTING_INTERVAL,
    MAX_DATASET_LENGTH_FOR_CACHING,
)


class DataIngestion:
    def __init__(
        self,
        elements: List[Dict[str, Any]],
        dataset: DeepLakeDataset,
        embedding_function: Optional[Callable],
        ingestion_batch_size: int,
        num_workers: int,
        retry_attempt: int,
        total_samples_processed: int,
    ):
        self.elements = elements
        self.dataset = dataset
        self.embedding_function = embedding_function
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        self.retry_attempt = retry_attempt
        self.total_samples_processed = total_samples_processed

    def collect_batched_data(self, ingestion_batch_size=None):
        ingestion_batch_size = ingestion_batch_size or self.ingestion_batch_size
        batch_size = min(ingestion_batch_size, len(self.elements))
        if batch_size == 0:
            raise ValueError("batch_size must be a positive number greater than zero.")

        elements = self.elements
        if self.total_samples_processed:
            elements = self.elements[self.total_samples_processed * batch_size :]

        batched = [
            elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
        ]
        return batched

    def get_num_workers(self, batched):
        return min(self.num_workers, len(batched) // max(self.num_workers, 1))

    def get_checkpoint_interval_and_batched_data(self, batched, num_workers):
        checkpoint_interval = max(
            int(
                (0.1 * len(batched) // max(num_workers, 1)) * max(num_workers, 1),
            ),
            num_workers,
            1,
        )

        if checkpoint_interval * self.ingestion_batch_size > MAX_CHECKPOINTING_INTERVAL:
            checkpoint_interval = 100

        return checkpoint_interval

    def run(self):
        if (
            len(self.elements) < MAX_DATASET_LENGTH_FOR_CACHING
            and self.embedding_function
        ):
            full_text = [element["text"] for element in self.elements]
            embeddings = self.embedding_function(full_text)

            self.elements = [
                {
                    "text": element["text"],
                    "id": element["id"],
                    "metadata": element["metadata"],
                    "embedding": embeddings[i],
                }
                for i, element in enumerate(self.elements)
            ]
            self.embedding_function = None

        batched_data = self.collect_batched_data()
        num_workers = self.get_num_workers(batched_data)
        checkpoint_interval = self.get_checkpoint_interval_and_batched_data(
            batched_data, num_workers=num_workers
        )

        self._ingest(
            batched=batched_data,
            num_workers=num_workers,
            checkpoint_interval=checkpoint_interval,
        )

    def _ingest(
        self,
        batched,
        num_workers,
        checkpoint_interval,
    ):
        try:
            ingest(embedding_function=self.embedding_function).eval(
                batched,
                self.dataset,
                num_workers=num_workers,
                checkpoint_interval=checkpoint_interval,
            )
        except Exception as e:
            self.retry_attempt += 1
            last_checkpoint = self.dataset.version_state["commit_node"].parent
            self.total_samples_processed += last_checkpoint.total_samples_processed

            if self.retry_attempt > MAX_VECTORSTORE_INGESTION_RETRY_ATTEMPTS:
                raise FailedIngestionError(
                    f"Maximum retry attempts exceeded. You can resume ingestion, from the latest saved checkpoint.\n"
                    "To do that you should run:\n"
                    "```\n"
                    "deeplake_vector_store.add(\n"
                    "    texts=texts,\n"
                    "    metadatas=metadatas,\n"
                    "    ids=ids,\n"
                    "    embeddings=embeddings,\n"
                    f"    total_samples_processed={self.total_samples_processed},\n"
                    ")\n"
                    "```"
                )

            data_ingestion = DataIngestion(
                elements=self.elements,
                dataset=self.dataset,
                embedding_function=self.embedding_function,
                ingestion_batch_size=self.ingestion_batch_size,
                num_workers=num_workers,
                retry_attempt=self.retry_attempt,
                total_samples_processed=self.total_samples_processed,
            )
            data_ingestion.run()


@deeplake.compute
def ingest(sample_in: list, sample_out: list, embedding_function) -> None:
    text_list = [s["text"] for s in sample_in]

    embeds: List[Optional[np.ndarray]] = [None] * len(text_list)
    if embedding_function is not None:
        try:
            embeddings = embedding_function(text_list)
        except Exception as e:
            raise Exception(
                "Could not use embedding function. Please try again with a different embedding function."
            )
        embeds = [np.array(e, dtype=np.float32) for e in embeddings]

    for s, emb in zip(sample_in, embeds):
        embedding = emb if embedding_function else s["embedding"]
        sample_out.append(
            {
                "text": s["text"],
                "metadata": s["metadata"],
                "ids": s["id"],
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )
