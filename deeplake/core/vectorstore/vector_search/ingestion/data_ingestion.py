from typing import Dict, List, Any, Callable, Optional, Union

import numpy as np

import deeplake
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search import utils
from deeplake.util.exceptions import (
    TransformError,
    FailedIngestionError,
    IncorrectEmbeddingShapeError,
)
from deeplake.constants import (
    MAX_VECTORSTORE_INGESTION_RETRY_ATTEMPTS,
    MAX_CHECKPOINTING_INTERVAL,
)
import sys
from deeplake.constants import MAX_BYTES_PER_MINUTE, TARGET_BYTE_SIZE


class DataIngestion:
    def __init__(
        self,
        elements: List[Dict[str, Any]],
        dataset: DeepLakeDataset,
        embedding_function: Optional[List[Callable]],
        embedding_tensor: Optional[List[str]],
        ingestion_batch_size: int,
        num_workers: int,
        retry_attempt: int,
        total_samples_processed: int,
        logger,
    ):
        self.elements = elements
        self.dataset = dataset
        self.embedding_function = embedding_function
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        self.retry_attempt = retry_attempt
        self.total_samples_processed = total_samples_processed
        self.embedding_tensor = embedding_tensor
        self.logger = logger

    def collect_batched_data(self, ingestion_batch_size=None):
        ingestion_batch_size = ingestion_batch_size or self.ingestion_batch_size
        batch_size = min(ingestion_batch_size, len(self.elements))
        if batch_size == 0:
            raise ValueError("batch_size must be a positive number greater than zero.")

        elements = self.elements
        if self.total_samples_processed:
            elements = self.elements[self.total_samples_processed :]

        batched = [
            elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
        ]

        if self.logger:
            batch_upload_str = f"Batch upload: {len(elements)} samples are being uploaded in {len(batched)} batches of batch size {batch_size}"
            if self.total_samples_processed:
                batch_upload_str = (
                    f"Batch reupload: {len(self.elements)-len(elements)} samples already uploaded, while "
                    f"{len(elements)} samples are being uploaded in {len(batched)} batches of batch size {batch_size}"
                )
            self.logger.warning(batch_upload_str)
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
            ingest(
                embedding_function=self.embedding_function,
                embedding_tensor=self.embedding_tensor,
            ).eval(
                batched,
                self.dataset,
                num_workers=num_workers,
                checkpoint_interval=checkpoint_interval,
                verbose=False,
            )
        except Exception as e:
            if isinstance(e.__cause__, IncorrectEmbeddingShapeError):
                raise IncorrectEmbeddingShapeError()

            self.retry_attempt += 1
            last_checkpoint = self.dataset.version_state["commit_node"].parent
            self.total_samples_processed += (
                last_checkpoint.total_samples_processed * self.ingestion_batch_size
            )

            index = int(self.total_samples_processed / self.ingestion_batch_size)
            if isinstance(e, TransformError) and e.index is not None:
                index += e.index

            if self.retry_attempt > MAX_VECTORSTORE_INGESTION_RETRY_ATTEMPTS:
                raise FailedIngestionError(
                    f"Ingestion failed at batch index {index}. Maximum retry attempts exceeded. You can resume ingestion "
                    "from the latest saved checkpoint.\n"
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
                logger=self.logger,
                embedding_tensor=self.embedding_tensor,
            )
            data_ingestion.run()
