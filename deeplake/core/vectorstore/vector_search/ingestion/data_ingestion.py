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
from deeplake.constants import MAX_BYTES_PER_MINUTE, TARGET_BATCH_SIZE


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

    def collect_batched_data(self, ingestion_byte_size=10000):
        elements = self.elements
        if self.total_samples_processed:
            elements = self.elements[self.total_samples_processed :]

        current_batch = []
        current_byte_size = 0
        batched = []
        cumulative_length_to_idx = {0: 0}
        cum_length = 0
        for idx, element in enumerate(elements):
            element_size = sys.getsizeof(element)
            
            if current_byte_size + element_size > ingestion_byte_size:
                batched.append(current_batch)
                current_batch = []
                current_byte_size = 0
            
            current_batch.append(element)
            current_byte_size += element_size
            
            cum_length += len(current_batch)
            cumulative_length_to_idx[cum_length] = idx+1

        if current_batch:
            batched.append(current_batch)
        return batched, cumulative_length_to_idx

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
        batched_data, cumulative_length_to_idx = self.collect_batched_data()
        num_workers = self.get_num_workers(batched_data)
        checkpoint_interval = self.get_checkpoint_interval_and_batched_data(
            batched_data, num_workers=num_workers
        )

        self._ingest(
            batched=batched_data,
            num_workers=num_workers,
            checkpoint_interval=checkpoint_interval,
            cumulative_length_to_idx=cumulative_length_to_idx,
        )

    def _ingest(
        self,
        batched,
        num_workers,
        checkpoint_interval,
        cumulative_length_to_idx,
    ):
        # Calculate the number of batches you can send each minute
        avg_batch_size = get_size_of_list_strings(batched) / len(batched)
        batches_per_minute = MAX_BYTES_PER_MINUTE / avg_batch_size
        try:
            ingest(
                embedding_function=self.embedding_function,
                embedding_tensor=self.embedding_tensor,
            ).eval(
                batched,
                self.dataset,
                num_workers=num_workers,
                checkpoint_interval=checkpoint_interval,
                requests_per_minute=batches_per_minute,
                verbose=False,
            )
        except Exception as e:
            if isinstance(e.__cause__, IncorrectEmbeddingShapeError):
                raise IncorrectEmbeddingShapeError()

            self.retry_attempt += 1
            last_checkpoint = self.dataset.version_state["commit_node"].parent
            current_idx = cumulative_length_to_idx[self.total_samples_processed]
            slice_length = compute_length(
                batched, 
                start_idx=current_idx, 
                # end_idx=last_checkpoint.total_samples_processed or 0
                end_idx=0,
            )
            self.total_samples_processed += slice_length

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


def compute_length(data: List[List[str]], start_idx: int, end_idx: int):
    return sum([len(d) for d in data[start_idx:end_idx]])


@deeplake.compute
def ingest(
    sample_in: list,
    sample_out: list,
    embedding_function,
    embedding_tensor,
) -> None:
    embeds: List[Optional[np.ndarray]] = [None] * len(sample_in)
    if embedding_function:
        try:
            for func, tensor in zip(embedding_function, embedding_tensor):
                embedding_data = [s[tensor] for s in sample_in]
                embeddings = func(embedding_data)
        except Exception as exc:
            raise Exception(
                "Could not use embedding function. Please try again with a different embedding function."
            )

        shape = np.array(embeddings[0]).shape
        embeds = []
        for e in embeddings:
            embedding = np.array(e, dtype=np.float32)
            if shape != embedding.shape:
                raise IncorrectEmbeddingShapeError()
            embeds.append(embedding)

    for s, emb in zip(sample_in, embeds):
        sample_in_i = {tensor_name: s[tensor_name] for tensor_name in s}

        if embedding_function:
            for tensor in embedding_tensor:
                sample_in_i[tensor] = np.array(emb, dtype=np.float32)

        sample_out.append(sample_in_i)


import sys

def get_size_of_list_strings(lst):
    total_size = sys.getsizeof(lst)  # size of the list itself

    for s in lst:
        total_size += sys.getsizeof(s)  # size of each string

    return total_size