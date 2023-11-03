import functools
import time
import types
from typing import Any, Optional, List, Dict, Callable

from deeplake.constants import TARGET_BYTE_SIZE, MAX_BYTES_PER_MINUTE


class RateLimitedDataIterator:
    def __init__(self, data, func, rate_limiter):
        self.data = chunk_by_bytes(data, rate_limiter["batch_byte_size"])
        self.data_iter = iter(self.data)
        self.index = 0
        self.bytes_per_minute = rate_limiter["bytes_per_minute"]
        self.target_byte_size = rate_limiter["batch_byte_size"]
        self.func = func

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        batch = next(self.data_iter)
        self.index += 1
        # Calculate the number of batches you can send each minute
        batches_per_minute = self.bytes_per_minute / self.target_byte_size

        # Calculate sleep time in seconds between batches
        sleep_time = 60 / batches_per_minute

        start = time.time()
        batch = self.func(batch)
        end = time.time()

        # we need to take into account the time spent on openai call
        diff = sleep_time - (end - start)
        if diff > 0:
            time.sleep(diff)
        return batch

    def __len__(self):
        return len(self.data)


class DeepLakeEmbedder:
    def __init__(
        self,
        embedding_function: Any,
    ):
        self.embedding_function = embedding_function

    def _get_embedding_func(self, default_func):
        valid_function_types = (
            types.MethodType,
            types.FunctionType,
            types.LambdaType,
            functools.partial,
        )
        if isinstance(self.embedding_function, valid_function_types):
            return self.embedding_function
        return getattr(self.embedding_function, default_func)

    def embed_documents(
        self,
        documents: List[str],
        rate_limiter: Dict = {
            "enabled": False,
            "bytes_per_minute": MAX_BYTES_PER_MINUTE,
            "batch_byte_size": TARGET_BYTE_SIZE,
        },
    ):
        embedding_func = self._get_embedding_func("embed_documents")

        if rate_limiter["enabled"]:
            return self._apply_rate_limiter(documents, embedding_func, rate_limiter)
        return embedding_func(documents)

    def embed_query(self, query: str):
        return self._get_embedding_func("embed_query")(query)

    @staticmethod
    def _apply_rate_limiter(documents, embedding_function, rate_limiter):
        data_iterator = RateLimitedDataIterator(
            documents,
            embedding_function,
            rate_limiter,
        )
        output = []
        for data in data_iterator:
            output.extend(data)
        return output


def chunk_by_bytes(data, target_byte_size=TARGET_BYTE_SIZE):
    """
    Splits a list of strings into chunks where each chunk has approximately the given target byte size.

    Args:
    - strings (list of str): List of strings to be chunked.
    - target_byte_size (int): The target byte size for each chunk.

    Returns:
    - list of lists containing the chunked strings.
    """
    # Calculate byte sizes for all strings
    sizes = [len(s.encode("utf-8")) for s in data]

    chunks = []
    current_chunk = []
    current_chunk_size = 0
    index = 0

    while index < len(data):
        if current_chunk_size + sizes[index] > target_byte_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_size = 0
        current_chunk.append(data[index])
        current_chunk_size += sizes[index]
        index += 1

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
