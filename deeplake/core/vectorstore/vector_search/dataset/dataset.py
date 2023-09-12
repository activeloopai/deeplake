import uuid
import sys
import time
from math import ceil
from typing import List, Dict, Any, Optional, Callable, Union
from tqdm import tqdm

import numpy as np

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import deeplake
from deeplake.util.path import get_path_type
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search.ingestion import ingest_data
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
    VECTORSTORE_EXTEND_MAX_SIZE,
    DEFAULT_VECTORSTORE_TENSORS,
    VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE,
    MAX_BYTES_PER_MINUTE,
    TARGET_BYTE_SIZE,
)
from deeplake.util.exceptions import IncorrectEmbeddingShapeError


def create_or_load_dataset(
    tensor_params,
    dataset_path,
    token,
    creds,
    logger,
    read_only,
    exec_option,
    embedding_function,
    overwrite,
    runtime,
    org_id,
    **kwargs,
):
    utils.check_indra_installation(
        exec_option=exec_option, indra_installed=_INDRA_INSTALLED
    )
    org_id = org_id if get_path_type(dataset_path) == "local" else None

    if not overwrite and dataset_exists(dataset_path, token, creds, **kwargs):
        if tensor_params is not None and tensor_params != DEFAULT_VECTORSTORE_TENSORS:
            raise ValueError(
                "Vector Store is not empty. You shouldn't specify tensor_params if you're loading from existing dataset."
            )

        return load_dataset(
            dataset_path,
            token,
            creds,
            logger,
            read_only,
            runtime,
            org_id,
            **kwargs,
        )

    return create_dataset(
        logger,
        tensor_params,
        dataset_path,
        token,
        exec_option,
        embedding_function,
        overwrite,
        creds,
        runtime,
        org_id,
        **kwargs,
    )


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, creds=creds)
        and "overwrite" not in kwargs
    )


def load_dataset(
    dataset_path,
    token,
    creds,
    logger,
    read_only,
    runtime,
    org_id,
    **kwargs,
):
    if dataset_path == DEFAULT_VECTORSTORE_DEEPLAKE_PATH:
        logger.warning(
            f"The default deeplake path location is used: {DEFAULT_VECTORSTORE_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset."
        )
    dataset = deeplake.load(
        dataset_path,
        token=token,
        read_only=read_only,
        creds=creds,
        verbose=False,
        org_id=org_id,
        **kwargs,
    )
    check_tensors(dataset)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )

    if runtime is not None and runtime["tensor_db"] == True:
        logger.warning(
            "Specifying runtime option when loading a Vector Store is not supported and this parameter will "
            "be ignored. If you wanted to create a new Vector Store, please specify a path to a Vector Store "
            "that does not already exist. To transfer an existing Vector Store to the Managed Tensor Database, "
            "use the steps in the link below: "
            "(https://docs.activeloop.ai/enterprise-features/managed-database/migrating-datasets-to-the-tensor-database)."
        )

    return dataset


def check_tensors(dataset):
    tensors = dataset.tensors

    embedding_tensor_exist = False
    ids_exist = False

    for tensor in tensors:
        htype = dataset[tensor].htype

        if tensor in ("id", "ids"):
            ids_exist = True

        if tensor in ("embedding", "embeddings"):
            embedding_tensor_exist = True

            # TODO: Add back once old datasets without embedding htype are not in circulation
            # if htype not in (None, "embedding"):
            #     raise ValueError(
            #         f"`{htype}` is not supported htype for embedding tensor. "
            #         "Supported htype for embedding tensor is: `embedding`"
            #     )

        if htype == "embedding":
            if tensor in ("id", "ids"):
                raise ValueError(
                    f"`{tensor}` is not valid name for embedding tensor, as the name is preserved for another tensor"
                )

            embedding_tensor_exist = True

    if not embedding_tensor_exist:
        raise ValueError("At least one embedding tensor should exist.")

    if not ids_exist:
        raise ValueError("`id` tensor was not found in the dataset.")


def create_dataset(
    logger,
    tensor_params,
    dataset_path,
    token,
    exec_option,
    embedding_function,
    overwrite,
    creds,
    runtime,
    org_id,
    **kwargs,
):
    if exec_option == "tensor_db" and (
        runtime is None or runtime == {"tensor_db": False}
    ):
        raise ValueError(
            "To execute queries using exec_option = 'tensor_db', "
            "the Vector Store must be stored in Deep Lake's Managed "
            "Tensor Database. To create the Vector Store in the Managed "
            "Tensor Database, specify runtime = {'tensor_db': True} when "
            "creating the Vector Store."
        )

    dataset = deeplake.empty(
        dataset_path,
        token=token,
        runtime=runtime,
        verbose=False,
        overwrite=overwrite,
        creds=creds,
        org_id=org_id,
        **kwargs,
    )
    create_tensors(tensor_params, dataset, logger, embedding_function)

    return dataset


def create_tensors(tensor_params, dataset, logger, embedding_function):
    tensor_names = [tensor["name"] for tensor in tensor_params]
    if "id" not in tensor_names and "ids" not in tensor_names:
        tensor_params.append(
            {
                "name": "id",
                "htype": "text",
                "create_id_tensor": False,
                "create_sample_info_tensor": False,
                "create_shape_tensor": False,
                "chunk_compression": "lz4",
            },
        )

    with dataset:
        for tensor_args in tensor_params:
            dataset.create_tensor(**tensor_args)

        update_embedding_info(logger, dataset, embedding_function)


def delete_and_commit(dataset, ids):
    with dataset:
        for id in sorted(ids)[::-1]:
            dataset.pop(id)
        dataset.commit(f"deleted {len(ids)} samples", allow_empty=True)
    return True


def delete_all_samples_if_specified(dataset, delete_all):
    if delete_all:
        dataset = deeplake.like(
            dataset.path,
            dataset,
            overwrite=True,
            verbose=False,
        )

        return dataset, True
    return dataset, False


def fetch_embeddings(view, embedding_tensor: str = "embedding"):
    return view[embedding_tensor].numpy()


def get_embedding(embedding, embedding_data, embedding_function=None):
    if isinstance(embedding_data, str):
        embedding_data = [embedding_data]

    if (
        embedding is None
        and embedding_function is not None
        and embedding_data is not None
    ):
        if len(embedding_data) > 1:
            raise NotImplementedError("Searching batched queries is not supported yet.")

        embedding = embedding_function(embedding_data)  # type: ignore

    if embedding is not None and (
        isinstance(embedding, list) or embedding.dtype != "float32"
    ):
        embedding = np.array(embedding, dtype=np.float32)

    return embedding


def preprocess_tensors(
    embedding_data=None, embedding_tensor=None, dataset=None, **tensors
):
    # generate id list equal to the length of the tensors
    # dont use None tensors to get length of tensor
    not_none_tensors, num_items = get_not_none_tensors(tensors, embedding_data)
    ids_tensor = get_id_tensor(dataset)
    tensors = populate_id_tensor_if_needed(
        ids_tensor, tensors, not_none_tensors, num_items
    )

    processed_tensors = {ids_tensor: tensors[ids_tensor]}

    for tensor_name, tensor_data in tensors.items():
        tensor_data = convert_tensor_data_to_list(tensor_data, tensors, ids_tensor)
        tensor_data = read_tensor_data_if_needed(tensor_data, dataset, tensor_name)
        processed_tensors[tensor_name] = tensor_data

    if embedding_data:
        for k, v in zip(embedding_tensor, embedding_data):
            processed_tensors[k] = v

    return processed_tensors, tensors[ids_tensor]


def read_tensor_data_if_needed(tensor_data, dataset, tensor_name):
    # generalize this method for other htypes that need reading.
    if dataset and tensor_name != "id" and dataset[tensor_name].htype == "image":
        tensor_data = [
            deeplake.read(data) if isinstance(data, str) else data
            for data in tensor_data
        ]
    return tensor_data


def convert_tensor_data_to_list(tensor_data, tensors, ids_tensor):
    if tensor_data is None:
        tensor_data = [None] * len(tensors[ids_tensor])
    elif not isinstance(tensor_data, list):
        tensor_data = list(tensor_data)
    return tensor_data


def get_not_none_tensors(tensors, embedding_data):
    not_none_tensors = {k: v for k, v in tensors.items() if v is not None}
    try:
        num_items = len(next(iter(not_none_tensors.values())))
    except StopIteration:
        if embedding_data:
            num_items = len(embedding_data[0])
        else:
            num_items = 0
    return not_none_tensors, num_items


def populate_id_tensor_if_needed(ids_tensor, tensors, not_none_tensors, num_items):
    if "id" not in not_none_tensors and "ids" not in not_none_tensors:
        found_id = [str(uuid.uuid1()) for _ in range(num_items)]
        tensors[ids_tensor] = found_id
    else:
        for tensor in not_none_tensors:
            if tensor in ("id", "ids"):
                break

        tensors[ids_tensor] = list(
            map(
                lambda x: str(x) if isinstance(x, uuid.UUID) else x,
                not_none_tensors[tensor],
            )
        )
    return tensors


def get_id_tensor(dataset):
    return "ids" if "ids" in dataset.tensors else "id"


def create_elements(
    processed_tensors: Dict[str, List[Any]],
):
    tensor_names = list(processed_tensors)
    elements = [
        {tensor_name: processed_tensors[tensor_name][i] for tensor_name in tensor_names}
        for i in range(len(processed_tensors[tensor_names[0]]))
    ]
    return elements


def set_embedding_info(tensor, embedding_function):
    embedding_info = tensor.info.get("embedding")
    if embedding_function and not embedding_info:
        tensor.info["embedding"] = {
            "model": embedding_function.__dict__.get("model"),
            "deployment": embedding_function.__dict__.get("deployment"),
            "embedding_ctx_length": embedding_function.__dict__.get(
                "embedding_ctx_length"
            ),
            "chunk_size": embedding_function.__dict__.get("chunk_size"),
            "max_retries": embedding_function.__dict__.get("max_retries"),
        }


def update_embedding_info(logger, dataset, embedding_function):
    embeddings_tensors = utils.find_embedding_tensors(dataset)
    num_embedding_tensors = len(embeddings_tensors)

    if num_embedding_tensors == 0:
        logger.warning(
            "No embedding tensors were found, so the embedding function metadata will not be added to any tensor. "
            "Consider doing that manually using `vector_store.dataset.tensor_name.info. = <embedding_function_info_dictionary>`"
        )
        return
    if num_embedding_tensors > 1:
        logger.warning(
            f"{num_embedding_tensors} embedding tensors were found. "
            "It is not clear to which tensor the embedding function information should be added, so the embedding function metadata will not be added to any tensor. "
            "Consider doing that manually using `vector_store.dataset.tensor_name.info = <embedding_function_info_dictionary>`"
        )
        return

    set_embedding_info(dataset[embeddings_tensors[0]], embedding_function)


def extend(
    embedding_function: List[Callable],
    embedding_data: List[Any],
    embedding_tensor: Union[str, List[str]],
    processed_tensors: Dict[str, Union[List[Any], np.ndarray]],
    dataset: deeplake.core.dataset.Dataset,
    batch_byte_size: int,
    rate_limiter: Dict,
):
    """
    Function to extend the dataset with new data.

    Args:
        embedding_function (List[Callable]): List of embedding functions to be used to create embedding data.
        embedding_data (List[Any]): List of data to be embedded.
        embedding_tensor (Union[str, List[str]]): Name of the tensor(s) to store the embedding data.
        processed_tensors (Dict[str, List[Any]]): Dictionary of tensors to be added to the dataset.
        dataset (deeplake.core.dataset.Dataset): Dataset to be extended.
        batch_byte_size (int): Batch size to use for parallel ingestion.
        rate_limiter (Dict): Rate limiter configuration.

    Raises:
        IncorrectEmbeddingShapeError: If embeding function shapes is incorrect.
        ValueError: If embedding function returned empty list

    """
    if embedding_function:
        for func, data, tensor in zip(
            embedding_function, embedding_data, embedding_tensor
        ):
            data_iterator = data_iteratot_factory(
                data, func, batch_byte_size, rate_limiter
            )
            embedded_data = []

            for data in tqdm(
                data_iterator, total=len(data_iterator), desc="creating embeddings"
            ):
                embedded_data.append(data)

            try:
                return_embedded_data = np.vstack(embedded_data).astype(dtype=np.float32)
            except ValueError:
                raise IncorrectEmbeddingShapeError()

            if len(return_embedded_data) == 0:
                raise ValueError("embedding function returned empty list")

            processed_tensors[tensor] = return_embedded_data

    dataset.extend(processed_tensors, progressbar=True)


class DataIterator:
    def __init__(self, data, func, batch_byte_size):
        self.data = chunk_by_bytes(data, batch_byte_size)
        self.data_itr = iter(self.data)
        self.index = 0
        self.func = func

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        batch = next(self.data_itr)
        batch = self.func(batch)
        self.index += 1
        return batch

    def __len__(self):
        return len(self.data)


class RateLimitedDataIterator:
    def __init__(self, data, func, batch_byte_size, rate_limiter):
        self.data = chunk_by_bytes(data, batch_byte_size)
        self.data_iter = iter(self.data)
        self.index = 0
        self.rate_limiter = rate_limiter
        self.bytes_per_minute = rate_limiter["bytes_per_minute"]
        self.target_byte_size = batch_byte_size
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


def data_iteratot_factory(data, func, batch_byte_size, rate_limiter):
    if rate_limiter["enabled"]:
        return RateLimitedDataIterator(data, func, batch_byte_size, rate_limiter)
    else:
        return DataIterator(data, func, batch_byte_size)


def extend_or_ingest_dataset(
    processed_tensors,
    dataset,
    embedding_function,
    embedding_tensor,
    embedding_data,
    batch_byte_size,
    rate_limiter,
):
    # TODO: Add back the old logic with checkpointing after indexing is fixed
    extend(
        embedding_function,
        embedding_data,
        embedding_tensor,
        processed_tensors,
        dataset,
        batch_byte_size,
        rate_limiter,
    )


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


def convert_id_to_row_id(ids, dataset, search_fn, query, exec_option, filter):
    if ids is None:
        delete_view = search_fn(
            filter=filter,
            query=query,
            exec_option=exec_option,
            return_view=True,
            k=int(1e9),
        )

    else:
        # backwards compatibility
        tensors = dataset.tensors
        id_tensor = "id"
        if "ids" in tensors:
            id_tensor = "ids"

        delete_view = dataset.filter(lambda x: x[id_tensor].data()["value"] in ids)

    row_ids = list(delete_view.sample_indices)
    return row_ids


def check_arguments_compatibility(
    ids, filter, query, exec_option, select_all=None, row_ids=None
):
    if (
        ids is None
        and filter is None
        and query is None
        and row_ids is None
        and select_all is None
    ):
        raise ValueError(
            "Either ids, row_ids, filter, query, or select_all must be specified."
        )
    if exec_option not in ("python", "compute_engine", "tensor_db"):
        raise ValueError(
            "Invalid `exec_option` it should be either `python`, `compute_engine` or `tensor_db`."
        )


def search_row_ids(
    dataset: deeplake.core.dataset.Dataset,
    search_fn: Callable,
    ids: Optional[List[str]] = None,
    filter: Optional[Union[Dict, Callable]] = None,
    query: Optional[str] = None,
    exec_option: Optional[str] = "python",
    select_all: Optional[bool] = None,
):
    check_arguments_compatibility(
        ids=ids,
        filter=filter,
        query=query,
        select_all=select_all,
        exec_option=exec_option,
    )

    if select_all:
        return None

    row_ids = convert_id_to_row_id(
        ids=ids,
        dataset=dataset,
        search_fn=search_fn,
        query=query,
        exec_option=exec_option,
        filter=filter,
    )

    return row_ids
