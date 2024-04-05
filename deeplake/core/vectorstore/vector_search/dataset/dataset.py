import uuid
import sys
import time
from math import ceil
from typing import List, Dict, Any, Optional, Callable, Union
from tqdm import tqdm

import numpy as np

import deeplake
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search.ingestion import ingest_data
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
    VECTORSTORE_EXTEND_MAX_SIZE,
    DEFAULT_VECTORSTORE_TENSORS,
    VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE,
    MAX_BYTES_PER_MINUTE,
    TARGET_BYTE_SIZE,
    VECTORSTORE_EXTEND_BATCH_SIZE,
    DEFAULT_RATE_LIMITER_KEY_TO_VALUE,
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
    branch="main",
    **kwargs,
):
    utils.check_indra_installation(exec_option=exec_option)

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
            org_id,
            branch,
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
        branch,
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
    org_id,
    branch,
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
    dataset.checkout(branch)
    check_tensors(dataset)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
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
    branch,
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
    dataset.checkout(branch)
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


def delete_and_without_commit(dataset, ids, index_maintenance):
    with dataset:
        for id in sorted(ids)[::-1]:
            dataset.pop(id, index_maintenance=index_maintenance)


def delete_all_samples_if_specified(dataset, delete_all):
    if delete_all:
        # delete any indexes linked to any tensors.
        for t in dataset.tensors:
            dataset[t]._verify_and_delete_vdb_indexes()

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
    if (
        embedding is None
        and embedding_function is not None
        and embedding_data is not None
    ):
        if isinstance(embedding_data, list):
            if len(embedding_data) > 1:
                raise NotImplementedError("Batched quering is not supported yet.")
            elif len(embedding_data) == 0:
                raise ValueError("embedding_data must not be empty.")
            else:
                embedding_data = embedding_data[0]

        if not isinstance(embedding_data, str):
            raise ValueError("embedding_data must be a string.")

        embedding = embedding_function.embed_query(embedding_data)  # type: ignore

    if embedding is not None and (
        isinstance(embedding, list) or embedding.dtype != "float32"
    ):
        embedding = np.array(embedding, dtype=np.float32)

    if isinstance(embedding, np.ndarray):
        assert (
            embedding.ndim == 1 or embedding.shape[0] == 1
        ), "Query embedding must be 1-dimensional. Please consider using another embedding function for converting query string to embedding."

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


def _compute_batched_embeddings(
    embedding_function,
    embedding_data,
    embedding_tensor,
    start_idx,
    end_idx,
    rate_limiter,
):
    """
    Computes embeddings for a given slice of data.
    """
    batched_processed_tensors = {}

    for func, data, tensor in zip(embedding_function, embedding_data, embedding_tensor):
        data_slice = data[start_idx:end_idx]
        embedded_data = func(data_slice, rate_limiter=rate_limiter)

        try:
            return_embedded_data = np.vstack(embedded_data).astype(dtype=np.float32)
        except ValueError:
            raise IncorrectEmbeddingShapeError()

        if len(return_embedded_data) == 0:
            raise ValueError("embedding function returned empty list")

        batched_processed_tensors[tensor] = return_embedded_data

    return batched_processed_tensors


def _slice_non_embedding_tensors(
    processed_tensors, embedding_tensor, start_idx, end_idx
):
    """
    Slices tensors that are not embeddings for a given range.
    """
    batched_processed_tensors = {}

    for tensor_name, tensor_data in processed_tensors.items():
        if tensor_name not in embedding_tensor:
            batched_processed_tensors[tensor_name] = tensor_data[start_idx:end_idx]

    return batched_processed_tensors


def extend(
    embedding_function: List[Callable],
    embedding_data: List[Any],
    embedding_tensor: Union[str, List[str]],
    processed_tensors: Dict[str, Union[List[Any], np.ndarray]],
    dataset: deeplake.core.dataset.Dataset,
    rate_limiter: Dict,
    _extend_batch_size: int = VECTORSTORE_EXTEND_BATCH_SIZE,
    logger=None,
):
    """
    Function to extend the dataset with new data.
    """
    if embedding_data and not isinstance(embedding_data[0], list):
        embedding_data = [embedding_data]

    if embedding_function:
        number_of_batches = ceil(len(embedding_data[0]) / _extend_batch_size)
        progressbar_str = (
            f"Creating {len(embedding_data[0])} embeddings in "
            f"{number_of_batches} batches of size {min(_extend_batch_size, len(embedding_data[0]))}:"
        )

        for idx in tqdm(
            range(0, len(embedding_data[0]), _extend_batch_size),
            progressbar_str,
        ):
            batch_start, batch_end = idx, idx + _extend_batch_size

            batched_embeddings = _compute_batched_embeddings(
                embedding_function,
                embedding_data,
                embedding_tensor,
                batch_start,
                batch_end,
                rate_limiter,
            )

            batched_tensors = _slice_non_embedding_tensors(
                processed_tensors, embedding_tensor, batch_start, batch_end
            )

            batched_processed_tensors = {**batched_embeddings, **batched_tensors}

            dataset.extend(batched_processed_tensors, progressbar=False)
    else:
        logger.info("Uploading data to deeplake dataset.")
        dataset.extend(processed_tensors, progressbar=True)


def populate_rate_limiter(rate_limiter):
    if rate_limiter is None or rate_limiter == {}:
        return {
            "enabled": False,
            "bytes_per_minute": MAX_BYTES_PER_MINUTE,
            "batch_byte_size": TARGET_BYTE_SIZE,
        }
    else:
        rate_limiter_keys = ["enabled", "bytes_per_minute", "batch_byte_size"]

        for key in rate_limiter_keys:
            if key not in rate_limiter:
                rate_limiter[key] = DEFAULT_RATE_LIMITER_KEY_TO_VALUE[key]

        for item in rate_limiter:
            if item not in rate_limiter_keys:
                raise ValueError(
                    f"Invalid rate_limiter key: {item}. Valid keys are: 'enabled', 'bytes_per_minute', 'batch_byte_size'."
                )
        return rate_limiter


def extend_or_ingest_dataset(
    processed_tensors,
    dataset,
    embedding_function,
    embedding_tensor,
    embedding_data,
    rate_limiter,
    logger,
):
    rate_limiter = populate_rate_limiter(rate_limiter)
    # TODO: Add back the old logic with checkpointing after indexing is fixed
    extend(
        embedding_function,
        embedding_data,
        embedding_tensor,
        processed_tensors,
        dataset,
        rate_limiter,
        logger=logger,
    )


def convert_id_to_row_id(ids, dataset, search_fn, query, exec_option, filter):
    if ids is None:
        delete_view = search_fn(
            embedding_data=None,
            embedding_function=None,
            embedding=None,
            distance_metric=None,
            embedding_tensor=None,
            filter=filter,
            query=query,
            exec_option=exec_option,
            return_tensors=False,
            return_view=True,
            k=int(1e9),
            deep_memory=False,
            return_tql=False,
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
