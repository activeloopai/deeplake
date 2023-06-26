import uuid
from typing import List, Dict, Any

import numpy as np
from math import ceil

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import deeplake
from deeplake.constants import MB
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search.ingestion import ingest_data
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
    VECTORSTORE_EXTEND_MAX_SIZE,
    DEFAULT_VECTORSTORE_TENSORS,
    VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE,
)


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
    **kwargs,
):
    utils.check_indra_installation(
        exec_option=exec_option, indra_installed=_INDRA_INSTALLED
    )

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
        **kwargs,
    )


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(
    dataset_path,
    token,
    creds,
    logger,
    read_only,
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
        **kwargs,
    )

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
    **kwargs,
):
    runtime = None
    if exec_option == "tensor_db":
        runtime = {"tensor_db": True}

    dataset = deeplake.empty(
        dataset_path,
        token=token,
        runtime=runtime,
        verbose=False,
        overwrite=overwrite,
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
    if (
        embedding is None
        and embedding_function is not None
        and embedding_data is not None
    ):
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
        id = [str(uuid.uuid1()) for _ in range(num_items)]
        tensors[ids_tensor] = id
    else:
        for tensor in not_none_tensors:
            if tensor in ("id", "ids"):
                break

        tensors[ids_tensor] = not_none_tensors[tensor]
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
            f"No embedding tensors were found, so the embedding function metadata will not be added to any tensor. "
            "Consider doing that manually using `vector_store.dataset.tensor_name.info. = <embedding_function_info_dictionary>`"
        )
        return
    if num_embedding_tensors > 1:
        logger.warning(
            f"{num_embedding_tensors} embedding tensors were found. "
            f"It is not clear to which tensor the embedding function information should be added, so the embedding function metadata will not be added to any tensor. "
            "Consider doing that manually using `vector_store.dataset.tensor_name.info = <embedding_function_info_dictionary>`"
        )
        return

    set_embedding_info(dataset[embeddings_tensors[0]], embedding_function)


def extend_or_ingest_dataset(
    processed_tensors,
    dataset,
    embedding_function,
    embedding_tensor,
    embedding_data,
    ingestion_batch_size,
    num_workers,
    total_samples_processed,
    logger,
):
    first_item = next(iter(processed_tensors))

    htypes = [
        dataset[item].meta.htype for item in dataset.tensors
    ]  # Inspect raw htype (not parsed htype like tensor.htype) in order to avoid parsing links and sequences separately.
    threshold_by_htype = [
        VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE.get(h, int(1e10)) for h in htypes
    ]
    extend_threshold = min(threshold_by_htype + [VECTORSTORE_EXTEND_MAX_SIZE])

    if len(processed_tensors[first_item]) <= extend_threshold:
        if embedding_function:
            for func, data, tensor in zip(
                embedding_function, embedding_data, embedding_tensor
            ):
                embedded_data = func(data)
                embedded_data = np.array(embedded_data, dtype=np.float32)
                if len(embedded_data) == 0:
                    raise ValueError("embedding function returned empty list")

                processed_tensors[tensor] = embedded_data

        dataset.extend(processed_tensors)
    else:
        elements = dataset_utils.create_elements(processed_tensors)

        num_workers_auto = ceil(len(elements) / ingestion_batch_size)
        if num_workers_auto < num_workers:
            logger.warning(
                f"Number of workers is {num_workers}, but {len(elements)} rows of data are being added and the ingestion_batch_size is {ingestion_batch_size}. "
                f"Setting the number of workers to {num_workers_auto} instead, in order reduce overhead from excessive workers that will not accelerate ingestion."
                f"If you want to parallelizing using more workers, please reduce ``ingestion_batch_size``."
            )
            num_workers = min(num_workers_auto, num_workers)

        ingest_data.run_data_ingestion(
            elements=elements,
            dataset=dataset,
            embedding_function=embedding_function,
            embedding_tensor=embedding_tensor,
            ingestion_batch_size=ingestion_batch_size,
            num_workers=num_workers,
            total_samples_processed=total_samples_processed,
            logger=logger,
        )


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


def check_delete_arguments(ids, filter, query, delete_all, row_ids, exec_option):
    if (
        ids is None
        and filter is None
        and query is None
        and delete_all is None
        and row_ids is None
    ):
        raise ValueError(
            "Either ids, row_ids, filter, query, or delete_all must be specified."
        )
    if exec_option not in ("python", "compute_engine", "tensor_db"):
        raise ValueError(
            "Invalid `exec_option` it should be either `python`, `compute_engine`."
        )
