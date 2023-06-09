import uuid
from typing import List, Dict, Any

import numpy as np

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
)
from deeplake.util.warnings import always_warn


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
    if embedding is None and embedding_function is not None:
        embedding = embedding_function(embedding_data)  # type: ignore

    if embedding is not None and (
        isinstance(embedding, list) or embedding.dtype != "float32"
    ):
        embedding = np.array(embedding, dtype=np.float32)

    return embedding


def preprocess_tensors(
    embedding_data=None, embedding_tensor=None, dataset=None, **tensors
):
    first_item = next(iter(tensors))
    ids_tensor = "ids" if "ids" in tensors else "id"
    if ids_tensor not in tensors or ids_tensor is None:
        id = [str(uuid.uuid1()) for _ in tensors[first_item]]
        tensors[ids_tensor] = id

    processed_tensors = {ids_tensor: tensors[ids_tensor]}

    for tensor_name, tensor_data in tensors.items():
        if not isinstance(tensor_data, list):
            tensor_data = list(tensor_data)
        if dataset and dataset[tensor_name].htype == "image":
            tensor_data = [
                deeplake.read(data) if isinstance(data, str) else data
                for data in tensor_data
            ]
        processed_tensors[tensor_name] = tensor_data

    if embedding_data:
        processed_tensors[embedding_tensor] = embedding_data

    return processed_tensors, tensors[ids_tensor]


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
    if len(processed_tensors[first_item]) <= VECTORSTORE_EXTEND_MAX_SIZE:
        if embedding_function:
            embedded_data = embedding_function(embedding_data)
            embedded_data = np.array(embedded_data, dtype=np.float32)
            processed_tensors[embedding_tensor] = embedded_data

        dataset.extend(processed_tensors)
    else:
        elements = dataset_utils.create_elements(processed_tensors)

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
