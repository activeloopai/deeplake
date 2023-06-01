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
    VECTORSTORE_INGESTION_THRESHOLD,
)
from deeplake.util.warnings import always_warn


def create_or_load_dataset(
    tensors_dict,
    dataset_path,
    token,
    creds,
    logger,
    read_only,
    exec_option,
    embedding_function,
    **kwargs,
):
    utils.check_indra_installation(
        exec_option=exec_option, indra_installed=_INDRA_INSTALLED
    )

    if "overwrite" in kwargs and kwargs["overwrite"] == False:
        del kwargs["overwrite"]

    if dataset_exists(dataset_path, token, creds, **kwargs):
        return load_dataset(
            tensors_dict,
            dataset_path,
            token,
            creds,
            logger,
            read_only,
            embedding_function,
            **kwargs,
        )

    return create_dataset(
        logger,
        tensors_dict,
        dataset_path,
        token,
        exec_option,
        embedding_function,
        **kwargs,
    )


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(
    tensors_dict,
    dataset_path,
    token,
    creds,
    logger,
    read_only,
    embedding_function,
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
    create_tensors_if_needed(tensors_dict, dataset, logger, embedding_function)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )
    return dataset


def create_tensors_if_needed(tensors_dict, dataset, logger, embedding_function):
    tensors = dataset.tensors

    for tensor_args in tensors_dict:
        if tensor_args["name"] not in tensors:
            warn_and_create_missing_tensor(dataset, logger, **tensor_args)
    update_embedding_info(logger, dataset, embedding_function)


def warn_and_create_missing_tensor(dataset, logger, **kwargs):
    logger.warning(
        f"Creating `{kwargs['name']}` tensor since it does not exist in the dataset. If you created dataset manually "
        "and stored text data in another tensor, consider copying the contents of that "
        f"tensor into `{kwargs['name']}` tensor and deleting if afterwards. To view dataset content "
        "run ds.summary()"
    )

    dataset.create_tensor(
        **kwargs,
    )


def create_dataset(
    logger, tensors_dict, dataset_path, token, exec_option, embedding_function, **kwargs
):
    runtime = None
    if exec_option == "tensor_db":
        runtime = {"tensor_db": True}

    dataset = deeplake.empty(
        dataset_path, token=token, runtime=runtime, verbose=False, **kwargs
    )
    create_tensors(tensors_dict, dataset, logger, embedding_function)

    return dataset


def create_tensors(tensors_dict, dataset, logger, embedding_function):
    tensor_names = [tensor["name"] for tensor in tensors_dict]
    if "id" not in tensor_names:
        tensors_dict.append(
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
        for tensor_args in tensors_dict:
            dataset.create_tensor(**tensor_args)

        update_embedding_info(logger, dataset, embedding_function)


def delete_and_commit(dataset, ids):
    with dataset:
        for id in sorted(ids)[::-1]:
            dataset.pop(id)
        dataset.commit(f"deleted {len(ids)} samples", allow_empty=True)


def delete_all_samples_if_specified(dataset, delete_all, vector_store):
    if delete_all:
        dataset = deeplake.like(
            vector_store.dataset.path,
            vector_store.dataset,
            overwrite=True,
            verbose=False,
        )

        return dataset, True
    return dataset, False


def fetch_embeddings(exec_option, view, logger, embedding_tensor: str = "embedding"):
    try:
        return view[embedding_tensor].numpy()
    except Exception:
        raise ValueError(
            "Could not find embedding tensor. If you're using non-default tensor_dict, "
            "please specify `embedding_tensor` that you want to use. "
            "Ex: vector_store.search(embedding=query_embedding, embedding_tensor='your_embedding_tensor')"
        )


def get_embedding(embedding, data_for_embedding, embedding_function=None):
    if embedding_function is not None:
        if embedding is not None:
            always_warn(
                "both embedding and embedding_function are specified. The embedding will be ignored."
            )

        if data_for_embedding is None:
            raise ValueError(
                "data_for_embedding is not specified. Please specify data_for_embedding wheverer embedding_function is specified."
            )

        embedding = embedding_function(data_for_embedding)  # type: ignore

    if embedding is not None and (
        isinstance(embedding, list) or embedding.dtype != "float32"
    ):
        embedding = np.array(embedding, dtype=np.float32)

    return embedding


def preprocess_tensors(**tensors):
    first_item = next(iter(tensors))

    if "id" not in tensors or tensors["id"] is None:
        id = [str(uuid.uuid1()) for _ in tensors[first_item]]
        tensors["id"] = id

    processed_tensors = {"id": tensors["id"]}

    for tensor, tensor_array in tensors.items():
        if not isinstance(tensor_array, list):
            tensor_array = list(tensor_array)
        processed_tensors[tensor] = tensor_array

    return processed_tensors, tensors["id"]


def create_elements(
    processed_tensors: Dict[str, List[Any]],
):
    tensor_names = list(processed_tensors.keys())
    elements = [
        {tensor_name: processed_tensors[tensor_name][i] for tensor_name in tensor_names}
        for i in range(len(processed_tensors[tensor_names[0]]))
    ]
    return elements


def fetch_tensor_based_on_htype(logger, dataset, htype):
    tensors = dataset.tensors

    if "embedding" in tensors:
        return dataset.embedding

    num_of_tensors_with_htype = 0

    tensor_names = []
    for tensor in tensors:
        if dataset[tensor].htype == "embedding":
            num_of_tensors_with_htype += 1
            tensor_names.append(tensor)

    tensor_names_str = "".join(f"`{tensor_name}`, " for tensor_name in tensor_names)
    tensor_names_str = tensor_names_str[:-2]

    if num_of_tensors_with_htype > 1:
        logger.warning(
            f"{num_of_tensors_with_htype} tensors with `embedding` htype were found. "
            f"They are: {tensor_names_str}. Embedding function info will be appended to "
            f"`{tensor_names[0]}`. If you want to update other embedding tensor's information "
            "consider doing that manually. Example: `dataset.tensor['info'] = info_dictionary`"
        )

    return dataset[tensor_names[0]]


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
    tensor = fetch_tensor_based_on_htype(logger, dataset, embedding_function)
    set_embedding_info(tensor, embedding_function)


def extend_or_ingest_dataset(
    processed_tensors,
    dataset,
    embedding_function,
    embed_data_to,
    embed_data_from,
    ingestion_batch_size,
    num_workers,
    total_samples_processed,
):
    if len(processed_tensors) <= VECTORSTORE_INGESTION_THRESHOLD:
        if embedding_function:
            embedded_data = embedding_function(processed_tensors[embed_data_from])
            embedded_data = np.array(embedded_data, dtype=np.float32)
            processed_tensors[embed_data_to] = embedded_data

        dataset.extend(processed_tensors)
    else:
        elements = dataset_utils.create_elements(processed_tensors)

        ingest_data.run_data_ingestion(
            elements=elements,
            dataset=dataset,
            embedding_function=embedding_function,
            embed_data_to=embed_data_to,
            embed_data_from=embed_data_from,
            ingestion_batch_size=ingestion_batch_size,
            num_workers=num_workers,
            total_samples_processed=total_samples_processed,
        )
