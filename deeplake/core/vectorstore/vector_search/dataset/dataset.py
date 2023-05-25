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
from deeplake.constants import DEFAULT_VECTORSTORE_DEEPLAKE_PATH, VECTORSTORE_INGESTION_THRESHOLD
from deeplake.util.warnings import always_warn


def create_or_load_dataset(
    dataset_path, token, creds, logger, read_only, exec_option, embedding_function, **kwargs
):
    utils.check_indra_installation(
        exec_option=exec_option, indra_installed=_INDRA_INSTALLED
    )

    if "overwrite" in kwargs and kwargs["overwrite"] == False:
        del kwargs["overwrite"]

    if dataset_exists(dataset_path, token, creds, **kwargs):
        return load_dataset(dataset_path, token, creds, logger, read_only, embedding_function, **kwargs)

    return create_dataset(dataset_path, token, exec_option, embedding_function, **kwargs)


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(dataset_path, token, creds, logger, read_only, embedding_function, **kwargs):
    if dataset_path == DEFAULT_VECTORSTORE_DEEPLAKE_PATH:
        logger.warning(
            f"The default deeplake path location is used: {DEFAULT_VECTORSTORE_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset."
        )

    dataset = deeplake.load(
        dataset_path, token=token, read_only=read_only, creds=creds, **kwargs
    )
    create_tensors_if_needed(dataset, logger, embedding_function)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )
    return dataset


def create_tensors_if_needed(dataset, logger, embedding_function):
    tensors = dataset.tensors

    if "text" not in tensors:
        warn_and_create_missing_tensor(
            logger=logger,
            dataset=dataset,
            tensor_name="text",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )

    if "metadata" not in tensors:
        warn_and_create_missing_tensor(
            logger=logger,
            dataset=dataset,
            tensor_name="metadata",
            htype="json",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )

    if "embedding" not in tensors:
        warn_and_create_missing_tensor(
            logger=logger,
            dataset=dataset,
            tensor_name="embedding",
            htype="embedding",
            dtype=np.float32,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            max_chunk_size=64 * MB,
            create_shape_tensor=True,
        )
        
        update_embedding_info(dataset, embedding_function)

    if "ids" not in tensors:
        warn_and_create_missing_tensor(
            logger=logger,
            dataset=dataset,
            tensor_name="ids",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )


def warn_and_create_missing_tensor(dataset, tensor_name, logger, **kwargs):
    logger.warning(
        f"`{tensor_name}` tensor does not exist in the dataset. If you created dataset manually "
        "and stored text data in another tensor, consider copying the contents of that "
        f"tensor into `{tensor_name}` tensor and deleting if afterwards. To view dataset content "
        "run ds.summary()"
    )
    dataset.create_tensor(
        tensor_name,
        **kwargs,
    )


def create_dataset(dataset_path, token, exec_option, embedding_function, **kwargs):
    runtime = None
    if exec_option == "tensor_db":
        runtime = {"tensor_db": True}

    dataset = deeplake.empty(dataset_path, token=token, runtime=runtime, **kwargs)

    with dataset:
        dataset.create_tensor(
            "text",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
        dataset.create_tensor(
            "metadata",
            htype="json",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
        dataset.create_tensor(
            "embedding",
            htype="embedding",
            dtype=np.float32,
            create_id_tensor=False,
            create_sample_info_tensor=False,
            max_chunk_size=64 * MB,
            create_shape_tensor=True,
        )
        update_embedding_info(dataset, embedding_function)
        
        dataset.create_tensor(
            "ids",
            htype="text",
            create_id_tensor=False,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            chunk_compression="lz4",
        )
    return dataset


def delete_and_commit(dataset, ids):
    with dataset:
        for id in sorted(ids)[::-1]:
            dataset.pop(id)
        dataset.commit(f"deleted {len(ids)} samples", allow_empty=True)


def delete_all_samples_if_specified(dataset, delete_all):
    if delete_all:
        dataset = deeplake.empty(dataset.path, overwrite=True)
        return dataset, True
    return dataset, False


def fetch_embeddings(exec_option, view, logger):
    if exec_option == "python":
        logger.warning(
            "Python implementation fetches all of the dataset's embedding into memory. "
            "With big datasets this could be quite slow and potentially result in performance issues. "
            "Use `exec_option = 'tensor_db'` for better performance."
        )
        embeddings = view.embedding.numpy()
    elif exec_option in ("compute_engine", "tensor_db"):
        embeddings = None
    return embeddings


def get_embedding(embedding, query, embedding_function=None):
    if embedding is None:
        if embedding_function is None:
            raise Exception(
                "Either embedding array or embedding_function should be specified!"
            )

    if embedding_function is not None:
        if embedding is not None:
            always_warn("both embedding and embedding_function are specified. ")
        embedding = embedding_function(query)  # type: ignore

    if isinstance(embedding, list) or embedding.dtype != "float32":
        embedding = np.array(embedding, dtype=np.float32)

    return embedding


def preprocess_tensors(ids, texts, metadatas, embeddings):
    if ids is None:
        ids = [str(uuid.uuid1()) for _ in texts]

    if not isinstance(texts, list):
        texts = list(texts)

    if metadatas is None:
        metadatas = [{}] * len(texts)

    if embeddings is None:
        embeddings = [None] * len(texts)

    processed_tensors = {
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }

    return processed_tensors, ids


def create_elements(
    processed_tensors: Dict[str, List[Any]],
):
    utils.check_length_of_each_tensor(processed_tensors)

    elements = [
        {
            "text": processed_tensors["texts"][i],
            "id": processed_tensors["ids"][i],
            "metadata": processed_tensors["metadatas"][i],
            "embedding": processed_tensors["embeddings"][i],
        }
        for i in range(len(processed_tensors["texts"]))
    ]
    return elements


def fetch_tensor_based_on_htype(dataset, htype):
    tensors = dataset.tensors
    
    if "embedding" in tensors:
        return dataset.embedding
    
    num_of_tensors_with_htype = 0
    
    tensor_names = []
    for tensor in tensors:
        if dataset[tensor].htype == "embedding":
            num_of_tensors_with_htype += 1
            tensor_names += []
    tensor_names = tensor_names
    tensor_names_str = " ".join(tensor_name+" ," for tensor_name in tensor_names)
    tensor_names_str = tensor_names_str[:-2]
    
    if num_of_tensors_with_htype > 1:
        always_warn(
            f"{num_of_tensors_with_htype} tensors with `embedding` htype were found. "
            f"They are: {tensor_names_str}. Embedding function info will be appended to "
            f"{tensor_names[0]}."
        )

    return dataset[tensor_names[0]]


def set_embedding_info(tensor, embedding_function):
    if embedding_function:
        tensor.info["embeding"] = {
            "model": embedding_function.__dict__.get("model"),
            "deployment": embedding_function.__dict__.get("deployment"),
            "embedding_ctx_length": embedding_function.__dict__.get(
                "embedding_ctx_length"
            ),
            "openai_api_key": embedding_function.__dict__.get(
                "openai_api_key"
            ),
            "openai_organization": embedding_function.__dict__.get(
                "openai_organization"
            ),
            "chunk_size": embedding_function.__dict__.get("chunk_size"),
            "max_retries": embedding_function.__dict__.get("max_retries"),
        }


def update_embedding_info(dataset, embedding_function):
    tensor = fetch_tensor_based_on_htype(dataset, embedding_function)
    set_embedding_info(tensor, embedding_function)


def extend_or_ingest_dataset(processed_tensors, dataset, embedding_function, ingestion_batch_size, num_workers, total_samples_processed):
    if len(processed_tensors) < VECTORSTORE_INGESTION_THRESHOLD:
        dataset.extend(processed_tensors, skip_ok=True)
    else:
        elements = dataset_utils.create_elements(processed_tensors)

        ingest_data.run_data_ingestion(
            elements=elements,
            dataset=dataset,
            embedding_function=embedding_function,
            ingestion_batch_size=ingestion_batch_size,
            num_workers=num_workers,
            total_samples_processed=total_samples_processed,
        )