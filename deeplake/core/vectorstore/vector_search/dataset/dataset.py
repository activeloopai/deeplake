import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core import tensor as tensor_utils

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import numpy as np

import uuid
from typing import Iterable, List, Union, Optional

from deeplake.constants import DEFAULT_VECTORSTORE_DEEPLAKE_PATH
from deeplake.util.warnings import always_warn


def create_or_load_dataset(
    dataset_path, token, creds, logger, read_only, exec_option, **kwargs
):
    utils.check_indra_installation(
        exec_option=exec_option, indra_installed=_INDRA_INSTALLED
    )

    if "overwrite" in kwargs and kwargs["overwrite"] == False:
        del kwargs["overwrite"]

    if dataset_exists(dataset_path, token, creds, **kwargs):
        return load_dataset(dataset_path, token, creds, logger, read_only, **kwargs)

    return create_dataset(dataset_path, token, exec_option, **kwargs)


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(dataset_path, token, creds, logger, read_only, **kwargs):
    if dataset_path == DEFAULT_VECTORSTORE_DEEPLAKE_PATH:
        logger.warning(
            f"The default deeplake path location is used: {DEFAULT_VECTORSTORE_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset."
        )

    dataset = deeplake.load(
        dataset_path, token=token, read_only=read_only, creds=creds, **kwargs
    )
    create_tensors_if_needed(dataset, logger)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )
    return dataset


def create_tensors_if_needed(dataset, logger):
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


def create_dataset(dataset_path, token, exec_option, **kwargs):
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
    texts: Iterable[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
    embeddings: Optional[Union[List[float], np.ndarray]] = None,
):
    processed_tensors, ids = preprocess_tensors(ids, texts, metadatas, embeddings)
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
    return elements, ids
