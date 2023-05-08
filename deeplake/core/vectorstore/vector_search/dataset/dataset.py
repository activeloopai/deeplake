import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core import tensor as tensor_utils

try:
    from indra import api

    _INDRA_INSTALLED = True
except ImportError:
    _INDRA_INSTALLED = False

import numpy as np

import uuid
from typing import Iterable, List, Union

from deeplake.constants import DEFAULT_DEEPLAKE_PATH
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

    return create_dataset(dataset_path, token, **kwargs)


def dataset_exists(dataset_path, token, creds, **kwargs):
    return (
        deeplake.exists(dataset_path, token=token, **creds)
        and "overwrite" not in kwargs
    )


def load_dataset(dataset_path, token, creds, logger, read_only, **kwargs):
    if dataset_path == DEFAULT_DEEPLAKE_PATH:
        logger.warning(
            f"Default deeplake path location is used: {DEFAULT_DEEPLAKE_PATH}"
            " and it is not free. All addtionally added data will be added on"
            " top of already existing deeplake dataset."
        )

    dataset = deeplake.load(dataset_path, token=token, read_only=read_only, **kwargs)

    logger.warning(
        f"Deep Lake Dataset in {dataset_path} already exists, "
        f"loading from the storage"
    )
    return dataset


def create_dataset(dataset_path, token, **kwargs):
    dataset = deeplake.empty(dataset_path, token=token, **kwargs)

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


def fetch_embeddings(exec_option, view):
    if exec_option == "python":
        embeddings = view.embedding.numpy()
    elif exec_option in ("indra", "db_engine"):
        embeddings = None
    return embeddings


def get_embedding(embedding, query, embedding_function=None):
    if embedding is None:
        if embedding_function is None:
            raise Exception(
                "Either embedding array or embedding_function should be specified!"
            )

        if embedding is not None:
            always_warn("both embedding and embedding_function are specified. ")
        embedding = embedding_function.embed_query(query)  # type: ignore

    if embedding.dtype != "float32":
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

    return processed_tensors


def create_elements(
    ids: List[str],
    texts: Iterable[str],
    metadatas: List[dict],
    embeddings: Union[List[float], np.ndarray],
):
    processed_tensors = preprocess_tensors(ids, texts, metadatas, embeddings)
    utils.check_length_of_each_tensor(processed_tensors)

    elements = [
        {
            "text": processed_tensors["texts"][i],
            "id": processed_tensors["ids"][i],
            "metadata": processed_tensors["metadatas"][i],
            "embedding": processed_tensors["embeddings"][i],
        }
        for i in range(0, len(processed_tensors["texts"]))
    ]
    return elements
