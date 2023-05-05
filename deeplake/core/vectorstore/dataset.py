import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error
from deeplake.core.vectorstore import utils

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
        always_warn(
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
        dataset.delete(large_ok=True)
        return True
    return False


def fetch_embeddings(exec_option, view):
    if exec_option == "python":
        embeddings = view.embedding.numpy()
    elif exec_option in ("indra", "db_engine"):
        embeddings = None
    return embeddings


def get_embedding(embedding, query, _embedding_function=None):
    if embedding is None:
        if _embedding_function is None:
            raise Exception(
                "Either embedding array or embedding_function should be specified!"
            )
        embedding = _embedding_function.embed_query(query)  # type: ignore

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

    return ids, texts, metadatas, embeddings


def create_elements(
    ids: List[str],
    texts: Iterable[str],
    metadatas: List[dict],
    embeddings: Union[List[float], np.ndarray],
):
    ids, texts, metadatas, embeddings = preprocess_tensors(
        ids, texts, metadatas, embeddings
    )

    elements = [
        {"text": text, "metadata": metadata, "id": id_, "embedding": embedding}
        for text, metadata, id_, embedding in zip(texts, metadatas, ids, embeddings)
    ]
    return elements
