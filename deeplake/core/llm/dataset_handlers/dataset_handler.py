import logging
import pathlib
from typing import Optional, Any, List, Dict, Union, Callable
import jwt

import numpy as np

import deeplake
from deeplake.core import index_maintenance
from deeplake.core.distance_type import DistanceType
from deeplake.util.dataset import try_flushing
from deeplake.util.path import convert_pathlib_to_string_if_needed

from deeplake.api import dataset
from deeplake.core.dataset import Dataset
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    MAX_BYTES_PER_MINUTE,
    TARGET_BYTE_SIZE,
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    DEFAULT_DEEPMEMORY_DISTANCE_METRIC,
    _INDEX_OPERATION_MAPPING,
)
from deeplake.client.utils import read_token
from deeplake.core.llm import utils
from deeplake.core.llm.vector_search import vector_search
from deeplake.core.llm.vector_search import dataset as dataset_utils
from deeplake.core.llm.vector_search import filter as filter_utils
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


EXEC_OPTION_TO_DATASET_HANDLER = {
    "compute_engine": RegularDatasetHandler,
    "python": RegularDatasetHandler,
    "tensor_db": ManagerDatasetHandler,
}


def get_dataset_manager(
    path: Union[str, pathlib.Path],
    tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
    embedding_function: Optional[Any] = None,
    read_only: Optional[bool] = None,
    ingestion_batch_size: int = 1000,
    index_params: Optional[Dict[str, Union[int, str]]] = None,
    num_workers: int = 0,
    exec_option: str = "auto",
    token: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
    runtime: Optional[Dict] = None,
    creds: Optional[Union[Dict, str]] = None,
    org_id: Optional[str] = None,
    logger: logging.Logger = None,
    branch: str = "main",
    **kwargs: Any,
):
    try:
        from indra import api  # type: ignore

        indra_installed = True
    except Exception:  # pragma: no cover
        indra_installed = False  # pragma: no cover

    token = get_token(token)
    username = get_username(token)
    exec_option = get_exec_option(dataset, exec_option, indra_installed, username)
    return EXEC_OPTION_TO_DATASET_HANDLER[exec_option](
        path=path,
        tensor_params=tensor_params,
        embedding_function=embedding_function,
        read_only=read_only,
        ingestion_batch_size=ingestion_batch_size,
        index_params=index_params,
        num_workers=num_workers,
        exec_option=exec_option,
        token=token,
        overwrite=overwrite,
        verbose=verbose,
        runtime=runtime,
        creds=creds,
        org_id=org_id,
        logger=logger,
        branch=branch,
        **kwargs,
    )


def get_exec_option(dataset, exec_option, indra_installed, username) -> str:
    return utils.parse_exec_option(dataset, exec_option, indra_installed, username)


def get_username(
    token,
):
    username = "public"
    if token is not None:
        username = jwt.decode(token, options={"verify_signature": False})["id"]
    return username


def get_token(
    token,
):
    return token or read_token(from_env=True)
