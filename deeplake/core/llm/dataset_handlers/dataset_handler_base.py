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
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


class BaseDatasetHandler:
    def __init__(
        self,
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
        username: str = "public",
        **kwargs: Any,
    ):
        self.path = convert_pathlib_to_string_if_needed(path)
        self.logger = logger
        self.org_id = org_id if get_path_type(self.path) == "local" else None
        self.token = token
        self.username = username

        feature_report_path(
            path,
            "vs.initialize",
            {
                "tensor_params": "default"
                if tensor_params is not None
                else tensor_params,
                "embedding_function": True if embedding_function is not None else False,
                "num_workers": num_workers,
                "overwrite": overwrite,
                "read_only": read_only,
                "ingestion_batch_size": ingestion_batch_size,
                "index_params": index_params,
                "exec_option": exec_option,
                "token": self.token,
                "verbose": verbose,
                "runtime": runtime,
            },
            token=self.token,
            username=self.username,
        )

    self.creds = creds or {}
    self.branch = branch
