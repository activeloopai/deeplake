import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import deeplake
from deeplake.client.utils import read_token
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    _INDEX_OPERATION_MAPPING,
)
from deeplake.core import index_maintenance
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.dataset_handlers.dataset_handler_base import DHBase
from deeplake.core.vectorstore.deep_memory.deep_memory import (
    use_deep_memory,
    DeepMemory,
)
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.util.bugout_reporter import feature_report_path
from deeplake.util.exceptions import DeepMemoryWaitingListError


class ClientSideDH(DHBase):
    def __init__(
        self,
        path: Union[str, pathlib.Path],
        dataset: Dataset,
        tensor_params: List[Dict[str, object]],
        embedding_function: Any,
        read_only: bool,
        ingestion_batch_size: int,
        index_params: Dict[str, Union[int, str]],
        num_workers: int,
        exec_option: str,
        token: str,
        overwrite: bool,
        verbose: bool,
        runtime: Dict,
        creds: Union[Dict, str],
        org_id: str,
        logger: logging.Logger,
        branch: str,
        **kwargs: Any,
    ):
        super().__init__(
            path=path,
            dataset=dataset,
            tensor_params=tensor_params,
            embedding_function=embedding_function,
            read_only=read_only,
            ingestion_batch_size=ingestion_batch_size,
            index_params=index_params,
            num_workers=num_workers,
            exec_option=exec_option,
            token=token,
            overwrite=overwrite,
            verbose=True,
            runtime=runtime,
            creds=creds,
            org_id=org_id,
            logger=logger,
            **kwargs,
        )

        self.verbose = verbose
        self.tensor_params = tensor_params
