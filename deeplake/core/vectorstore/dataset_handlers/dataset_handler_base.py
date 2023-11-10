import logging
import pathlib
from abc import abstractmethod, ABC
from typing import Optional, Any, List, Dict, Union, Callable
import jwt

import numpy as np

from deeplake.util.path import convert_pathlib_to_string_if_needed
from deeplake.api import dataset
from deeplake.core.dataset import Dataset
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    MAX_BYTES_PER_MINUTE,
    TARGET_BYTE_SIZE,
)
from deeplake.client.utils import read_token
from deeplake.core.vectorstore import utils
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


class DHBase(ABC):
    """Base class for dataset handlers."""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        dataset: Optional[Dataset] = None,
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
        **kwargs: Any,
    ):
        self.path = path
        self.dataset = dataset
        if dataset and path:
            raise ValueError(
                "Only one of `dataset` or path should be provided to the dataset handler."
            )
        elif not dataset and not path:
            raise ValueError("Either `dataset` or path should be provided.")
        elif self.path:
            self.path = convert_pathlib_to_string_if_needed(path)
        else:
            self.dataset = dataset
            self.path = dataset.path

        self._token = token
        self.logger = logger
        self.org_id = org_id if get_path_type(self.path) == "local" else None

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
                "path": self.path,
            },
            token=self.token,
            username=self.username,
        )

        self.ingestion_batch_size = ingestion_batch_size
        self.index_params = utils.parse_index_params(index_params)
        kwargs["index_params"] = self.index_params
        self.num_workers = num_workers
        self.creds = creds or {}
        self.embedding_function = utils.create_embedding_function(embedding_function)

    @property
    def token(self):
        return self._token or read_token(from_env=True)

    @property
    def exec_option(self) -> str:
        return utils.parse_exec_option(
            self.dataset, self._exec_option, self.indra_installed, self.username
        )

    @property
    def username(self) -> str:
        username = "public"
        if self.token is not None:
            try:
                username = jwt.decode(self.token, options={"verify_signature": False})[
                    "id"
                ]
            except Exception:
                pass
        return username

    @abstractmethod
    def add(
        self,
        embedding_function: Optional[Union[Callable, List[Callable]]] = None,
        embedding_data: Optional[Union[List, List[List]]] = None,
        embedding_tensor: Optional[Union[str, List[str]]] = None,
        return_ids: bool = False,
        rate_limiter: Dict = {
            "enabled": False,
            "bytes_per_minute": MAX_BYTES_PER_MINUTE,
            "batch_byte_size": TARGET_BYTE_SIZE,
        },
        **tensors,
    ):
        pass

    @abstractmethod
    def search(
        self,
        embedding_data: Union[str, List[str], None] = None,
        embedding_function: Optional[Callable] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: Optional[str] = None,
        query: Optional[str] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        exec_option: Optional[str] = None,
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        return_view: bool = False,
        deep_memory: bool = False,
    ) -> Union[Dict, Dataset]:
        pass

    @abstractmethod
    def delete(
        self,
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        pass

    @abstractmethod
    def update_embedding(
        self,
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        embedding_function: Optional[Union[Callable, List[Callable]]] = None,
        embedding_source_tensor: Union[str, List[str]] = "text",
        embedding_tensor: Optional[Union[str, List[str]]] = None,
    ):
        pass

    @abstractmethod
    def tensors(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def commit(self, allow_empty: bool = True) -> None:
        """Commits the Vector Store.

        Args:
            allow_empty (bool): Whether to allow empty commits. Defaults to True.
        """
        raise NotImplementedError()

    def checkout(self, branch: str = "main") -> None:
        """Checkout the Vector Store to a specific branch.

        Args:
            branch (str): Branch name to checkout. Defaults to "main".
        """
        raise NotImplementedError()
