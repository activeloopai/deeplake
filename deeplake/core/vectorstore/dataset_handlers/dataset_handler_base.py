import logging
import os
import pathlib
from abc import abstractmethod, ABC
from typing import Optional, Any, List, Dict, Union, Callable
import jwt

import numpy as np

from deeplake.client.config import DEEPLAKE_AUTH_TOKEN
from deeplake.util.path import convert_pathlib_to_string_if_needed
from deeplake.core.dataset import Dataset
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
        **kwargs: Any,
    ):
        self._exec_option = exec_option

        self.path: Optional[str] = None
        self.dataset = dataset
        if dataset and path:
            raise ValueError(
                "Only one of `dataset` or path should be provided to the dataset handler."
            )
        elif not dataset and not path:
            raise ValueError("Either `dataset` or path should be provided.")
        elif path:
            self.path = convert_pathlib_to_string_if_needed(path)
        else:
            self.dataset = dataset
            self.path = dataset.path

        self._token = token
        self.logger = logger
        self.org_id = org_id if get_path_type(self.path) == "local" else None
        self.bugout_reporting_path = self.path or dataset.path

        feature_report_path(
            self.bugout_reporting_path,
            "vs.initialize",
            {
                "tensor_params": (
                    "default" if tensor_params is not None else tensor_params
                ),
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
        return self._token or os.environ.get(DEEPLAKE_AUTH_TOKEN)

    @property
    def exec_option(self) -> str:
        return utils.parse_exec_option(self.dataset, self._exec_option, self.username)

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
        embedding_function: Union[Callable, List[Callable]],
        embedding_data: Union[List, List[List]],
        embedding_tensor: Union[str, List[str]],
        return_ids: bool,
        rate_limiter: Dict,
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
        row_ids: List[int],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        exec_option: str,
        delete_all: bool,
    ) -> bool:
        pass

    @abstractmethod
    def update_embedding(
        self,
        row_ids: List[str],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        exec_option: str,
        embedding_function: Union[Callable, List[Callable]],
        embedding_source_tensor: Union[str, List[str]],
        embedding_tensor: Union[str, List[str]],
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

        Raises:
            NotImplementedError: This method is not implemented by the base class.
        """
        raise NotImplementedError()

    def checkout(self, branch: str, create: bool) -> None:
        """Checkout the Vector Store to a specific branch.

        Args:
            branch (str): Branch name to checkout. Defaults to "main".
            create (bool): Whether to create the branch if it does not exist. Defaults to False.

        Raises:
            NotImplementedError: This method is not implemented by the base class.
        """
        raise NotImplementedError()
