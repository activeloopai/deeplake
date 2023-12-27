import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from deeplake.client.managed.managed_client import ManagedServiceClient
from deeplake.client.utils import read_token
from deeplake.constants import MAX_BYTES_PER_MINUTE, TARGET_BYTE_SIZE
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.dataset_handlers.dataset_handler_base import DHBase
from deeplake.core.vectorstore.deep_memory.deep_memory import (
    DeepMemory,
    use_deep_memory,
)
from deeplake.core.vectorstore import utils
from deeplake.util.bugout_reporter import feature_report_path
from deeplake.util.path import convert_pathlib_to_string_if_needed, get_path_type
from deeplake.util.logging import log_visualizer_link
from deeplake.client.log import logger


class ManagedVectorStoreArgsVerifier:
    def __init__(self) -> None:
        pass

    @staticmethod
    def verify_init_args(cls, **kwargs):
        args_verifier = InitArgsVerfier(**kwargs)
        args_verifier.verify(cls)

    @staticmethod
    def verify_add_args(**kwargs):
        args_verifier = AddArgsVerfier(**kwargs)
        args_verifier.verify()

    @staticmethod
    def verify_search_args(**kwargs):
        args_verifier = SearchArgsVerfier(**kwargs)
        args_verifier.verify()

    @staticmethod
    def verify_delete_args(**kwargs):
        args_verifier = DeleteArgsVerfier(**kwargs)
        args_verifier.verify()

    @staticmethod
    def verify_delete_by_path_args(**kwargs):
        args_verifier = DeleteByPathArgsVerfier(**kwargs)
        args_verifier.verify()

    @staticmethod
    def verify_update_args(**kwargs):
        args_verifier = UpdateArgsVerfier(**kwargs)
        args_verifier.verify()


class ManagedDH(DHBase):
    args_verifier = ManagedVectorStoreArgsVerifier()

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
            branch=branch,
            **kwargs,
        )
        # because we don't support read/write access to the managed vectorstore using deeplake
        del self.dataset

        self.verbose = verbose

        # verifying not implemented args
        self.args_verifier.verify_init_args(
            cls=self,
            dataset=dataset,
            embedding_function=embedding_function,
            exec_option=exec_option,
            creds=creds,
            org_id=org_id,
            other_kwargs=kwargs,
        )

        self.client = ManagedServiceClient(token=self.token)
        response = self.client.init_vectorstore(
            path=self.bugout_reporting_path,
            overwrite=overwrite,
            tensor_params=tensor_params,
            index_params=index_params,
            verbose=verbose,
        )

        if self.verbose:
            log_visualizer_link(response.path)
            logger.info(response.summary)

        self.deep_memory = DeepMemory(
            path=self.path,
            token=self.token,
            logger=self.logger,
            embedding_function=self.embedding_function,
            creds=self.creds,
        )

    def add(
        self,
        embedding_function: Union[Callable, List[Callable]],
        embedding_data: Union[List, List[List]],
        embedding_tensor: Union[str, List[str]],
        return_ids: bool,
        rate_limiter: Dict,
        **tensors,
    ) -> Optional[List[str]]:
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.add",
            parameters={
                "tensors": list(tensors.keys()) if tensors else None,
                "embedding_tensor": embedding_tensor,
                "return_ids": return_ids,
                "embedding_function": True if embedding_function is not None else False,
                "embedding_data": True if embedding_data is not None else False,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if self.verbose:
            self.logger.info("Uploading data to deeplake dataset.")

        # verifying not implemented args
        self.args_verifier.verify_add_args(
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            rate_limiter=rate_limiter,
        )

        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_tensors_kwargs(
            tensors, embedding_function, embedding_data, embedding_tensor
        )

        processed_tensors = {
            t: tensors[t].tolist() if isinstance(tensors[t], np.ndarray) else tensors[t]
            for t in tensors
        }
        utils.check_length_of_each_tensor(processed_tensors)

        response = self.client.vectorstore_add(
            path=self.path,
            processed_tensors=processed_tensors,
            rate_limiter=rate_limiter,
            return_ids=return_ids,
        )

        if self.verbose:
            self.summary()

        if return_ids:
            return response.ids

    @use_deep_memory
    def search(
        self,
        embedding_data: Union[str, List[str]],
        embedding_function: Optional[Callable],
        embedding: Union[List[float], np.ndarray],
        k: int,
        distance_metric: str,
        query: str,
        filter: Union[Dict, Callable],
        embedding_tensor: str,
        return_tensors: List[str],
        return_view: bool,
        deep_memory: bool,
        exec_option: Optional[str] = "tensor_db",
    ) -> Union[Dict, Dataset]:
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.search",
            parameters={
                "embedding_data": True if embedding_data is not None else False,
                "embedding_function": True if embedding_function is not None else False,
                "k": k,
                "distance_metric": distance_metric,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "embedding_tensor": embedding_tensor,
                "embedding": True if embedding is not None else False,
                "return_tensors": return_tensors,
                "return_view": return_view,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        # verifying not implemented args
        self.args_verifier.verify_search_args(
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            exec_option=exec_option,
            return_view=return_view,
            filter=filter,
        )

        response = self.client.vectorstore_search(
            path=self.path,
            embedding=embedding,
            k=k,
            distance_metric=distance_metric,
            query=query,
            filter=filter,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            deep_memory=deep_memory,
        )
        return response.data

    def delete(
        self,
        row_ids: List[int],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        exec_option: str,
        delete_all: bool,
    ) -> bool:
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "delete_all": delete_all,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        self.args_verifier.verify_delete_args(
            filter=filter,
            exec_option=exec_option,
        )

        self.client.vectorstore_remove_rows(
            path=self.bugout_reporting_path,
            row_ids=row_ids,
            ids=ids,
            filter=filter,
            query=query,
            delete_all=delete_all,
        )
        return True

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
        embedding_dict: Union[
            List[float], np.ndarray, List[List[float]], List[np.ndarray]
        ],
    ):
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        self.args_verifier.verify_update_args(
            exec_option=exec_option,
            embedding_function=embedding_function,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=embedding_tensor,
        )

        self.client.vectorstore_update_embeddings(
            path=self.bugout_reporting_path,
            embedding_tensor=embedding_tensor,
            row_ids=row_ids,
            ids=ids,
            filter=filter,
            query=query,
            embedding_dict=embedding_dict,
        )

    @staticmethod
    def delete_by_path(
        path: str,
        force: bool,
        creds: Union[Dict, str],
        token: str,
    ):
        feature_report_path(
            path=path,
            feature_name="vs.delete_by_path",
            parameters={
                "path": path,
                "force": force,
                "creds": creds,
                "managed": True,
            },
            token=token,
        )

        ManagedVectorStoreArgsVerifier.verify_delete_by_path_args(
            force=force,
            creds=creds,
        )

        client = ManagedServiceClient(token=token)
        client.delete_vectorstore_by_path(
            path=path,
            force=force,
            creds=creds,
        )

    def _get_summary(self):
        """Returns a summary of the Managed Vector Store."""
        return self.client.get_vectorstore_summary(self.path)

    def tensors(self):
        """Returns the list of tensors present in the dataset"""
        return [t["name"] for t in self._get_summary().tensors]

    def summary(self):
        """Prints a summary of the dataset"""
        print(self._get_summary().summary)

    def __len__(self):
        """Length of the dataset"""
        return self._get_summary().length


class ArgsVerifierBase:
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs

    def verify(self):
        for arg in self.kwargs:
            self._verify_argument(arg, self.kwargs)

        self._exec_option_verifier(self.kwargs)

    @classmethod
    def _verify_argument(cls, argument, kwargs):
        if argument in cls._not_implemented_args and kwargs[argument] is not None:
            raise NotImplementedError(
                f"{argument} is not supported for ManagedVectorStore for now."
            )

    @staticmethod
    def _exec_option_verifier(kwargs):
        exec_option = kwargs.get("exec_option", "auto")
        if exec_option is not None and exec_option not in ("tensor_db", "auto"):
            raise NotImplementedError(
                "ManagedVectorStore does not support passing exec_option other than `tensor_db` for now."
            )

    def _verify_filter_is_dictionary(self):
        filter_ = self.kwargs.get("filter", None)
        if filter_ is not None and not isinstance(filter_, dict):
            raise NotImplementedError(
                "Only Filter Dictionary is supported for the ManagedVectorStore."
            )


class InitArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "dataset",
        "embedding_function",
        "creds",
        "org_id",
    ]

    def verify(self, cls):
        super().verify()
        if cls.deserialized_vectorstore:
            raise NotImplementedError(
                "ManagedVectorStore does not support passing path to serialized vectorstore object for now."
            )

        if get_path_type(cls.path) != "hub":
            raise ValueError(
                "ManagedVectorStore can only be initialized with a Deep Lake Cloud path."
            )

        if self.kwargs.get("other_kwargs", {}) != {}:
            other_kwargs = self.kwargs["other_kwargs"]
            other_kwargs_names = list(other_kwargs.keys())
            other_kwargs_names = "`" + "` ,`".join(other_kwargs_names) + "`"

            raise NotImplementedError(
                f"ManagedVectorStore does not support passing: {other_kwargs_names} for now."
            )


class AddArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "embedding_function",
        "embedding_data",
        "embedding_tensor",
    ]

    def verify(self):
        super().verify()

        if (
            self.kwargs.get("rate_limiter", None) is not None
            and self.kwargs["rate_limiter"]["enabled"] != False
        ):
            raise NotImplementedError(
                "rate_limiter is not supported for the ManagedVectorStore."
            )


class SearchArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "embedding_function",
        "embedding_data",
        "exec_option",
    ]

    def verify(self):
        super().verify()
        self._verify_filter_is_dictionary()

        if self.kwargs.get("return_view", False) is not False:
            raise NotImplementedError(
                "return_view is not supported for the ManagedVectorStore."
            )


class UpdateArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "embedding_function",
        "exec_option",
    ]

    def verify(self):
        super().verify()
        self._verify_filter_is_dictionary()


class DeleteArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "exec_option",
    ]

    def verify(self):
        super().verify()
        self._verify_filter_is_dictionary()


class DeleteByPathArgsVerfier(ArgsVerifierBase):
    _not_implemented_args = [
        "force",
        "creds",
    ]
