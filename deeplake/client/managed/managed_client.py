import numpy as np
from typing import Callable, Dict, List, Any, Optional, Union

from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.utils import (
    check_response_status,
)
from deeplake.client.config import (
    GET_VECTORSTORE_SUMMARY_SUFFIX,
    INIT_VECTORSTORE_SUFFIX,
    DELETE_VECTORSTORE_SUFFIX,
    VECTORSTORE_ADD_SUFFIX,
    VECTORSTORE_REMOVE_ROWS_SUFFIX,
    VECTORSTORE_SEARCH_SUFFIX,
)

from deeplake.client.managed.models import (
    VectorStoreSummaryResponse,
    VectorStoreInitResponse,
    VectorStoreSearchResponse,
    VectorStoreAddResponse,
)


class ManagedServiceClient(DeepLakeBackendClient):
    def _preprocess_embedding(self, embedding: Union[List[float], np.ndarray, None]):
        if embedding is not None and isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

    def init_vectorstore(
        self,
        path: str,
        overwrite: Optional[bool] = None,
        tensor_params: Optional[List[Dict[str, Any]]] = None,
    ):
        response = self.request(
            method="POST",
            relative_url=INIT_VECTORSTORE_SUFFIX,
            json={
                "dataset": path,
                "overwrite": overwrite,
                "tensor_params": tensor_params,
            },
        )
        data = response.json()

        return VectorStoreInitResponse(
            status_code=response.status_code,
            path=data["path"],
            summary=data["summary"],
            length=data["length"],
            tensors=data["tensors"],
            exists=data.get("exists", False),
        )

    def delete_vectorstore(self, path: str, force: bool = False):
        response = self.request(
            method="DELETE",
            relative_url=DELETE_VECTORSTORE_SUFFIX,
            json={"dataset": path, "force": force},
        )
        check_response_status(response)

    def get_vectorstore_summary(self, path: str):
        org_id, dataset_id = path[6:].split("/")
        response = self.request(
            method="GET",
            relative_url=GET_VECTORSTORE_SUMMARY_SUFFIX.format(org_id, dataset_id),
        )
        check_response_status(response)
        data = response.json()

        return VectorStoreSummaryResponse(
            status_code=response.status_code,
            summary=data["summary"],
            length=data["length"],
            tensors=data["tensors"],
        )

    def vectorstore_search(
        self,
        path: str,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: Optional[str] = None,
        query: Optional[str] = None,
        filter: Optional[Dict[str, str]] = None,
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        deep_memory: bool = False,
    ):
        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_SEARCH_SUFFIX,
            json={
                "dataset": path,
                "embedding": self._preprocess_embedding(embedding),
                "k": k,
                "distance_metric": distance_metric,
                "query": query,
                "filter": filter,
                "embedding_tensor": embedding_tensor,
                "return_tensors": return_tensors,
                "deep_memory": deep_memory,
            },
        )
        check_response_status(response)
        data = response.json()

        return VectorStoreSearchResponse(
            status_code=response.status_code,
            length=data["length"],
            data=data["data"],
        )

    def vectorstore_add(
        self,
        path: str,
        processed_tensors: List[Dict[str, List[Any]]],
        rate_limiter: Optional[Dict[str, Any]] = None,
        return_ids: bool = False,
    ):
        rest_api_tensors = []
        for tensor in processed_tensors:
            for key, value in tensor.items():
                tensor[key] = self._preprocess_embedding(value)
            rest_api_tensors.append(tensor)

        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_ADD_SUFFIX,
            json={
                "dataset": path,
                "data": rest_api_tensors,
                "rate_limiter": rate_limiter,
                "return_ids": return_ids,
            },
        )
        check_response_status(response)
        data = response.json().get("result", {})

        return VectorStoreAddResponse(status_code=response.status_code, ids=data)

    def vectorstore_remove_rows(
        self,
        path: str,
        row_ids: Optional[List[int]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, str]] = None,
        query: Optional[str] = None,
        delete_all: bool = False,
    ):
        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_REMOVE_ROWS_SUFFIX,
            json={
                "dataset": path,
                "row_ids": row_ids,
                "ids": ids,
                "filter": filter,
                "query": query,
                "delete_all": delete_all,
            },
        )
        check_response_status(response)

    def vectorstore_update_embeddings(
        self,
        path: str,
        row_ids: List[str],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        embedding_function: Union[Callable, List[Callable]],
        embedding_source_tensor: Union[str, List[str]],
        embedding_tensor: Union[str, List[str]],
    ):
        """
        TODO: implement
        """
        pass
