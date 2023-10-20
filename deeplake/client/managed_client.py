from requests import Response
from typing import Dict, List, Any

from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.utils import (
    check_response_status,
)
from deeplake.client.config import (
    GET_VECTORSTORE_SUMMARY_SUFFIX,
    INIT_VECTORSTORE_SUFFIX,
    DELETE_VECTORSTORE_SUFFIX,
    VECTORSTORE_ADD_SUFFIX,
    VECTORSTORE_REMOVE_INDICES_SUFFIX,
    VECTORSTORE_SEARCH_SUFFIX,
)


class ManagedServiceClient(DeepLakeBackendClient):
    def _process_response(self, response: Response):
        return response

    def init_vectorstore(
        self, path: str, overwrite: bool, tensor_params: List[Dict[str, Any]]
    ):
        response = self.request(
            method="GET",
            relative_url=INIT_VECTORSTORE_SUFFIX,
            params={
                "path": path,
                "overwrite": overwrite,
                "tensor_params": tensor_params,
            },
        )
        check_response_status(response)
        response = response.json()

        return response["summary"]

    def get_vectorstore_summary(self, path: str):
        org_id, dataset_id = path[6:].split("/")
        response = self.request(
            method="GET",
            relative_url=GET_VECTORSTORE_SUMMARY_SUFFIX % (org_id, dataset_id),
            params={
                "path": path,
            },
        )
        response = response.json()

        return response

    def vectorstore_search(self, path: str, query: str):
        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_SEARCH_SUFFIX,
            json={"dataset": path, "tql_query": query},
        )

        return self._process_response(response)

    def vectorstore_add(self, path: str, processed_tensors: List[Dict[str, Any]]):
        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_ADD_SUFFIX,
            json={"dataset": path, "data": processed_tensors},
        )

        return self._process_response(response)

    def remove_vectorstore_indices(self, path: str, indices: List[int]):
        response = self.request(
            method="POST",
            relative_url=VECTORSTORE_REMOVE_INDICES_SUFFIX,
            json={"dataset": path, "indices": indices},
        )

        return self._process_response(response)

    def update_vectorstore_indices(
        self, path: str, row_ids: List[Dict[str, Any]], embedding_tensor_data
    ):
        pass
