from requests import Response
from typing import Dict, List, Any

from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.utils import (
    check_response_status,
)
from deeplake.client.config import (
    GET_VECTORSTORE_SUMMARY_SUFFIX,
    LOAD_VECTORSTORE_SUFFIX,
    CREATE_VECTORSTORE_SUFFIX,
    DELETE_VECTORSTORE_SUFFIX,
    FETCH_VECTORSTORE_INDICES_SUFFIX,
    SEARCH_VECTORSTORE_SUFFIX,
    EXTEND_VECTORSTORE_SUFFIX,
    REMOVE_VECTORSTORE_INDICES_SUFFIX,
)


class ManagedServiceClient(DeepLakeBackendClient):
    def _process_response(self, response: Response):
        return response

    def load_vectorstore(self, path: str, mode: str):
        response = self.request(
            method="GET",
            relative_url=LOAD_VECTORSTORE_SUFFIX,
            params={"path": path, "mode": mode},
        )
        check_response_status(response)
        response = response.json()

        return response["mode"]

    def get_vectorstore_summary(self, path: str):
        response = self.request(
            method="GET",
            relative_url=LOAD_VECTORSTORE_SUFFIX,
            params={"path": path},
        )
        response = response.json()

        return response

    def create_vectorstore(self, path: str, tensor_params: List[Dict[str, Any]]):
        response = self.request(
            method="POST",
            relative_url=CREATE_VECTORSTORE_SUFFIX,
            json={"dataset": path, "tensor_params": tensor_params},
        )

        return self._process_response(response)

    def fetch_vectorstore_indices(self, path: str, indices: List[int]):
        response = self.request(
            method="POST",
            relative_url=FETCH_VECTORSTORE_INDICES_SUFFIX,
            json={"dataset": path, "indices": indices},
        )

        return self._process_response(response)

    def search_vectorstore(self, path: str, query: str):
        response = self.request(
            method="POST",
            relative_url=SEARCH_VECTORSTORE_SUFFIX,
            json={"dataset": path, "tql_query": query},
        )

        return self._process_response(response)

    def extend_vectorstore(self, path: str, processed_tensors: List[Dict[str, Any]]):
        response = self.request(
            method="POST",
            relative_url=EXTEND_VECTORSTORE_SUFFIX,
            json={"dataset": path, "data": processed_tensors},
        )

        return self._process_response(response)

    def remove_vectorstore_indices(self, path: str, indices: List[int]):
        response = self.request(
            method="POST",
            relative_url=REMOVE_VECTORSTORE_INDICES_SUFFIX,
            json={"dataset": path, "indices": indices},
        )

        return self._process_response(response)
