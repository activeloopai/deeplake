import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional

from deeplake.enterprise.util import INDRA_INSTALLED
from deeplake.core.vectorstore.vector_search.indra import query
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.dataset.indra_dataset_view import IndraDatasetView


class SearchBasic(ABC):
    def __init__(
        self,
        deeplake_dataset: DeepLakeDataset,
        org_id: Optional[str] = None,
        token: Optional[str] = None,
        runtime: Optional[Dict] = None,
        deep_memory: bool = False,
    ):
        """Base class for all search algorithms.
        Args:
            deeplake_dataset (DeepLakeDataset): DeepLake dataset object.
            org_id (Optional[str], optional): Organization ID, is needed only for local datasets. Defaults to None.
            token (Optional[str], optional): Token used for authentication. Defaults to None.
            runtime (Optional[Dict], optional): Whether to run query on managed_db or indra. Defaults to None.
            deep_memory (bool): Use DeepMemory for the search. Defaults to False.
        """
        self.deeplake_dataset = deeplake_dataset
        self.org_id = org_id
        self.token = token
        self.runtime = runtime
        self.deep_memory = deep_memory

    def run(
        self,
        tql_string: str,
        return_view: bool,
        return_tql: bool,
        distance_metric: str,
        k: int,
        query_embedding: np.ndarray,
        embedding_tensor: str,
        tql_filter: str,
        return_tensors: List[str],
    ):
        tql_query = self._create_tql_string(
            tql_string,
            distance_metric,
            k,
            query_embedding,
            embedding_tensor,
            tql_filter,
            return_tensors,
        )

        view = self._get_view(
            tql_query,
            runtime=self.runtime,
        )

        if return_view:
            return view

        return_data = self._collect_return_data(view)

        if return_tql:
            return {"data": return_data, "tql": tql_query}
        return return_data

    @abstractmethod
    def _collect_return_data(
        self,
        view: DeepLakeDataset,
    ):
        pass

    @staticmethod
    def _create_tql_string(
        tql_string: str,
        distance_metric: str,
        k: int,
        query_embedding: np.ndarray,
        embedding_tensor: str,
        tql_filter: str,
        return_tensors: List[str],
    ):
        """Creates TQL query string for the vector search."""
        if tql_string:
            return tql_string
        else:
            return query.parse_query(
                distance_metric,
                k,
                query_embedding,
                embedding_tensor,
                tql_filter,
                return_tensors,
            )

    @abstractmethod
    def _get_view(self, tql_query: str, runtime: Optional[Dict] = None):
        pass


class SearchIndra(SearchBasic):
    def _get_view(self, tql_query, runtime: Optional[Dict] = None):
        indra_dataset = self._get_indra_dataset()
        indra_view = indra_dataset.query(tql_query)
        view = IndraDatasetView(indra_ds=indra_view)
        view._tql_query = tql_query
        return view

    def _get_indra_dataset(self):
        if not INDRA_INSTALLED:
            from deeplake.enterprise.util import raise_indra_installation_error

            raise raise_indra_installation_error(indra_import_error=None)

        if self.deeplake_dataset.libdeeplake_dataset is not None:
            indra_dataset = self.deeplake_dataset.libdeeplake_dataset
        else:
            from deeplake.enterprise.convert_to_libdeeplake import (
                dataset_to_libdeeplake,
            )

            if self.org_id is not None:
                self.deeplake_dataset.org_id = self.org_id
            if self.token is not None:
                self.deeplake_dataset.set_token(self.token)

            indra_dataset = dataset_to_libdeeplake(self.deeplake_dataset)
        return indra_dataset

    def _collect_return_data(
        self,
        view: DeepLakeDataset,
    ):
        return_data = {}
        for tensor in view.tensors:
            return_data[tensor] = utils.parse_tensor_return(view[tensor])
        return return_data


class SearchManaged(SearchBasic):
    def _get_view(self, tql_query, runtime: Optional[Dict] = None):
        view, data = self.deeplake_dataset.query(
            tql_query, runtime=runtime, return_data=True
        )
        self.data = data
        return view

    def _collect_return_data(
        self,
        view: DeepLakeDataset,
    ):
        return self.data


def search(
    query_embedding: np.ndarray,
    distance_metric: str,
    deeplake_dataset: DeepLakeDataset,
    k: int,
    tql_string: str,
    tql_filter: str,
    embedding_tensor: str,
    runtime: dict,
    return_tensors: List[str],
    return_view: bool = False,
    token: Optional[str] = None,
    org_id: Optional[str] = None,
    return_tql: bool = False,
) -> Union[Dict, DeepLakeDataset]:
    """Generalized search algorithm that uses indra. It combines vector search and other TQL queries.

    Args:
        query_embedding (Optional[Union[List[float], np.ndarray): embedding representation of the query.
        distance_metric (str): Distance metric to compute similarity between query embedding and dataset embeddings
        deeplake_dataset (DeepLakeDataset): DeepLake dataset object.
        k (int): number of samples to return after the search.
        tql_string (str): Standalone TQL query for execution without other filters.
        tql_filter (str): Additional filter using TQL syntax
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        runtime (dict): Runtime parameters for the query.
        return_tensors (List[str]): List of tensors to return data for.
        return_view (bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
        token (Optional[str], optional): Token used for authentication. Defaults to None.
        org_id (Optional[str], optional): Organization ID, is needed only for local datasets. Defaults to None.
        return_tql (bool): Return TQL query used for the search. Defaults to False.

    Raises:
        ValueError: If both tql_string and tql_filter are specified.
        raise_indra_installation_error: If the indra is not installed

    Returns:
        Union[Dict, DeepLakeDataset]: Dictionary where keys are tensor names and values are the results of the search, or a Deep Lake dataset view.
    """
    searcher: SearchBasic
    if runtime and runtime.get("db_engine", False):
        searcher = SearchManaged(deeplake_dataset, org_id, token, runtime=runtime)
    else:
        searcher = SearchIndra(deeplake_dataset, org_id, token)

    return searcher.run(
        tql_string=tql_string,
        return_view=return_view,
        return_tql=return_tql,
        distance_metric=distance_metric,
        k=k,
        query_embedding=query_embedding,
        embedding_tensor=embedding_tensor,
        tql_filter=tql_filter,
        return_tensors=return_tensors,
    )
