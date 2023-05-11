import deeplake
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.constants import DEFAULT_DEEPLAKE_PATH
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search.ingestion import data_ingestion

try:
    from indra import api

    _INDRA_INSTALLED = True
except Exception:
    _INDRA_INSTALLED = False

import logging
from typing import Optional, Any, Iterable, List, Dict, Union

import numpy as np


logger = logging.getLogger(__name__)


class DeepLakeVectorStore:
    """Base class for DeepLakeVectorStore"""

    def __init__(
        self,
        dataset_path: str = DEFAULT_DEEPLAKE_PATH,
        token: Optional[str] = None,
        embedding_function: Optional[callable] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 0,
        exec_option: str = "python",
        verbose=False,
        **kwargs: Any,
    ) -> None:
        """DeepLakeVectorStore initialization

        Args:
            dataset_path (str, optional): path to the deeplake dataset. Defaults to DEFAULT_DEEPLAKE_PATH.
            token (str, optional): Activeloop token, used for fetching credentials for Deep Lake datasets. This is Optional, tokens are normally autogenerated. Defaults to None.
            embedding_function (Optional[callable], optional): Function that converts query into embedding. Defaults to None.
            read_only (Optional[bool], optional):  Opens dataset in read only mode if this is passed as True. Defaults to False. Defaults to False.
            ingestion_batch_size (int, optional): The batch size to use during ingestion. Defaults to 1024.
            num_workers (int, optional): The number of workers to use for ingesting data in parallel. Defaults to 0.
            exec_option (str, optional): Type of query execution. It could be either "python", "indra" or "db_engine". Defaults to "python".
        """
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        creds = {"creds": kwargs["creds"]} if "creds" in kwargs else {}
        self.dataset = dataset_utils.create_or_load_dataset(
            dataset_path, token, creds, logger, read_only, exec_option, **kwargs
        )
        self.embedding_function = embedding_function
        self._exec_option = exec_option
        self.verbose = verbose

    def add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        verbose: Optional[bool] = False,
    ) -> List[str]:
        """Adding elements to deeplake vector store

        Args:
            texts (Iterable[str]): texts to add to deeplake vector store
            metadatas (Optional[List[dict]], optional): List of metadatas.. Defaults to None.
            ids (Optional[List[str]], optional): List of document IDs. Defaults to None.
            embeddings (Optional[np.ndarray): embedding of texts. Defaults to None.
        Returns:
            ids (List[str]): List of document IDs
        """
        elements = dataset_utils.create_elements(ids, texts, metadatas, embeddings)
        data_ingestion.run_data_ingestion(
            elements=elements,
            dataset=self.dataset,
            embedding_function=self.embedding_function,
            ingestion_batch_size=self.ingestion_batch_size,
            num_workers=self.num_workers,
        )
        self.dataset.commit(allow_empty=True)
        if self.verbose:
            self.dataset.summary()
        return ids

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        k: int = 4,
        distance_metric: str = "L2",
        filter: Optional[Any] = None,
        exec_option: Optional[str] = None,
    ):
        """DeepLakeVectorStore search method

        Args:
            query (Optional[str], optional): String representation of the query to run. Defaults to None.
            embedding (Optional[np.ndarray, optional): Embedding representation of the query to run. Defaults to None.
            k (int, optional): Number of elements to return after running query. Defaults to 4.
            distance_metric (str, optional): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX". Defaults to "L2".
            filter (Optional[Any], optional): Metadata dictionary for exact search. Defaults to None.
            exec_option (Optional[str], optional): Type of query execution. It could be either "python", "indra" or "db_engine". Defaults to None.

        Raises:
            ValueError: When invalid execution option is specified

        Returns:
            tuple (view, indices, scores): View is the dataset view generated from the queried samples, indices are the indices of the ordered samples, scores are respectively the scores of the ordered samples
        """
        exec_option = exec_option or self._exec_option
        if exec_option not in ("python", "indra", "db_engine"):
            raise ValueError(
                "Invalid `exec_option` it should be either `python`, `indra` or `db_engine`."
            )
        view = filter_utils.attribute_based_filtering(self.dataset, filter, exec_option)
        utils.check_indra_installation(exec_option, indra_installed=_INDRA_INSTALLED)

        if len(view) == 0:
            return view, [], []

        return self._search(
            view=view,
            exec_option=exec_option,
            embedding=embedding,
            query=query,
            k=k,
            distance_metric=distance_metric,
        )

    def _search(
        self,
        view,
        exec_option: str,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        query: Optional[str] = None,
        k: int = 4,
        distance_metric: Optional[str] = "L2",
    ):
        """Internal DeepLakeVectorStore search method

        Args:
            query (Optional[str], optional): String representation of the query to run. Defaults to None.
            embedding (Optional[Union[List[float], np.ndarray]], optional): Embedding representation of the query to run. Defaults to None.
            k (int): Number of elements to return after running query. Defaults to 4.
            distance_metric (str, optional): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX". Defaults to "L2".
            filter (Optional[Any], optional): Metadata dictionary for exact search. Defaults to None.
            exec_option (str): Type of query execution. It could be either "python", "indra" or "db_engine". Defaults to None.

        Returns:
            tuple (view, indices, scores): View is the dataset view generated from the queried samples, indices are the indices of the ordered samples, scores are respectively the scores of the ordered samples
        """
        if self.embedding_function is None and embedding is None:
            view, scores, indices = filter_utils.exact_text_search(view, query)
        else:
            query_emb = dataset_utils.get_embedding(
                embedding, query, embedding_function=self.embedding_function
            )
            exec_option = exec_option or self._exec_option
            embeddings = dataset_utils.fetch_embeddings(
                exec_option=exec_option, view=view, logger=logger
            )

            indices, scores = vector_search.search(
                query_embedding=query_emb,
                embedding=embeddings,
                k=k,
                distance_metric=distance_metric.lower(),
                exec_option=exec_option,
                deeplake_dataset=self.dataset,
            )
        return (view, indices, scores)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, str]] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """Delete the entities in the dataset
        Args:
            ids (Optional[List[str]]): The document_ids to delete.
                Defaults to None.
            filter (Optional[Dict[str, str]]): The filter to delete by.
                Defaults to None.
            delete_all (Optional[bool]): Whether to drop the dataset.
                Defaults to None.
        """
        self.dataset, dataset_deleted = dataset_utils.delete_all_samples_if_specified(
            self.dataset, delete_all
        )
        if dataset_deleted:
            return True

        ids = filter_utils.get_converted_ids(self.dataset, filter, ids)
        dataset_utils.delete_and_commit(self.dataset, ids)
        return True

    @classmethod
    def force_delete_by_path(cls, path: str) -> None:
        """Force delete dataset by path"""
        deeplake.delete(path, large_ok=True, force=True)

    def __len__(self):
        return len(self.dataset)
