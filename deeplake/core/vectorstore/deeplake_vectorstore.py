import deeplake
from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utlils
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
        **kwargs: Any,
    ) -> None:
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        creds = {"creds": kwargs["creds"]} if "creds" in kwargs else {}
        self.dataset = dataset_utils.create_or_load_dataset(
            dataset_path, token, creds, logger, read_only, exec_option, **kwargs
        )
        self._embedding_function = embedding_function
        self._exec_option = exec_option

    def add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[Union[List[float], np.ndarray]] = None,
        verbose: Optional[bool] = False,
    ) -> List[str]:
        """Adding elements to deeplake vector store

        Args:
            texts (Iterable[str]): texts to add to deeplake vector store
            metadatas (Optional[List[dict]], optional): List of metadatas.. Defaults to None.
            ids (Optional[List[str]], optional): List of document IDs. Defaults to None.
            embeddings (Optional[Union[List[float], np.ndarray]]): embedding of texts. Defaults to None.
        Returns:
            ids (List[str]): List of document IDs
        """
        elements = dataset_utils.create_elements(ids, texts, metadatas, embeddings)
        data_ingestion.run_data_ingestion(
            elements=elements,
            dataset=self.dataset,
            embedding_function=self._embedding_function,
            ingestion_batch_size=self.ingestion_batch_size,
            num_workers=self.num_workers,
        )
        self.dataset.commit(allow_empty=True)
        if verbose:
            self.dataset.summary()
        return ids

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: str = "L2",
        filter: Optional[Any] = None,
        exec_option: Optional[str] = None,
        db_engine: bool = False,
    ):
        exec_option = self._parse_exec_option(
            exec_option=exec_option, db_engine=db_engine
        )

        # TO DO:
        # 1. check filter with indra

        view = filter_utlils.attribute_based_filtering(self.dataset, filter)
        utils.check_indra_installation(exec_option, indra_installed=_INDRA_INSTALLED)

        if len(view) == 0:
            return view

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
        exec_option: bool,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        query: Optional[str] = None,
        k: Optional[int] = 4,
        distance_metric: Optional[str] = "L2",
    ):
        if self._embedding_function is None and embedding is None:
            view, scores, indices = filter_utlils.exact_text_search(view, query)
        else:
            query_emb = dataset_utils.get_embedding(embedding, query)
            exec_option = exec_option or self._exec_option
            embeddings = dataset_utils.fetch_embeddings(
                exec_option=exec_option, view=view
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

    def _parse_exec_option(self, exec_option, db_engine=None):
        if db_engine == True:
            return "db_engine"
        return exec_option or self._exec_option

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
        self.dataset, deleted = dataset_utils.delete_all_samples_if_specified(
            self.dataset, delete_all
        )
        if deleted:
            return True

        ids = filter_utlils.get_id_indices(self.dataset, ids)
        ids = filter_utlils.get_filtered_ids(self.dataset, filter, ids)
        dataset_utils.delete_and_commit(self.dataset, ids)
        return True

    @classmethod
    def force_delete_by_path(cls, path: str) -> None:
        """Force delete dataset by path"""
        deeplake.delete(path, large_ok=True, force=True)

    def __len__(self):
        return len(self.dataset)
