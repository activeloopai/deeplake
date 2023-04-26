import deeplake
from deeplake.core.vectorstore import utils, vector_search

import logging
import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable

import numpy as np


logger = logging.getLogger(__name__)


class DeepLakeVectorStore:
    """Base class for DeepLakeVectorStore"""

    _DEFAULT_DEEPLAKE_PATH = "./deeplake_vector_store"

    def __init__(
        self,
        dataset_path: str = _DEFAULT_DEEPLAKE_PATH,
        token: Optional[str] = None,
        embedding_function: Optional[callable] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 4,
        exec_option: str = "python",
        **kwargs: Any,
    ) -> None:
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        creds = {"creds": kwargs["creds"]} if "creds" in kwargs else {}
        self.dataset = utils.create_or_load_dataset(
            dataset_path, token, creds, logger, read_only, **kwargs
        )
        self._embedding_function = embedding_function
        self.exec_option = exec_option

    def add(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[Any] = None,
    ) -> List[str]:
        """Adding elements to deeplake vector store

        Args:
            texts (Iterable[str]): texts to add to deeplake vector store
            metadatas (Optional[List[dict]], optional): List of metadatas.. Defaults to None.
            ids (Optional[List[str]], optional): List of document IDs. Defaults to None.
            embeddings (Optional[Any], optional): embedding of texts. Defaults to None.
        Returns:
            ids (List[str]): List of document IDs
        """
        elements = self._create_elements(ids, texts, metadatas, embeddings)
        self._run_data_injestion(elements)
        self.dataset.commit(allow_empty=True)
        self.dataset.summary()
        return ids

    def query(
        self,
        query: Any[str, None] = None,
        embedding: Any[float, None] = None,
        k: int = 4,
        distance_metric: str = "L2",
        filter: Optional[Any[Dict[str, str], Callable, str]] = None,
        exec_option: Optional[str] = None,
    ):
        view = self._attribute_based_filtering(filter)
        if len(view) == 0:
            return view

        if self._embedding_function is None and embedding is None:
            view, scores, indices = self._exact_text_search(view, query)
        else:
            emb = embedding or self._embedding_function.embed_query(
                query
            )  # type: ignore
            query_emb = np.array(emb, dtype=np.float32)
            embeddings = view.embedding.numpy(fetch_chunks=True)
            indices, scores = vector_search.search(
                query_embedding=query_emb,
                embeddings=embeddings,
                k=k,
                distance_metric=distance_metric.lower(),
                exec_option=exec_option or self._exec_option,
                deeplake_dataset=self.dataset,
            )
            return (view, indices, scores)

    def delete(
        self,
        ids: Any[List[str], None] = None,
        filter: Any[Dict[str, str], None] = None,
        delete_all: Any[bool, None] = None,
    ) -> bool:
        """Delete the entities in the dataset
        Args:
            ids (Optional[List[str]], optional): The document_ids to delete.
                Defaults to None.
            filter (Optional[Dict[str, str]], optional): The filter to delete by.
                Defaults to None.
            delete_all (Optional[bool], optional): Whether to drop the dataset.
                Defaults to None.
        """
        if delete_all:
            self.dataset.delete(large_ok=True)
            return True

        view = None
        if ids:
            view = self.dataset.filter(lambda x: x["ids"].data()["value"] in ids)
            ids = list(view.sample_indices)

        if filter:
            if view is None:
                view = self.dataset
            view = view.filter(partial(utils.dp_filter, filter=filter))
            ids = list(view.sample_indices)

        with self.dataset:
            for id in sorted(ids)[::-1]:
                self.ds.pop(id)
            self.dataset.commit(f"deleted {len(ids)} samples", allow_empty=True)
        return True

    @classmethod
    def force_delete_by_path(cls, path: str) -> None:
        """Force delete dataset by path"""
        try:
            import deeplake
        except ImportError:
            raise ValueError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake`."
            )
        deeplake.delete(path, large_ok=True, force=True)

    def _exact_text_search(self, view, query):
        view = view.filter(lambda x: query in x["text"].data()["value"])
        scores = [1.0] * len(view)
        index = view.index.values[0].value[0]
        return (view, scores, index)

    def _attribute_based_filtering(self, filter):
        view = self.dataset
        # attribute based filtering
        if filter is not None:
            if isinstance(filter, dict):
                filter = partial(utils.dp_filter, filter=filter)

            view = view.filter(filter)
            if len(view) == 0:
                return []
        return view

    def _run_data_injestion(self, elements):
        batch_size = min(self.ingestion_batch_size, len(elements))
        if batch_size == 0:
            return []

        batched = [
            elements[i : i + batch_size] for i in range(0, len(elements), batch_size)
        ]

        self.ingest().eval(
            batched,
            self.dataset,
            num_workers=min(self.num_workers, len(batched) // max(self.num_workers, 1)),
            _embedding_function=self._embedding_function,
        )

    @deeplake.compute
    def ingest(sample_in: list, sample_out: list, _embedding_function) -> None:
        text_list = [s["text"] for s in sample_in]

        embeds = [None] * len(text_list)
        if _embedding_function is not None:
            embeddings = _embedding_function.embed_documents(text_list)
            embeds = [np.array(e, dtype=np.float32) for e in embeddings]

        for s, e in zip(sample_in, embeds):
            embedding = e if _embedding_function else sample_in["embedding"]
            sample_out.append(
                {
                    "text": s["text"],
                    "metadata": s["metadata"],
                    "ids": s["ids"],
                    "embedding": embedding,
                }
            )

    def _create_elements(self, ids, texts, metadatas, embeddings):
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not isinstance(texts, list):
            texts = list(texts)

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if embeddings is None:
            embeddings = [None] * len(texts)

        elements = (
            {"text": text, "metadata": metadata, "id": id_, "embedding": embedding}
            for text, metadata, id_, embedding in zip(texts, metadatas, ids, embeddings)
        )
        return elements
