import os
from typing import Any, Optional, List, Dict, Union, Callable, Tuple
import textwrap

import numpy as np
from time import time

from deeplake.client import config
from deeplake.core.vectorstore.vector_search import vector_search

try:
    from indra import api

    INDRA_AVAILABLE = True
except ImportError:
    INDRA_AVAILABLE = False

from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.client.client import DeepMemoryBackendClient
from deeplake.core.vectorstore.vector_search import filter as filter_utils


class DeepMemory:
    def __init__(
        self,
        dataset: Dataset,
        embedding_function: Optional[Callable] = None,
        token: Optional[str] = None,
    ):
        """Based Deep Memory class to train and evaluate models on DeepMemory managed service.

        Args:
            dataset (Dataset): deeplake dataset object.
            embedding_function (Optional[Callable], optional): Embedding funtion used to convert queries/documents to embeddings. Defaults to None.
            token (Optional[str], optional): API token for the DeepMemory managed service. Defaults to None.

        Raises:
            ImportError: if indra is not installed
        """
        self.dataset = dataset
        self.embedding_function = embedding_function
        self.client = DeepMemoryBackendClient(token=token)

    def train(
        self,
        queries: List[str],
        relevances: List[List[Tuple[str, int]]],
        query_embeddings: Optional[List[np.ndarray]] = None,
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
    ):
        """Train a model on DeepMemory managed service.

        Args:
            queries (List[str]): List of queries to train the model on.
            relevances (List[List[Tuple[str, int]]]): List of relevant documents for each query.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.

        Returns:
            str: job_id of the training job.
        """
        corpus_path = self.dataset.path
        queries_path = corpus_path + "_queries"

        queries_vs = VectorStore(
            path=queries_path,
            overwrite=True,
            runtime={"tensor_db": True},
            embedding_function=embedding_function
            or self.embedding_function.embed_documents,
        )

        add_kwargs = {
            "text": [query for query in queries],
            "metadata": [
                {"relevance": [(doc_id, 1) for doc_id in relevance]}
                for relevance in relevances
            ],
        }

        if query_embeddings:
            add_kwargs["embedding"] = query_embeddings
        else:
            add_kwargs["embedding_data"] = [query for query in queries]
        queries_vs.add(**add_kwargs)
        # do some rest_api calls to train the model
        response = self.client.start_taining(
            corpus_path=corpus_path,
            queries_path=queries_path,
        )
        return response["job_id"]

    def cancel(self, job_id: str):
        """Cancel a training job on DeepMemory managed service.

        Args:
            job_id (str): job_id of the training job.
        """
        try:
            self.client.cancel_job(job_id=job_id)
            print(f"Job with job_id='{job_id}' was sucessfully cancelled!")
        except Exception as e:
            print(f"Job with job_id='{job_id}' was not cancelled! Error: {e}")

    def status(self, job_id: Union[str, List[str]]):
        """Get the status of a training job on DeepMemory managed service.

        Args:
            job_id (Union[str, List[str]]): job_id of the training job.
        """
        self.client.check_status(job_id=job_id)

    @classmethod
    def list_jobs(self, dataset_path, token=None):
        """List all training jobs on DeepMemory managed service."""
        client = DeepMemoryBackendClient(token=token)
        response = client.list_jobs(dataset_path=dataset_path)
        return response.json()

    def evaluate(
        self,
        queries: List[str],
        relevances: List[List[Tuple[str, int]]],
        run_locally: bool = True,
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
    ):
        """Evaluate a model on DeepMemory managed service.

        Args:
            queries (List[str]): List of queries to evaluate the model on.
            relevance (List[List[Tuple[str, int]]]): List of relevant documents for each query.
            run_locally (bool, optional): Whether to run the evaluation locally or on the DeepMemory managed service. Defaults to True.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            top_k (List[int], optional): List of top_k values to evaluate the model on. Defaults to [1, 3, 5, 10, 50, 100].
        """
        if not run_locally:
            raise NotImplementedError(
                "Evaluation on DeepMemory managed service is not yet implemented"
            )

        if not INDRA_AVAILABLE:
            raise ImportError(
                "Evaluation on DeepMemory managed service requires the indra package. "
                "Please install it with `pip install deeplake[enterprise]`."
            )

        indra_dataset = api.dataset(self.dataset.path)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        # TODO: validate user permissions
        start = time()
        query_embs = embedding_function(queries)
        print(f"Embedding queries took {time() - start:.2f} seconds")

        for use_model, metric in [
            (False, "COSINE_SIMILARITY"),
            (True, "deepmemory_norm"),
        ]:
            print(f"---- Evaluating {'with' if use_model else 'without'} model ---- ")
            for k in top_k:
                recall = recall_at_k(
                    self.dataset,
                    queries,
                    indra_dataset,
                    relevances,
                    top_k=k,
                    query_embs=query_embs,
                    metric=metric,
                )
                print(f"Recall@{k}:\t {100*recall: .1f}%")


def recall_at_k(
    dataset: Dataset,
    queries: List[str],
    indra_dataset: Any,
    relevances: List[List[Tuple[str, int]]],
    query_embs,
    metric,
    top_k: int = 10,
):
    recalls = []

    for query_idx, query in enumerate(queries):
        query_emb = query_embs[query_idx]
        # Get the indices of the relevant data for this query
        query_relevance = relevances[query_idx]
        correct_labels = [label for label, _ in query_relevance]

        # Compute the cosine similarity between the query and all data points
        view_top_k = get_view_top_k(
            dataset=dataset,
            metric=metric,
            query_emb=query_emb,
            top_k=top_k,
            indra_dataset=indra_dataset,
            return_deeplake_view=False,
        )

        top_k_retrieved = [
            sample.id.numpy() for sample in view_top_k
        ]  # TODO: optimize this

        # Compute the recall: the fraction of relevant items found in the top k
        num_relevant_in_top_k = len(
            set(correct_labels).intersection(set(top_k_retrieved))
        )
        if len(correct_labels) == 0:
            continue
        recall = num_relevant_in_top_k / len(correct_labels)
        recalls.append(recall)

    # Average the recalls for each query
    avg_recall = np.mean(np.array(recalls))

    return avg_recall


def get_view_top_k(
    metric,
    query_emb,
    top_k,
    indra_dataset,
    dataset,
    return_deeplake_view,
    return_tensors=["text", "metadata", "id"],
    tql_filter="",
):
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    query_emb = ",".join([f"{q}" for q in query_emb])
    return_tensors = ", ".join(return_tensors)
    tql = f"SELECT * FROM (SELECT {return_tensors}, ROW_NUMBER() as indices, {metric}(embedding, ARRAY[{query_emb}]) as score {tql_filter_str} order by {metric}(embedding, ARRAY[{query_emb}]) desc limit {top_k})"
    indra_view = indra_dataset.query(tql)
    if return_deeplake_view:
        indices = [[sample.indices.numpy() for sample in indra_view]]
        deeplake_view = dataset[indices]
        return deeplake_view
    return indra_view
