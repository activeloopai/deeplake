import os
from typing import Any, Optional, List, Dict, Union, Callable, Tuple

import torch
import numpy as np
from time import time

from deeplake.client import config

config.USE_DEV_ENVIRONMENT = True

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


def deep_memory_available() -> bool:
    # some check whether deepmemory is available
    return True


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
        if not INDRA_AVAILABLE:
            raise ImportError(
                "Evaluation on DeepMemory managed service requires the indra package. "
                "Please install it with `pip install deeplake[enterprise]`."
            )
        self.embedding_function = embedding_function
        self.client = DeepMemoryBackendClient(token=token)

    def train(
        self,
        queries: List[str],
        relevances: List[List[Tuple[str, int]]],
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
        queries_path = os.path.join(corpus_path, "queries")

        queries_vs = VectorStore(
            path=queries_path,
            overwrite=True,
            embedding_function=embedding_function or self.embedding.embed_query,
        )
        queries_vs.add(
            text=[query for query in queries],
            metadata=[{"relevance": relevance} for relevance in relevances],
            embedding_data=[relelvence for _, relelvence in queries.items()],
        )
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

    def list_jobs(self):
        """List all training jobs on DeepMemory managed service."""
        self.client.list_jobs()

    def evaluate(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
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
                    queries,
                    indra_dataset,
                    relevance,
                    top_k=k,
                    query_embs=query_embs,
                    metric=metric,
                )
                print(f"Recall@{k}:\t {100*recall: .1f}%")

    def search(
        self,
        embedding_data=str,
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        return_view: bool = False,
        run_locally: bool = False,
        k: int = 4,
        metric="deepmemory_norm",
    ):
        """Search the dataset on DeepMemory managed service.

        Args:
            embedding_data (str): Query string to search for.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            embedding (Optional[Union[List[float], np.ndarray]], optional): Embedding representation of the query string. Defaults to None.
            filter (Optional[Union[Dict, Callable]], optional): Filter to apply to the dataset. Defaults to None.
            embedding_tensor (str, optional): Name of the tensor in the dataset with `htype = "embedding"`. Defaults to "embedding".
            return_tensors (Optional[List[str]], optional): List of tensors to return data for. Defaults to None.
            return_view (bool, optional): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.
            run_locally (bool, optional): Whether to run the search locally or on the DeepMemory managed service. Defaults to False.
            k (int, optional): Number of samples to return after the search. Defaults to 4.
            metric (str, optional): Distance metric to compute similarity between query embedding and dataset embeddings. Defaults to "deepmemory_norm".

        Returns:
            Union[Dict, DeepLakeDataset]: Dictionary where keys are tensor names and values are the results of the search, or a Deep Lake dataset view.
        """
        if not run_locally:
            return NotImplementedError()

        if callable(filter):
            raise NotImplementedError(
                "UDF filter functions are not supported with the deepmemory yet."
            )

        indra_dataset = api.dataset(self.dataset.path)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        utils.parse_search_args(
            embedding_data=embedding_data,
            embedding_function=embedding_function,
            initial_embedding_function=self.embedding_function,
            embedding=embedding,
            k=k,
            distance_metric="deepmemory_norm",
            query=None,
            filter=filter,
            exec_option="compute_engine",
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
        )

        return_tensors = utils.parse_return_tensors(
            self.dataset, return_tensors, embedding_tensor, return_view
        )

        query_emb = dataset_utils.get_embedding(
            embedding,
            embedding_data,
            embedding_function=embedding_function or self.embedding_function,
        )

        _, tql_filter = filter_utils.attribute_based_filtering_tql(
            view=self.dataset,
            filter=filter,
        )

        # Compute the cosine similarity between the query and all data points
        view_top_k = get_view_top_k(
            metric=metric,
            query_emb=query_emb,
            k=k,
            indra_dataset=indra_dataset,
            return_deeplake_view=True,
            return_tensors=return_tensors,
            tql_filter=tql_filter,
        )
        if return_view:
            return view_top_k

        return_data = {}
        for tensor in view_top_k.tensors:
            if tensor == "indices":
                continue
            return_data[tensor] = utils.parse_tensor_return(view_top_k[tensor])
        return return_data


def recall_at_k(
    queries: torch.Tensor,
    indra_dataset: torch.Tensor,
    relevance: List[List[Tuple[str, int]]],
    query_embs,
    metric,
    top_k: int = 10,
):
    recalls = []

    for query_idx, query in enumerate(queries):
        query_emb = query_embs[query_idx]
        # Get the indices of the relevant data for this query
        query_relevance = relevance[query_idx]
        correct_labels = [label for label, _ in query_relevance]

        # Compute the cosine similarity between the query and all data points
        view_top_k = get_view_top_k(
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
    avg_recall = torch.FloatTensor(recalls).mean()

    return avg_recall


def get_view_top_k(
    metric,
    query_emb,
    top_k,
    indra_dataset,
    dataset,
    return_deeplake_view,
    return_tensors,
    tql_filter="",
):
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    query_emb = ",".join([f"{q}" for q in query_emb])
    tql = f"SELECT * FROM (SELECT {return_tensors}, ROW_NUMBER() as indices) {metric}(embedding, ARRAY[{query_emb}]) {tql_filter_str} order by {metric}(embedding, ARRAY[{query_emb}]) desc limit {top_k}"
    indra_view = indra_dataset.query(tql)
    if return_deeplake_view:
        indices = [[sample.indices.numpy() for sample in indra_view]]
        deeplake_view = dataset[indices]
        return deeplake_view
    return indra_view


def get_deep_memory() -> Optional[DeepMemory]:
    if deep_memory_available():
        return DeepMemory()
    return None
