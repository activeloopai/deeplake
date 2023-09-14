import os
from typing import Optional, List, Dict, Union, Callable, Tuple

import torch
import numpy as np
from time import time

try:
    from indra import api

    INDRA_AVAILABLE = True
except ImportError:
    INDRA_AVAILABLE = False

from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.client.client import DeepMemoryBackendClient


def deep_memory_available() -> bool:
    # some check whether deepmemory is available
    return True


class DeepMemory:
    def __init__(
        self,
        dataset: Dataset,
        embedding: Optional[Callable] = None,
        token: Optional[str] = None,
    ):
        self.dataset = dataset
        if not INDRA_AVAILABLE:
            raise ImportError(
                "Evaluation on DeepMemory managed service requires the indra package. "
                "Please install it with `pip install deeplake[enterprise]`."
            )
        self.indra_dataset = api.dataset(self.dataset.path)
        api.tql.prepare_deepmemory_metrics(self.indra_dataset)
        self.embedding = embedding
        self.client = DeepMemoryBackendClient(token=token)

    def train(
        self,
        queries: List[str],
        relevances: List[List[Tuple[str, int]]],
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
    ):
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
        response = self.client.cancel_job(job_id=job_id)

    def status(self):
        pass

    def list_jobs(self):
        pass

    def evaluate(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
        run_locally: bool = True,
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
    ):
        if not run_locally:
            raise NotImplementedError(
                "Evaluation on DeepMemory managed service is not yet implemented"
            )
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
                    self.indra_dataset,
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
        run_locally: bool = False,
        top_k: int = 10,
        metric="deepmemory_norm",
    ):
        if not run_locally:
            return NotImplementedError()

        embedding_function = embedding_function or self.embedding.embed_query
        query_emb = embedding_function(embedding_data)

        # Compute the cosine similarity between the query and all data points
        view_top_k = get_view_top_k(
            metric,
            query_emb,
            top_k,
            self.indra_dataset,
        )
        return view_top_k


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
            metric,
            query_emb,
            top_k,
            indra_dataset,
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
):
    # query_emb = embedding_function(query)
    query_emb = ",".join([f"{q}" for q in query_emb])
    tql = f"SELECT * {metric}(embedding, ARRAY[{query_emb}]) order by {metric}(embedding, ARRAY[{query_emb}]) desc limit {top_k}"
    return indra_dataset.query(tql)


def get_deep_memory() -> Optional[DeepMemory]:
    if deep_memory_available():
        return DeepMemory()
    return None
