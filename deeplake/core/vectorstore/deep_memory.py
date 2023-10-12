import uuid
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
from time import time

import numpy as np

import deeplake
from deeplake.enterprise.dataloader import indra_available
from deeplake.constants import DEFAULT_QUERIES_VECTORSTORE_TENSORS
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.client.client import DeepMemoryBackendClient
from deeplake.client.utils import JobResponseStatusSchema
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


class DeepMemory:
    def __init__(
        self,
        dataset: Dataset,
        client: DeepMemoryBackendClient,
        embedding_function: Optional[Any] = None,
        token: Optional[str] = None,
        creds: Optional[Dict[str, Any]] = None,
    ):
        """Based Deep Memory class to train and evaluate models on DeepMemory managed service.

        Args:
            dataset (Dataset): deeplake dataset object.
            client (DeepMemoryBackendClient): Client to interact with the DeepMemory managed service. Defaults to None.
            embedding_function (Optional[Any], optional): Embedding funtion class used to convert queries/documents to embeddings. Defaults to None.
            token (Optional[str], optional): API token for the DeepMemory managed service. Defaults to None.
            creds (Optional[Dict[str, Any]], optional): Credentials to access the dataset. Defaults to None.

        Raises:
            ImportError: if indra is not installed
        """
        feature_report_path(
            path=dataset.path,
            feature_name="dm.initialize",
            parameters={
                "embedding_function": True if embedding_function is not None else False,
                "client": client,
                "token": token,
            },
            token=token,
        )
        self.dataset = dataset
        self.token = token
        self.embedding_function = embedding_function
        self.client = client
        self.creds = creds or {}

    def train(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        token: Optional[str] = None,
    ) -> str:
        """Train a model on DeepMemory managed service.

        Examples:
            >>> queries: List[str] = ["What is the capital of India?", "What is the capital of France?"]
            >>> relevance: List[List[Tuple[str, int]]] = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            >>> # doc_id_1, doc_id_2, doc_id_3 are the ids of the documents in the corpus dataset that is relevant to the queries. It is stored in the `id` tensor of the corpus dataset.
            >>> job_id: str = vectorstore.deep_memory.train(queries, relevance)

        Args:
            queries (List[str]): List of queries to train the model on.
            relevance (List[List[Tuple[str, int]]]): List of relevant documents for each query with their respective relevance score.
                The outer list corresponds to the queries and the inner list corresponds to the doc_id, relevence_score pair for each query.
                doc_id is the document id in the corpus dataset. It is stored in the `id` tensor of the corpus dataset.
                relevence_score is the relevance score of the document for the query. The range is between 0 and 1, where 0 stands for not relevant and 1 stands for relevant.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            token (str, optional): API token for the DeepMemory managed service. Defaults to None.

        Returns:
            str: job_id of the training job.

        Raises:
            ValueError: if embedding_function is not specified either during initialization or during training.
        """
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.train",
            parameters={
                "queries": queries,
                "relevance": relevance,
                "embedding_function": embedding_function,
            },
            token=token or self.token,
        )
        # TODO: Support for passing query_embeddings directly without embedding function
        corpus_path = self.dataset.path
        queries_path = corpus_path + "_queries"

        if embedding_function is None and self.embedding_function is None:
            raise ValueError(
                "Embedding function should be specifed either during initialization or during training."
            )

        if embedding_function is None and self.embedding_function is not None:
            embedding_function = self.embedding_function.embed_documents

        runtime = None
        if get_path_type(corpus_path) == "hub":
            runtime = {"tensor_db": True}

        queries_vs = VectorStore(
            path=queries_path,
            overwrite=True,
            runtime=runtime,
            embedding_function=embedding_function,
            token=token or self.token,
            creds=self.creds,
        )

        queries_vs.add(
            text=[query for query in queries],
            metadata=[
                {"relevance": relevance_per_doc} for relevance_per_doc in relevance
            ],
            embedding_data=[query for query in queries],
        )

        # do some rest_api calls to train the model
        response = self.client.start_taining(
            corpus_path=corpus_path,
            queries_path=queries_path,
        )

        print(f"DeepMemory training job started. Job ID: {response['job_id']}")
        return response["job_id"]

    def cancel(self, job_id: str):
        """Cancel a training job on DeepMemory managed service.

        Examples:
            >>> cancelled: bool = vectorstore.deep_memory.cancel(job_id)

        Args:
            job_id (str): job_id of the training job.

        Returns:
            bool: True if job was cancelled successfully, False otherwise.
        """
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.cancel",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )
        return self.client.cancel_job(job_id=job_id)

    def delete(self, job_id: str):
        """Delete a training job on DeepMemory managed service.

        Examples:
            >>> deleted: bool = vectorstore.deep_memory.delete(job_id)

        Args:
            job_id (str): job_id of the training job.

        Returns:
            bool: True if job was deleted successfully, False otherwise.
        """
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.delete",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )
        return self.client.delete_job(job_id=job_id)

    def status(self, job_id: str):
        """Get the status of a training job on DeepMemory managed service.

        Examples:
            >>> vectorstore.deep_memory.status(job_id)
            --------------------------------------------------------------
            |                  6508464cd80cab681bfcfff3                  |
            --------------------------------------------------------------
            | status                     | pending                       |
            --------------------------------------------------------------
            | progress                   | None                          |
            --------------------------------------------------------------
            | results                    | not available yet             |
            --------------------------------------------------------------

        Args:
            job_id (str): job_id of the training job.
        """
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.status",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )
        try:
            recall, improvement = _get_best_model(
                self.dataset.embedding, job_id, latest_job=True
            )

            recall = "{:.2f}".format(100 * recall)
            improvement = "{:.2f}".format(100 * improvement)
        except:
            recall = None
            improvement = None
        self.client.check_status(job_id=job_id, recall=recall, improvement=improvement)

    def list_jobs(self, debug=False):
        """List all training jobs on DeepMemory managed service."""
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.list_jobs",
            parameters={
                "debug": debug,
            },
            token=self.token,
        )
        response = self.client.list_jobs(
            dataset_path=self.dataset.path,
        )

        response_status_schema = JobResponseStatusSchema(response=response)

        jobs = [job["id"] for job in response]

        recalls = {}
        deltas = {}

        latest_job = jobs[-1]
        for job in jobs:
            try:
                recall, delta = _get_best_model(
                    self.dataset.embedding,
                    job,
                    latest_job=job == latest_job,
                )
                recall = "{:.2f}".format(100 * recall)
                delta = "{:.2f}".format(100 * delta)
            except:
                recall = None
                delta = None

            recalls[f"{job}"] = recall
            deltas[f"{job}"] = delta

        reposnse_str = response_status_schema.print_jobs(
            debug=debug, recalls=recalls, improvements=deltas
        )
        return reposnse_str

    def evaluate(
        self,
        relevance: List[List[Tuple[str, int]]],
        queries: List[str],
        embedding_function: Optional[Callable[..., List[np.ndarray]]] = None,
        embedding: Optional[Union[List[np.ndarray], List[List[float]]]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
        qvs_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a model on DeepMemory managed service.

        Examples:
            >>> #1. Evaluate a model with embedding function
            >>> relevance: List[List[Tuple[str, int]]] = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            >>> # doc_id_1, doc_id_2, doc_id_3 are the ids of the documents in the corpus dataset that is relevant to the queries. It is stored in the `id` tensor of the corpus dataset.
            >>> queries: List[str] = ["What is the capital of India?", "What is the capital of France?"]
            >>> embedding_function: Callable[..., List[np.ndarray] = openai_embedding.embed_documents
            >>> vectorstore.deep_memory.evaluate(
            ...     relevance=relevance,
            ...     queries=queries,
            ...     embedding_function=embedding_function,
            ... )
            >>> #2. Evaluate a model with precomputed embeddings
            >>> relevance: List[List[Tuple[str, int]]] = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            >>> # doc_id_1, doc_id_2, doc_id_3 are the ids of the documents in the corpus dataset that is relevant to the queries. It is stored in the `id` tensor of the corpus dataset.
            >>> queries: List[str] = ["What is the capital of India?", "What is the capital of France?"]
            >>> embedding: Union[List[np.ndarray[Any, Any]], List[List[float]] = [[-1.2, 12, ...], ...]
            >>> vectorstore.deep_memory.evaluate(
            ...     relevance=relevance,
            ...     queries=queries,
            ...     embedding=embedding,
            ... )
            >>> #3. Evaluate a model with precomputed embeddings and log queries
            >>> relevance: List[List[Tuple[str, int]]] = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            >>> # doc_id_1, doc_id_2, doc_id_3 are the ids of the documents in the corpus dataset that is relevant to the queries. It is stored in the `id` tensor of the corpus dataset.
            >>> queries: List[str] = ["What is the capital of India?", "What is the capital of France?"]
            >>> embedding: Union[List[np.ndarray[Any, Any]], List[List[float]] = [[-1.2, 12, ...], ...]
            >>> vectorstore.deep_memory.evaluate(
            ...     relevance=relevance,
            ...     queries=queries,
            ...     embedding=embedding,
            ...     qvs_params={
            ...         "log_queries": True,
            ...     }
            ... )
            >>> #4. Evaluate a model with precomputed embeddings and log queries, and custom branch
            >>> relevance: List[List[Tuple[str, int]]] = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            >>> # doc_id_1, doc_id_2, doc_id_3 are the ids of the documents in the corpus dataset that is relevant to the queries. It is stored in the `id` tensor of the corpus dataset.
            >>> queries: List[str] = ["What is the capital of India?", "What is the capital of France?"]
            >>> embedding: Union[List[np.ndarray[Any, Any]], List[List[float]] = [[-1.2, 12, ...], ...]
            >>> vectorstore.deep_memory.evaluate(
            ...     relevance=relevance,
            ...     queries=queries,
            ...     embedding=embedding,
            ...     qvs_params={
            ...         "log_queries": True,
            ...         "branch": "queries",
            ...     }
            ... )

        Args:
            queries (List[str]): List of queries to evaluate the model on.
            relevance (List[List[Tuple[str, int]]]): List of relevant documents for each query with their respective relevance score.
                The outer list corresponds to the queries and the inner list corresponds to the doc_id, relevence_score pair for each query.
                doc_id is the document id in the corpus dataset. It is stored in the `id` tensor of the corpus dataset.
                relevence_score is the relevance score of the document for the query. The range is between 0 and 1, where 0 stands for not relevant and 1 stands for relevant.
            embedding (Optional[np.ndarray], optional): Embedding of the queries. Defaults to None.
            embedding_function (Optional[Callable[..., List[np.ndarray]]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            top_k (List[int], optional): List of top_k values to evaluate the model on. Defaults to [1, 3, 5, 10, 50, 100].
            qvs_params (Optional[Dict], optional): Parameters to initialize the queries vectorstore. Defaults to None.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of recalls for each top_k value.

        Raises:
            ImportError: if indra is not installed
            ValueError: if embedding_function is not specified either during initialization or during evaluation.
        """
        feature_report_path(
            path=self.dataset.path,
            feature_name="dm.evaluate",
            parameters={
                "relevance": relevance,
                "queries": queries,
                "embedding_function": embedding_function,
                "embedding": embedding,
                "top_k": top_k,
                "qvs_params": qvs_params,
            },
            token=self.token,
        )
        try:
            from indra import api  # type: ignore

            INDRA_INSTALLED = True
        except Exception:
            INDRA_INSTALLED = False

        if not INDRA_INSTALLED:
            raise ImportError(
                "indra is not installed. Please install indra to use this functionality with: pip install `deeplake[enterprise]`"
            )

        from indra import api  # type: ignore

        indra_dataset = api.dataset(self.dataset.path, token=self.token)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        parsed_qvs_params = parse_queries_params(qvs_params)

        start = time()
        query_embs: Union[List[np.ndarray], List[List[float]]]
        if embedding is not None:
            query_embs = embedding
        elif embedding is None:
            if self.embedding_function is not None:
                embedding_function = (
                    embedding_function or self.embedding_function.embed_documents
                )

            if embedding_function is None:
                raise ValueError(
                    "Embedding function should be specifed either during initialization or during evaluation."
                )
            query_embs = embedding_function(queries)

        print(f"Embedding queries took {time() - start:.2f} seconds")

        recalls: Dict[str, Dict] = {"with model": {}, "without model": {}}
        queries_data = {
            "text": queries,
            "metadata": [
                {"relvence": relevance_per_doc} for relevance_per_doc in relevance
            ],
            "embedding": query_embs,
            "id": [uuid.uuid4().hex for _ in range(len(queries))],
        }
        for use_model, metric in [
            (False, "COSINE_SIMILARITY"),
            (True, "deepmemory_distance"),
        ]:
            eval_type = "with" if use_model else "without"
            print(f"---- Evaluating {eval_type} model ---- ")
            callect_data = False
            for k in top_k:
                callect_data = k == 10

                recall, queries_dict = recall_at_k(
                    self.dataset,
                    indra_dataset,
                    relevance,
                    top_k=k,
                    query_embs=query_embs,
                    metric=metric,
                    collect_data=callect_data,
                    use_model=use_model,
                )

                if callect_data:
                    queries_data.update(queries_dict)

                print(f"Recall@{k}:\t {100*recall: .1f}%")
                recalls[f"{eval_type} model"][f"recall@{k}"] = recall

        log_queries = parsed_qvs_params.get("log_queries")
        branch = parsed_qvs_params.get("branch")

        if log_queries == False:
            return recalls

        self.queries_dataset = deeplake.empty(
            self.dataset.path + "_eval_queries",
            token=self.token,
            creds=self.creds,
            overwrite=True,
        )

        if len(self.queries_dataset) == 0:
            self.queries_dataset.commit(allow_empty=True)

        create = branch not in self.queries_dataset.branches
        self.queries_dataset.checkout(parsed_qvs_params["branch"], create=create)

        for tensor_params in DEFAULT_QUERIES_VECTORSTORE_TENSORS:
            if tensor_params["name"] not in self.queries_dataset.tensors:
                self.queries_dataset.create_tensor(**tensor_params)

        self.queries_dataset.extend(queries_data, progressbar=True)
        self.queries_dataset.commit()
        return recalls


def recall_at_k(
    dataset: Dataset,
    indra_dataset: Any,
    relevance: List[List[Tuple[str, int]]],
    query_embs: Union[List[np.ndarray], List[List[float]]],
    metric: str,
    top_k: int = 10,
    collect_data: bool = False,
    use_model: bool = False,
):
    recalls = []
    top_k_list = []

    for query_idx, _ in enumerate(query_embs):
        query_emb = query_embs[query_idx]
        # Get the indices of the relevant data for this query
        query_relevance = relevance[query_idx]
        correct_labels = [rel[0] for rel in query_relevance]

        # Compute the cosine similarity between the query and all data points
        view_top_k = get_view_top_k(
            metric=metric,
            query_emb=query_emb,
            top_k=top_k,
            indra_dataset=indra_dataset,
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

        if collect_data:
            top_k_list.append(top_k_retrieved)
        recalls.append(recall)

    # Average the recalls for each query
    avg_recall = np.mean(np.array(recalls))
    queries_data = {}
    if collect_data:
        model_type = "deep_memory" if use_model else "vector_search"

        queries_data = {
            f"{model_type}_top_10": top_k_list,
            f"{model_type}_recall": recalls,
        }
    return avg_recall, queries_data


def get_view_top_k(
    metric: str,
    query_emb: Union[List[float], np.ndarray],
    top_k: int,
    indra_dataset: Any,
    return_tensors: List[str] = ["text", "metadata", "id"],
    tql_filter: str = "",
):
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    query_emb_str = ",".join([f"{q}" for q in query_emb])
    return_tensors_str = ", ".join(return_tensors)
    tql = f"SELECT * FROM (SELECT {return_tensors_str}, ROW_NUMBER() as indices, {metric}(embedding, ARRAY[{query_emb_str}]) as score {tql_filter_str} order by {metric}(embedding, ARRAY[{query_emb_str}]) desc limit {top_k})"
    indra_view = indra_dataset.query(tql)
    return indra_view


def parse_queries_params(queries_params: Optional[Dict[str, Any]] = None):
    if queries_params is not None:
        log_queries = queries_params.get("log_queries")
        if log_queries is None:
            queries_params["log_queries"] = True

    if queries_params is None:
        queries_params = {
            "log_queries": False,
        }

    for query_param in queries_params:
        if query_param not in ["log_queries", "branch"]:
            raise ValueError(
                f"Invalid query param '{query_param}'. Valid query params are 'log_queries' and 'branch'."
            )

    if queries_params.get("log_queries") and not queries_params.get("branch"):
        queries_params = {
            "log_queries": True,
            "branch": "main",
        }

    return queries_params


def _get_best_model(embedding: Any, job_id: str, latest_job: bool = False):
    info = embedding.info
    best_recall = 0
    best_delta = 0
    if latest_job:
        best_recall = info["deepmemory/model.npy"]["recall@10"]
        best_delta = info["deepmemory/model.npy"]["delta"]

    for job, value in info.items():
        if job_id in job:
            recall = value["recall@10"]
            delta = value["delta"]
            if delta > best_delta:
                best_recall = recall
                best_delta = value["delta"]
    return best_recall, best_delta
