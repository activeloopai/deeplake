import logging
import pathlib
import uuid
from collections import defaultdict
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
from time import time

import numpy as np

import deeplake
from deeplake.util.exceptions import (
    DeepMemoryAccessError,
    IncorrectRelevanceTypeError,
    IncorrectQueriesTypeError,
)
from deeplake.util.path import convert_pathlib_to_string_if_needed
from deeplake.constants import (
    DEFAULT_QUERIES_VECTORSTORE_TENSORS,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    DEFAULT_DEEPMEMORY_DISTANCE_METRIC,
)
from deeplake.util.storage import get_storage_and_cache_chain
from deeplake.core.dataset import Dataset
from deeplake.core.dataset.deeplake_cloud_dataset import DeepLakeCloudDataset
from deeplake.client.client import DeepMemoryBackendClient
from deeplake.client.utils import JobResponseStatusSchema
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


DEEP_MEMORY_MODEL = "model.npy"
RECALL = "recall@10"


def access_control(func):
    def wrapper(self, *args, **kwargs):
        if self.client is None:
            raise DeepMemoryAccessError()
        return func(self, *args, **kwargs)

    return wrapper


def use_deep_memory(func):
    def wrapper(self, *args, **kwargs):
        use_deep_memory = kwargs.get("deep_memory")
        distance_metric = kwargs.get("distance_metric")

        if use_deep_memory and distance_metric is None:
            kwargs["distance_metric"] = DEFAULT_DEEPMEMORY_DISTANCE_METRIC

        return func(self, *args, **kwargs)

    return wrapper


class Relevance(BaseModel):
    data: List[List[Tuple[str, int]]]


class Queries(BaseModel):
    data: List[str]


def validate_relevance_and_queries(relevance, queries):
    try:
        Relevance(data=relevance)
    except ValidationError:
        raise IncorrectRelevanceTypeError()

    try:
        Queries(data=queries)
    except ValidationError:
        raise IncorrectQueriesTypeError()


class DeepMemory:
    def __init__(
        self,
        dataset: Dataset,
        path: Union[str, pathlib.Path],
        logger: logging.Logger,
        embedding_function: Optional[Any] = None,
        token: Optional[str] = None,
        creds: Optional[Union[Dict, str]] = None,
    ):
        """Based Deep Memory class to train and evaluate models on DeepMemory managed service.

        Args:
            dataset (Dataset): deeplake dataset object or path.
            path (Union[str, pathlib.Path]): Path to the dataset.
            logger (logging.Logger): Logger object.
            embedding_function (Optional[Any], optional): Embedding funtion class used to convert queries/documents to embeddings. Defaults to None.
            token (Optional[str], optional): API token for the DeepMemory managed service. Defaults to None.
            creds (Optional[Dict[str, Any]], optional): Credentials to access the dataset. Defaults to None.

        Raises:
            ImportError: if indra is not installed
        """
        self.dataset = dataset
        self.path = convert_pathlib_to_string_if_needed(path)

        feature_report_path(
            path=self.path,
            feature_name="dm.initialize",
            parameters={
                "embedding_function": True if embedding_function is not None else False,
                "token": token,
            },
            token=token,
        )

        self.token = token
        self.embedding_function = embedding_function
        self.client = self._get_dm_client()
        self.creds = creds or {}
        self.logger = logger

    @access_control
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
                relevence_score is the relevance score of the document for the query. The value is either 0 and 1, where 0 stands for not relevant (unknown relevance)
                and 1 stands for relevant. Currently, only values of 1 contribute to the training, and there is no reason to provide examples with relevance of 0.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            token (str, optional): API token for the DeepMemory managed service. Defaults to None.

        Returns:
            str: job_id of the training job.

        Raises:
            ValueError: if embedding_function is not specified either during initialization or during training.
        """
        from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore

        self.logger.info("Starting DeepMemory training job")
        feature_report_path(
            path=self.path,
            feature_name="dm.train",
            parameters={
                "queries": queries,
                "relevance": relevance,
                "embedding_function": embedding_function,
            },
            token=token or self.token,
        )
        validate_relevance_and_queries(relevance=relevance, queries=queries)

        # TODO: Support for passing query_embeddings directly without embedding function
        corpus_path = self.path
        queries_path = corpus_path + "_queries"

        if embedding_function is None and self.embedding_function is None:
            raise ValueError(
                "Embedding function should be specifed either during initialization or during training."
            )

        if embedding_function is None and self.embedding_function is not None:
            embedding_function = self.embedding_function

        runtime = None
        if get_path_type(corpus_path) == "hub":
            runtime = {"tensor_db": True}

        queries_vs = VectorStore(
            path=queries_path,
            overwrite=True,
            runtime=runtime,
            token=token or self.token,
            creds=self.creds,
            verbose=False,
        )

        self.logger.info("Preparing training data for deepmemory:")
        queries_vs.add(
            text=[query for query in queries],
            metadata=[
                {"relevance": relevance_per_doc} for relevance_per_doc in relevance
            ],
            embedding_data=[query for query in queries],
            embedding_function=embedding_function,
        )

        # do some rest_api calls to train the model
        response = self.client.start_taining(
            corpus_path=corpus_path,
            queries_path=queries_path,
        )

        self.logger.info(
            f"DeepMemory training job started. Job ID: {response['job_id']}"
        )
        return response["job_id"]

    @access_control
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
            path=self.path,
            feature_name="dm.cancel",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )
        return self.client.cancel_job(job_id=job_id)

    @access_control
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
            path=self.path,
            feature_name="dm.delete",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )
        return self.client.delete_job(job_id=job_id)

    @access_control
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
            path=self.path,
            feature_name="dm.status",
            parameters={
                "job_id": job_id,
            },
            token=self.token,
        )

        _, storage = get_storage_and_cache_chain(
            path=self.path,
            db_engine={"tensor_db": True},
            read_only=False,
            creds=self.creds,
            token=self.token,
            memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
        )

        loaded_dataset = DeepLakeCloudDataset(storage=storage)

        try:
            recall, improvement = _get_best_model(
                loaded_dataset.embedding, job_id, latest_job=True
            )

            recall = "{:.2f}".format(100 * recall)
            improvement = "{:.2f}".format(100 * improvement)
        except Exception:
            recall = None
            improvement = None
        self.client.check_status(job_id=job_id, recall=recall, improvement=improvement)

    @access_control
    def list_jobs(self, debug=False):
        """List all training jobs on DeepMemory managed service."""
        feature_report_path(
            path=self.path,
            feature_name="dm.list_jobs",
            parameters={
                "debug": debug,
            },
            token=self.token,
        )
        _, storage = get_storage_and_cache_chain(
            path=self.path,
            db_engine={"tensor_db": True},
            read_only=False,
            creds=self.creds,
            token=self.token,
            memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
        )
        loaded_dataset = DeepLakeCloudDataset(storage=storage)

        response = self.client.list_jobs(
            dataset_path=self.path,
        )

        response_status_schema = JobResponseStatusSchema(response=response)

        jobs = self._get_jobs(response)
        if jobs is None:
            reposnse_str = "No Deep Memory training jobs were found for this dataset"
            print(reposnse_str)
            if debug:
                return reposnse_str
            return None

        recalls = {}
        deltas = {}

        latest_job = jobs[-1]
        for job in jobs:
            try:
                recall, delta = _get_best_model(
                    loaded_dataset.embedding,
                    job,
                    latest_job=job == latest_job,
                )
                recall = "{:.2f}".format(100 * recall)
                delta = "{:.2f}".format(100 * delta)
            except Exception:
                recall = None
                delta = None

            recalls[f"{job}"] = recall
            deltas[f"{job}"] = delta

        reposnse_str = response_status_schema.print_jobs(
            debug=debug, recalls=recalls, improvements=deltas
        )
        return reposnse_str

    @access_control
    def evaluate(
        self,
        relevance: List[List[Tuple[str, int]]],
        queries: List[str],
        embedding_function: Optional[Callable[..., List[np.ndarray]]] = None,
        embedding: Optional[Union[List[np.ndarray], List[List[float]]]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
        qvs_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model using the DeepMemory managed service.

        Examples:
            # 1. Evaluate a model using an embedding function:
            relevance = [[("doc_id_1", 1), ("doc_id_2", 1)], [("doc_id_3", 1)]]
            queries = ["What is the capital of India?", "What is the capital of France?"]
            embedding_function = openai_embedding.embed_documents
            vectorstore.deep_memory.evaluate(
                relevance=relevance,
                queries=queries,
                embedding_function=embedding_function,
            )

            # 2. Evaluate a model with precomputed embeddings:
            embeddings = [[-1.2, 12, ...], ...]
            vectorstore.deep_memory.evaluate(
                relevance=relevance,
                queries=queries,
                embedding=embeddings,
            )

            # 3. Evaluate a model with precomputed embeddings and log queries:
            vectorstore.deep_memory.evaluate(
                relevance=relevance,
                queries=queries,
                embedding=embeddings,
                qvs_params={"log_queries": True},
            )

            # 4. Evaluate with precomputed embeddings, log queries, and a custom branch:
            vectorstore.deep_memory.evaluate(
                relevance=relevance,
                queries=queries,
                embedding=embeddings,
                qvs_params={
                    "log_queries": True,
                    "branch": "queries",
                }
            )

        Args:
            queries (List[str]): Queries for model evaluation.
            relevance (List[List[Tuple[str, int]]]): Relevant documents and scores for each query.
                - Outer list: matches the queries.
                - Inner list: pairs of doc_id and relevance score.
                - doc_id: Document ID from the corpus dataset, found in the `id` tensor.
                - relevance_score: Between 0 (not relevant) and 1 (relevant).
            embedding (Optional[np.ndarray], optional): Query embeddings. Defaults to None.
            embedding_function (Optional[Callable[..., List[np.ndarray]]], optional): Function to convert queries into embeddings. Defaults to None.
            top_k (List[int], optional): Ranks for model evaluation. Defaults to [1, 3, 5, 10, 50, 100].
            qvs_params (Optional[Dict], optional): Parameters to initialize the queries vectorstore. When specified, creates a new vectorstore to track evaluation queries, the Deep Memory response, and the naive vector search results. Defaults to None.

        Returns:
            Dict[str, Dict[str, float]]: Recalls for each rank.

        Raises:
            ImportError: If `indra` is not installed.
            ValueError: If no embedding_function is provided either during initialization or evaluation.
        """
        feature_report_path(
            path=self.path,
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

        validate_relevance_and_queries(relevance=relevance, queries=queries)

        from indra import api  # type: ignore

        indra_dataset = api.dataset(self.path, token=self.token)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        parsed_qvs_params = parse_queries_params(qvs_params)

        start = time()
        query_embs: Union[List[np.ndarray], List[List[float]]]

        if embedding is not None:
            query_embs = embedding
        else:
            if self.embedding_function is not None and embedding_function is None:
                embedding_function = self.embedding_function

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
                {"relevance": relevance_per_doc} for relevance_per_doc in relevance
            ],
            "embedding": query_embs,
            "id": [uuid.uuid4().hex for _ in range(len(queries))],
        }
        for use_model, metric in [
            (False, "COSINE_SIMILARITY"),
            (True, "deepmemory_distance"),
        ]:
            eval_type = "with" if use_model else "without"
            print(f"---- Evaluating {eval_type} Deep Memory ---- ")
            avg_recalls, queries_dict = recall_at_k(
                indra_dataset,
                relevance,
                top_k=top_k,
                query_embs=query_embs,
                metric=metric,
                use_model=use_model,
            )

            queries_data.update(queries_dict)

            for recall, recall_value in avg_recalls.items():
                print(f"Recall@{recall}:\t {100*recall_value: .1f}%")
                recalls[f"{eval_type} model"][f"recall@{recall}"] = recall_value

        log_queries = parsed_qvs_params.get("log_queries")
        branch = parsed_qvs_params.get("branch")

        if log_queries == False:
            return recalls

        self.queries_dataset = deeplake.empty(
            self.path + "_eval_queries",
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

    @access_control
    def get_model(self):
        """Get the name of the model currently being used by DeepMemory managed service."""
        return self.dataset.embedding.info["deepmemory"][DEEP_MEMORY_MODEL]["job_id"]

    @access_control
    def set_model(self, model_name: str):
        """Set model.npy to use `model_name` instead of default model
        Args:
            model_name (str): name of the model to use
        """

        if "npy" not in model_name:
            model_name += ".npy"

        # verify model_name
        self._verify_model_name(model_name)

        # set model.npy to use `model_name` instead of default model
        self._set_model_npy(model_name)

    def _verify_model_name(self, model_name: str):
        if model_name not in self.dataset.embedding.info["deepmemory"]:
            raise ValueError(
                "Invalid model name. Please choose from the following models: "
                + ", ".join(self.dataset.embedding.info["deepmemory"].keys())
            )

    def _set_model_npy(self, model_name: str):
        # get new model.npy
        new_model_npy = self.dataset.embedding.info["deepmemory"][model_name]

        # get old deepmemory dictionary and update it:
        old_deepmemory = self.dataset.embedding.info["deepmemory"]
        new_deepmemory = old_deepmemory.copy()
        new_deepmemory.update({DEEP_MEMORY_MODEL: new_model_npy})

        # assign new deepmemory dictionary to the dataset:
        self.dataset.embedding.info["deepmemory"] = new_deepmemory

    def _get_dm_client(self):
        path = self.path
        path_type = get_path_type(path)

        dm_client = DeepMemoryBackendClient(token=self.token)
        user_profile = dm_client.get_user_profile()

        if path_type == "hub":
            # TODO: add support for windows
            dataset_id = path[6:].split("/")[0]
        else:
            # TODO: change user_profile to user_id
            dataset_id = user_profile["name"]

        deepmemory_is_available = dm_client.deepmemory_is_available(dataset_id)
        if deepmemory_is_available:
            return dm_client
        return None

    def _get_jobs(self, response):
        jobs = None
        if response is not None and len(response) > 0:
            jobs = [job["id"] for job in response]
        return jobs


def recall_at_k(
    indra_dataset: Any,
    relevance: List[List[Tuple[str, int]]],
    query_embs: Union[List[np.ndarray], List[List[float]]],
    metric: str,
    top_k: List[int] = [1, 3, 5, 10, 50, 100],
    use_model: bool = False,
):
    recalls = defaultdict(list)
    top_k_list = []

    for query_idx, _ in enumerate(query_embs):
        query_emb = query_embs[query_idx]
        # Get the indices of the relevant data for this query
        query_relevance = relevance[query_idx]
        correct_labels = [rel[0] for rel in query_relevance]

        # Compute the cosine similarity between the query and all data points
        view = get_view(
            metric=metric,
            query_emb=query_emb,
            indra_dataset=indra_dataset,
        )

        for k in top_k:
            collect_data = k == 10
            view_top_k = view[:k]

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
            recalls[k].append(recall)

    # Average the recalls for each query
    avg_recalls = {
        f"{recall}": np.mean(np.array(recall_list))
        for recall, recall_list in recalls.items()
    }
    model_type = "deep_memory" if use_model else "vector_search"
    queries_data = {
        f"{model_type}_top_10": top_k_list,
        f"{model_type}_recall": recalls[10],
    }
    return avg_recalls, queries_data


def get_view(
    metric: str,
    query_emb: Union[List[float], np.ndarray],
    indra_dataset: Any,
    return_tensors: List[str] = ["text", "metadata", "id"],
    tql_filter: str = "",
):
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    query_emb_str = ",".join([f"{q}" for q in query_emb])
    return_tensors_str = ", ".join(return_tensors)
    tql = f"SELECT * FROM (SELECT {return_tensors_str}, ROW_NUMBER() as indices, {metric}(embedding, ARRAY[{query_emb_str}]) as score {tql_filter_str} order by {metric}(embedding, ARRAY[{query_emb_str}]) desc limit 100)"
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
        try:
            best_recall = info["deepmemory"][DEEP_MEMORY_MODEL][RECALL]
            best_delta = info["deepmemory"][DEEP_MEMORY_MODEL]["delta"]
        except:
            best_recall = info["deepmemory/model.npy"][RECALL]
            best_delta = info["deepmemory/model.npy"]["delta"]
        finally:
            pass

    for job, value in info.items():
        if job_id in job:
            recall = value[RECALL]
            delta = value["delta"]
            if delta > best_delta:
                best_recall = recall
                best_delta = value["delta"]
    return best_recall, best_delta
