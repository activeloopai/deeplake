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


class DeepMemory:
    def __init__(
        self,
        dataset: Dataset,
        embedding_function: Optional[Any] = None,
        token: Optional[str] = None,
    ):
        """Based Deep Memory class to train and evaluate models on DeepMemory managed service.

        Args:
            dataset (Dataset): deeplake dataset object.
            embedding_function (Optional[Any], optional): Embedding funtion class used to convert queries/documents to embeddings. Defaults to None.
            token (Optional[str], optional): API token for the DeepMemory managed service. Defaults to None.

        Raises:
            ImportError: if indra is not installed
        """
        self.dataset = dataset
        self.token = token
        self.embedding_function = embedding_function
        self.client = DeepMemoryBackendClient(token=token)
        self.queries_dataset = deeplake.dataset(
            self.dataset.path + "_queries",
            token=token,
            read_only=False,
        )
        if len(self.queries_dataset) == 0:
            self.queries_dataset.commit(allow_empty=True)

    def train(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        token: Optional[str] = None,
    ):
        """Train a model on DeepMemory managed service.

        Args:
            queries (List[str]): List of queries to train the model on.
            relevance (List[List[Tuple[str, int]]]): List of relevant documents for each query.
            embedding_function (Optional[Callable[[str], np.ndarray]], optional): Embedding funtion used to convert queries to embeddings. Defaults to None.
            token (str, optional): API token for the DeepMemory managed service. Defaults to None.

        Returns:
            str: job_id of the training job.

        Raises:
            ValueError: if embedding_function is not specified either during initialization or during training.
        """
        # TODO: Support for passing query_embeddings directly without embedding function
        corpus_path = self.dataset.path
        queries_path = corpus_path + "_queries"

        if embedding_function is None and self.embedding_function is None:
            raise ValueError(
                "Embedding function should be specifed either during initialization or during training."
            )

        if embedding_function is None and self.embedding_function is not None:
            embedding_function = self.embedding_function.embed_documents

        queries_vs = VectorStore(
            path=queries_path,
            overwrite=True,
            runtime={"tensor_db": True},
            embedding_function=embedding_function,
            token=token,
        )

        queries_vs.add(
            text=[query for query in queries],
            metadata=[
                {"relevance": [(doc_id, 1) for doc_id in relevance]}
                for relevance in relevance
            ],
            embedding_data=[query for query in queries],
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

        Returns:
            bool: True if job was cancelled successfully, False otherwise.
        """
        try:
            self.client.cancel_job(job_id=job_id)
            print(f"Job with job_id='{job_id}' was sucessfully cancelled!")
            return True
        except Exception as e:
            print(f"Job with job_id='{job_id}' was not cancelled! Error: {e}")
            return False

    def delete(self, job_id: str):
        """Delete a training job on DeepMemory managed service.

        Args:
            job_id (str): job_id of the training job.

        Returns:
            bool: True if job was deleted successfully, False otherwise.
        """
        try:
            self.client.delete_job(job_id=job_id)
            print(f"Job with job_id='{job_id}' was sucessfully deleted!")
            return True
        except Exception as e:
            print(f"Job with job_id='{job_id}' was not deleted! Error: {e}")
            return False

    def status(self, job_id: str):
        """Get the status of a training job on DeepMemory managed service.

        Args:
            job_id (str): job_id of the training job.
        """
        self.client.check_status(job_id=job_id)

    def list_jobs(self):
        """List all training jobs on DeepMemory managed service."""
        response = self.client.list_jobs(dataset_path=self.dataset.path)
        return response

    def evaluate(
        self,
        relevance: List[List[str]],
        queries: List[str],
        embedding_function: Optional[Callable[..., List[np.ndarray]]] = None,
        embedding: Optional[
            Union[List[np.ndarray[Any, Any]], List[List[float]]]
        ] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
        qvs_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a model on DeepMemory managed service.

        Args:
            queries (List[str]): List of queries to evaluate the model on.
            relevance (List[List[Tuple[str, int]]]): List of relevant documents for each query.
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
        try:
            indra_available()
        except ImportError:
            raise ImportError(
                "indra is not installed. Please install indra to use this functionality."
            )

        from indra import api  # type: ignore

        indra_dataset = api.dataset(
            self.dataset.path, token=self.token
        )  # somehow wheel is not working when used with token
        # indra_dataset = api.dataset(self.dataset.path)
        api.tql.prepare_deepmemory_metrics(indra_dataset)

        parsed_qvs_params = parse_queries_params(qvs_params)

        start = time()
        query_embs: Union[List[np.ndarray], List[List[float]]]
        if embedding is not None:
            query_embs = embedding
        elif embedding is None:
            if embedding_function is not None:
                query_embs = embedding_function(queries)
            else:
                raise ValueError(
                    "Embedding function should be specifed either during initialization or during evaluation."
                )
        print(f"Embedding queries took {time() - start:.2f} seconds")

        recalls: Dict[str, Dict] = {"with model": {}, "without model": {}}
        queries_data = {
            "text": queries,
            "metadata": [
                {"relvence": [rel for rel in rel_list]} for rel_list in relevance
            ],
            "embedding": query_embs,
            "id": [uuid.uuid4().hex for i in range(len(queries))],
        }
        for use_model, metric in [
            (False, "COSINE_SIMILARITY"),
            (True, "deepmemory_norm"),
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

        if not log_queries:
            return recalls

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
    relevance: List[List[str]],
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
        correct_labels = [label for label in query_relevance]

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

        if collect_data:
            top_k_list.append(top_k_retrieved)
        recalls.append(recall)

    # Average the recalls for each query
    avg_recall = np.mean(np.array(recalls))
    queries_data = {}
    if collect_data:
        model_type = "dm" if use_model else "naive_vs"

        queries_data = {
            f"{model_type}_top_10_docs": top_k_list,
            f"{model_type}_recall": recalls,
        }
    return avg_recall, queries_data


def get_view_top_k(
    metric: str,
    query_emb: Union[List[float], np.ndarray],
    top_k: int,
    indra_dataset: Any,
    dataset: Dataset,
    return_deeplake_view: bool,
    return_tensors: List[str] = ["text", "metadata", "id"],
    tql_filter: str = "",
):
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    query_emb_str = ",".join([f"{q}" for q in query_emb])
    return_tensors_str = ", ".join(return_tensors)
    tql = f"SELECT * FROM (SELECT {return_tensors_str}, ROW_NUMBER() as indices, {metric}(embedding, ARRAY[{query_emb_str}]) as score {tql_filter_str} order by {metric}(embedding, ARRAY[{query_emb_str}]) desc limit {top_k})"
    indra_view = indra_dataset.query(tql)
    if return_deeplake_view:
        indices = [[sample.indices.numpy() for sample in indra_view]]
        deeplake_view = dataset[indices]
        return deeplake_view
    return indra_view


def parse_queries_params(queries_params: Optional[Dict[str, Any]] = None):
    if queries_params is None or (
        queries_params["log_queries"] and not queries_params.get("branch")
    ):
        return {
            "log_queries": True,
            "branch": "main",
        }
    return queries_params
