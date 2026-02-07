"""
TQL API for running queries on data containers.

Tuning Search Accuracy vs Latency
----------------------------------

For vector similarity searches, you can trade latency for higher accuracy by adjusting
search configuration parameters on your dataset:

**For Clustered Indices (cluster_search_accuracy_factor):**
    Controls how many clusters are searched. Higher values search more clusters,
    improving recall at the cost of increased latency.

    - Default: 1.0 (search default number of clusters)
    - Range: 0.5 - 8.0
    - Higher values (e.g., 8.0) = better recall, slower search
    - Lower values (e.g., 0.5) = faster search, lower recall

**For Quantized Indices (accuracy_factor):**
    Controls reranking candidates. When using quantized indices or MMR,
    actual candidates examined = k * accuracy_factor.

    - Default: 10
    - Range: 5 - 20
    - Higher values (e.g., 20) = better accuracy, slower search
    - Lower values (e.g., 5) = faster search, lower accuracy

<!-- test-context
```python
import deeplake
import numpy as np
ds = deeplake.create("tmp://")
ds.add_column("embeddings", deeplake.types.Embedding(128))
ds.add_column("text", deeplake.types.Text())
embeddings_data = [np.random.rand(128).astype(np.float32).tolist() for _ in range(10)]
text_data = [f"sample_{i}" for i in range(10)]
ds.append({"embeddings": embeddings_data, "text": text_data})
```
-->

Examples:
    High accuracy configuration (slower but more accurate):
    ```python
    import numpy as np

    # For clustered indices: search more clusters
    ds.query_config.cluster_search_accuracy_factor = 2.0

    # For quantized indices: examine more candidates
    ds.query_config.accuracy_factor = 20

    # Now run your search
    query_embedding = np.random.rand(128).astype(np.float32).tolist()
    results = ds.query(f"SELECT * ORDER BY cosine_similarity(embeddings, ARRAY{query_embedding}) LIMIT 10")
    ```

    Balanced configuration (default):
    ```python
    ds.query_config.cluster_search_accuracy_factor = 1.0
    ds.query_config.accuracy_factor = 10
    ```

    Fast search configuration (lower recall):
    ```python
    ds.query_config.cluster_search_accuracy_factor = 0.5
    ds.query_config.accuracy_factor = 1
    ```

Note:
    These settings are per-dataset and persist for the lifetime of the dataset object.
    Adjust based on your specific accuracy/latency requirements.
"""

from __future__ import annotations

from typing import Callable, Any

__all__ = [
    "register_function",
    "get_max_num_parallel_queries",
    "set_max_num_parallel_queries",
]

def get_max_num_parallel_queries() -> int:
    """
    Returns the maximum number of parallel queries that can be run, 0 means no limit is set.

    <!-- test-context
    ```python
    import deeplake
    ```
    -->

    Examples:
        ```python
        deeplake.tql.get_max_num_parallel_queries()
        ```
    """

def register_function(function: typing.Callable) -> None:
    """
    Registers the given function in TQL, to be used in queries.
    TQL interacts with Python functions through `numpy.ndarray`. The Python function
    to be used in TQL should accept input arguments as numpy arrays and return numpy array.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("column_name", "int32")
    ds.append({"column_name": [1, 2, 3]})
    ```
    -->

    Examples:
        ```python
        def next_number(a):
            return a + 1

        deeplake.tql.register_function(next_number)

        r = ds.query("SELECT * WHERE next_number(column_name) > 10")
        ```
    """

def set_max_num_parallel_queries(num: int) -> None:
    """
    Sets the maximum number of parallel queries that can be run.

    <!-- test-context
    ```python
    import deeplake
    ```
    -->

    Examples:
        ```python
        deeplake.tql.set_max_num_parallel_queries(8)
        ```
    """
