"""
TQL api, to run queries on data containers.
"""

from __future__ import annotations

from typing import Callable, Any

__all__ = [
    "get_max_num_parallel_queries",
    "register_function",
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
