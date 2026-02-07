"""TQL (Tensor Query Language) module for querying datasets."""

from ._deeplake.tql import (
    register_function,
    get_max_num_parallel_queries,
    set_max_num_parallel_queries,
)

__all__ = [
    "register_function",
    "get_max_num_parallel_queries",
    "set_max_num_parallel_queries",
]
