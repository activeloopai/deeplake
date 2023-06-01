import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error
from deeplake.util.warnings import always_warn

import numpy as np

import random
import string
import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union

EXEC_OPTION_TO_RUNTIME: Dict[str, Optional[Dict]] = {
    "compute_engine": None,
    "python": None,
    "tensor_db": {"db_engine": True},
}


def parse_tensor_return(tensor):
    data = tensor.data()["value"]

    return data.tolist() if isinstance(data, np.ndarray) else data


def check_indra_installation(exec_option, indra_installed):
    if exec_option == "compute_engine" and not indra_installed:
        raise raise_indra_installation_error(
            indra_import_error=False
        )  # pragma: no cover


def get_runtime_from_exec_option(exec_option):
    return EXEC_OPTION_TO_RUNTIME[exec_option]


def check_length_of_each_tensor(tensors):
    tensor_length = len(tensors["text"])

    for tensor_name in tensors:
        if len(tensors[f"{tensor_name}"]) != tensor_length:
            tensor_lengths = create_tensor_to_length_str(tensors)

            raise Exception(
                f"All of the tensors should have equal length. Currently tensors have different length: {tensor_lengths}"
            )


def create_tensor_to_length_str(tensors):
    tensor_lengths = "\n"
    for tensor_name in tensors:
        tensor_lengths += (
            f"length of {tensor_name} = {len(tensors[f'{tensor_name}'])}\n"
        )
    return tensor_lengths


random.seed(0)
np.random.seed(0)


def generate_random_string(length):
    # Define the character set to include letters (both lowercase and uppercase) and digits
    characters = string.ascii_letters + string.digits
    # Generate a random string of the specified length
    random_string = "".join(random.choice(characters) for _ in range(length))

    return random_string


def generate_json(value):
    key = "abc"
    return {key: value}


def create_data(number_of_data, embedding_dim=100):
    embeddings = np.random.uniform(
        low=-10, high=10, size=(number_of_data, embedding_dim)
    ).astype(np.float32)
    texts = [generate_random_string(1000) for i in range(number_of_data)]
    ids = [f"{i}" for i in range(number_of_data)]
    metadata = [generate_json(i) for i in range(number_of_data)]
    return texts, embeddings, ids, metadata


def parse_search_args(**kwargs):
    """Helper function for raising errors if invalid parameters are specified to search"""
    if (
        kwargs["data_for_embedding"] is None
        and kwargs["embedding"] is None
        and kwargs["query"] is None
        and kwargs["filter"] is None
    ):
        raise ValueError(
            f"Either a embedding, data_for_embedding, query, or filter must be specified."
        )

    if (
        kwargs["embedding_function"] is None
        and kwargs["embedding"] is None
        and kwargs["query"] is None
    ):
        raise ValueError(
            f"Either an embedding, embedding_function, or query must be specified."
        )

    exec_option = kwargs["exec_option"]
    if exec_option == "python":
        if kwargs["query"] is not None:
            raise ValueError(
                f"User-specified TQL queries are not support for exec_option={exec_option}."
            )
        if kwargs["query"] is not None:
            raise ValueError(
                f"query parameter for directly running TQL is invalid for exec_option={exec_option}."
            )
        if kwargs["embedding"] is None and kwargs["embedding_function"] is None:
            raise ValueError(
                f"Either emebdding or embedding_function must be specified for exec_option={exec_option}."
            )
    else:
        if type(kwargs["filter"]) == Callable:
            raise ValueError(
                f"UDF filter function are not supported with exec_option={exec_option}"
            )
        if kwargs["query"] and kwargs["filter"]:
            raise ValueError(
                f"query and filter parameters cannot be specified simultaneously."
            )
        if (
            kwargs["embedding"] is None
            and kwargs["embedding_function"] is None
            and kwargs["query"] is None
        ):
            raise ValueError(
                f"Either emebdding, embedding_function, or query must be specified for exec_option={exec_option}."
            )
        if kwargs["return_tensors"] and kwargs["query"]:
            raise ValueError(
                f"return_tensors and query parameters cannot be specified simultaneously, becuase the data that is returned is directly specified in the query."
            )


def check_parameters_compataribility(
    embedding_function, embedding_data, embedding_tensor_name, **kwargs
):
    if embedding_function and not embedding_data:
        raise ValueError(
            f"embedding_data not specified. if embedding_function is specified you also need to specify embedding_data."
        )

    if embedding_function and not embedding_tensor_name:
        raise ValueError(
            f"embedding_tensor_name not specified. if embedding_function is specified you also need to specify embedding_tensor_name."
        )
