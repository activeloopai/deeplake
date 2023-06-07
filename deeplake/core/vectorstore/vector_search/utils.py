from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

import numpy as np

import random
import string
from typing import Optional, List, Dict, Callable

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
    first_item = next(iter(tensors))
    tensor_length = len(tensors[first_item])

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
    images = ["deeplake/tests/dummy_data/images/car.jpg" for i in range(number_of_data)]
    return texts, embeddings, ids, metadata, images


def parse_search_args(**kwargs):
    """Helper function for raising errors if invalid parameters are specified to search"""

    if kwargs["exec_option"] not in ("python", "compute_engine", "tensor_db"):
        raise ValueError(
            "Invalid `exec_option` it should be either `python`, `compute_engine` or `tensor_db`."
        )

    if (
        kwargs["embedding_function"] is None
        and kwargs["embedding"] is None
        and kwargs["query"] is None
        and kwargs["filter"] is None
    ):
        raise ValueError(
            f"Either an `embedding`, `embedding_function`, `filter`, or `query` must be specified."
        )
    if kwargs["embedding_data"] is None and kwargs["embedding_function"] is not None:
        raise ValueError(
            f"When an `embedding_function` is specified, `embedding_data` must also be specified."
        )

    exec_option = kwargs["exec_option"]
    if exec_option == "python":
        if kwargs["query"] is not None:
            raise ValueError(
                f"User-specified TQL queries are not support for exec_option={exec_option}."
            )

    else:
        if kwargs["query"] and kwargs["filter"]:
            raise ValueError(
                f"`query` and `filter` parameters cannot be specified simultaneously."
            )

        if kwargs["return_tensors"] and kwargs["query"]:
            raise ValueError(
                f"return_tensors and query parameters cannot be specified simultaneously, becuase the data that is returned is directly specified in the query."
            )


def parse_add_arguments(
    dataset,
    embedding_function=None,
    initial_embedding_function=None,
    embedding_data=None,
    embedding_tensor=None,
    **tensors,
):
    """Parse the input argument to the Vector Store add function to infer whether they are a valid combination."""

    if embedding_function:
        if not embedding_data:
            raise ValueError(
                f"embedding_data is not specified. When using embedding_function it is also necessary to specify the data that you want to embed"
            )

        embedding_tensor = find_embedding_tensor(embedding_tensor, tensors, dataset)

        check_tensor_name_consistency(tensors, dataset.tensors, embedding_tensor)
        return (embedding_function, embedding_data, embedding_tensor, tensors)

    if initial_embedding_function:
        if not embedding_data:
            check_tensor_name_consistency(tensors, dataset.tensors, None)
            return (None, None, None, tensors)

        embedding_tensor = find_embedding_tensor(embedding_tensor, tensors, dataset)

        check_tensor_name_consistency(tensors, dataset.tensors, embedding_tensor)

        return (initial_embedding_function, embedding_data, embedding_tensor, tensors)

    if embedding_tensor:
        raise ValueError(
            f"`embedding_tensor` is specified while `embedding_function` is not specified. "
            "Either specify `embedding_function` during Vector Store initialization or during `add` call."
        )

    if embedding_data:
        raise ValueError(
            f"`embedding_data` is specified while `embedding_function` is not specified. "
            "Either specify `embedding_function` during Vector Store initialization or during `add` call."
        )

    check_tensor_name_consistency(tensors, dataset.tensors, embedding_tensor)
    return (None, None, None, tensors)


def check_tensor_name_consistency(tensors, dataset_tensors, embedding_tensor):
    """Check if the tensors specified in the add function are consistent with the tensors in the dataset and the automatically generated tensors (like id)"""
    id_str = "ids" if "ids" in dataset_tensors else "id"
    expected_tensor_length = len(dataset_tensors)
    allowed_missing_tensors = [id_str, embedding_tensor]

    for allowed_missing_tensor in allowed_missing_tensors:
        if allowed_missing_tensor not in tensors and allowed_missing_tensor is not None:
            expected_tensor_length -= 1

    for tensor in tensors:
        if tensor not in dataset_tensors:
            raise ValueError(f"Tensor {tensor} does not exist in dataset")

    try:
        assert len(tensors) == expected_tensor_length
    except Exception:
        missing_tensors = ""
        for tensor in dataset_tensors:
            if tensor not in tensors and tensor not in allowed_missing_tensors:
                missing_tensors += f"`{tensor}`, "
        missing_tensors = missing_tensors[:-2]

        raise ValueError(f"{missing_tensors} tensor(s) is/are missing.")


def find_embedding_tensor(embedding_tensor, tensor_args, dataset) -> str:
    """Find the single embedding tensor to which embedding data should be uploaded. If there are none or multiple tensors, or if overlaps with the existing tensor arguments, raise an error."""

    if not embedding_tensor:
        embedding_tensors = find_embedding_tensors(dataset)

        if len(embedding_tensors) >= 2:
            raise ValueError(
                f"embedding_function is specified but multiple embedding tensors were found in the Vector Store, so it it not clear to which tensor the embeddings should be appended. Please specify the `embedding_tensor` parameter for storing the embeddings."
            )
        elif len(embedding_tensors) == 0:
            raise ValueError(
                f"embedding_function is specified but no embedding tensors were found in the Vector Store, so the embeddings cannot be added. Please specify the `embedding_tensor` parameter for storing the embeddings."
            )

        embedding_tensor = embedding_tensors[0]

    if embedding_tensor in tensor_args:
        raise ValueError(
            f"{embedding_tensor} was specified as a tensor parameter for adding data, in addition to being specified as the `embedding_tensor' for storing embedding from the embedding_function."
            f"Either `embedding_function` or `embedding_data` shouldn't be specified or `{embedding_tensor}` shouldn't be specified as a tensor for appending data."
        )
    return embedding_tensor


def find_embedding_tensors(dataset) -> List[str]:
    """Find all the embedding tensors in a dataset."""
    matching_tensors = []
    for tensor in dataset.tensors:
        if (
            dataset[tensor].htype == "embedding"
            or tensor == "embedding"
            or tensor == "embeddings"
        ):
            matching_tensors.append(tensor)

    return matching_tensors
