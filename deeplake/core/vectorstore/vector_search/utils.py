import functools
import types
import random
import string
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple

import deeplake
from deeplake.constants import MB, DEFAULT_VECTORSTORE_INDEX_PARAMS, TARGET_BYTE_SIZE
from deeplake.enterprise.util import INDRA_INSTALLED
from deeplake.util.exceptions import TensorDoesNotExistError
from deeplake.util.warnings import always_warn
from deeplake.core.dataset import DeepLakeCloudDataset, Dataset
from deeplake.core.vectorstore.embeddings.embedder import DeepLakeEmbedder
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.path import get_path_type

import numpy as np


EXEC_OPTION_TO_RUNTIME: Dict[str, Optional[Dict]] = {
    "compute_engine": None,
    "python": None,
    "tensor_db": {"db_engine": True},
}


def parse_tensor_return(tensor):
    return tensor.data(aslist=True)["value"]


class ExecOptionBase(ABC):
    def get_token(self, token):
        user_profile = self.client.get_user_profile()
        if user_profile["name"] != "public":
            token = token or self.client.get_token()
        return token

    @abstractmethod
    def get_exec_option(self):
        return NotImplementedError()


class ExecOptionCloudDataset(ExecOptionBase):
    def __init__(self, dataset, username, path_type):
        self.dataset = dataset
        self.client = dataset.client
        self.token = self.dataset.token
        self.username = username
        self.path_type = path_type

    def get_exec_option(self):
        # option 1: dataset is created in vector_db:
        if (
            isinstance(self.dataset, DeepLakeCloudDataset)
            and "vectordb/" in self.dataset.base_storage.root
            and self.token is not None
        ):
            return "tensor_db"
        # option 2: dataset is created in a linked storage or locally,
        # indra is installed user/org has access to indra
        elif self.path_type == "hub" and INDRA_INSTALLED and self.username != "public":
            return "compute_engine"
        else:
            return "python"


class ExecOptionLocalDataset(ExecOptionBase):
    def __init__(self, dataset, username):
        self.dataset = dataset
        self.token = self.dataset.token
        self.username = username

    def get_exec_option(self):
        if self.token is None:
            return "python"

        if "mem://" in self.dataset.path:
            return "python"

        if INDRA_INSTALLED and self.username != "public":
            return "compute_engine"
        return "python"


def exec_option_factory(dataset, username):
    path_type = get_path_type(dataset.path)
    if path_type == "local":
        return ExecOptionLocalDataset(dataset, username)
    return ExecOptionCloudDataset(dataset, username, path_type)


def parse_exec_option(dataset, exec_option, username):
    if exec_option is None or exec_option == "auto":
        exec_option = exec_option_factory(dataset, username)
        return exec_option.get_exec_option()
    return exec_option


def parse_index_params(index_params):
    ip = DEFAULT_VECTORSTORE_INDEX_PARAMS.copy()
    valid_keys = ip.keys()

    if index_params:
        for key, value in index_params.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid key '{key}' in index_params. Valid keys are: {valid_keys}"
                )

            ip[key] = value

    return ip


def parse_return_tensors(dataset, return_tensors, embedding_tensor, return_view):
    """Select the best selection of data and tensors to be returned"""
    if return_view:
        return_tensors = "*"

    if not return_tensors or return_tensors == "*":
        return_tensors = [
            tensor
            for tensor in dataset.tensors
            if (tensor != embedding_tensor or return_tensors == "*")
        ]
    for tensor in return_tensors:
        if tensor not in dataset.tensors:
            raise TensorDoesNotExistError(tensor)

    return return_tensors


def check_indra_installation(exec_option):
    if exec_option == "compute_engine" and not INDRA_INSTALLED:
        from deeplake.enterprise.util import raise_indra_installation_error

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

    return tensor_length


def create_tensor_to_length_str(tensors):
    tensor_lengths = "\n"
    for tensor_name in tensors:
        tensor_lengths += (
            f"length of {tensor_name} = {len(tensors[f'{tensor_name}'])}\n"
        )
    return tensor_lengths


def generate_random_string(length):
    # Define the character set to include letters (both lowercase and uppercase) and digits
    characters = string.ascii_letters + string.digits
    # Generate a random string of the specified length
    random_string = "".join(random.choice(characters) for _ in range(length))

    return random_string


def generate_json(value, key):
    return {key: value}


def create_data(
    number_of_data, embedding_dim=100, metadata_key="abc", string_length=1000
):
    embeddings = np.random.uniform(
        low=-10, high=10, size=(number_of_data, embedding_dim)
    ).astype(np.float32)
    texts = [generate_random_string(string_length) for i in range(number_of_data)]
    ids = [f"{i}" for i in range(number_of_data)]
    metadata = [generate_json(i, metadata_key) for i in range(number_of_data)]
    images = ["deeplake/tests/dummy_data/images/car.jpg" for i in range(number_of_data)]
    return texts, embeddings, ids, metadata, images


def parse_search_args(**kwargs):
    """Helper function for raising errors if invalid parameters are specified to search"""

    if kwargs["exec_option"] not in ("python", "compute_engine", "tensor_db"):
        raise ValueError(
            "Invalid `exec_option` it should be either `python`, `compute_engine` or `tensor_db`."
        )

    if kwargs.get("embedding") is not None and kwargs.get("query") is not None:
        raise ValueError(
            "Both `embedding` and `query` were specified. Please specify either one or the other."
        )

    if (
        kwargs["embedding_function"] is None
        and kwargs["initial_embedding_function"] is None
        and kwargs["embedding"] is None
        and kwargs["query"] is None
        and kwargs["filter"] is None
    ):
        raise ValueError(
            f"Either an `embedding`, `embedding_function`, `filter`, or `query` must be specified."
        )

    if kwargs["embedding"] is not None and kwargs["embedding_function"]:
        always_warn(
            "Both `embedding` and `embedding_function` were specified."
            " Already computed `embedding` will be used."
        )
    if kwargs["embedding_data"] is None and kwargs["embedding_function"] is not None:
        raise ValueError(
            f"When an `embedding_function` is specified, `embedding_data` must also be specified."
        )

    if (
        kwargs["embedding_data"] is not None
        and kwargs["embedding_function"] is None
        and kwargs["initial_embedding_function"] is None
    ):
        raise ValueError(
            f"When an `embedding_data` is specified, `embedding_function` must also be specified."
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


def get_embedding_tensor(embedding_tensor, embedding_source_tensor, dataset):
    if embedding_source_tensor is None:
        raise ValueError("`embedding_source_tensor` was not specified")

    embedding_tensor = get_embedding_tensors(
        embedding_tensor=embedding_tensor,
        tensor_args={},
        dataset=dataset,
    )

    return embedding_tensor


def parse_tensors_kwargs(
    tensors,
    embedding_function,
    embedding_data,
    embedding_tensor,
):
    tensors = tensors.copy()

    # embedding_tensor = (embedding_function, embedding_data) syntax
    func_comma_data_style = (
        lambda item: isinstance(item[1], tuple)
        and len(item[1]) == 2
        and callable(item[1][0])
    )

    funcs = []
    data = []
    tensors_ = []

    filtered = dict(filter(func_comma_data_style, tensors.items()))

    # cannot use both syntaxes (kwargs style and args style) at the same time
    if len(filtered) > 0:
        if embedding_function:
            raise ValueError(
                "Cannot specify embedding functions in both `tensors` and `embedding_function`."
            )

        if embedding_data:
            raise ValueError(
                "Cannot specify embedding data in both `tensors` and `embedding_data`."
            )

        if embedding_tensor:
            raise ValueError(
                "Cannot specify embedding tensors in both `tensors` and `embedding_tensor`."
            )
    else:
        if isinstance(embedding_function, list):
            embedding_function = [
                create_embedding_function(fn_i) for fn_i in embedding_function
            ]
        else:
            embedding_function = create_embedding_function(embedding_function)
        return embedding_function, embedding_data, embedding_tensor, tensors

    # separate embedding functions, data and tensors
    for k, v in filtered.items():
        func = create_embedding_function(v[0])
        funcs.append(func)
        data.append(v[1])
        tensors_.append(k)
        # remove embedding tensors (tuple format) from tensors
        del tensors[k]

    return funcs, data, tensors_, tensors


def _validate_embedding_functions(embedding_function, initial_embedding_function):
    if embedding_function is None and initial_embedding_function is None:
        raise ValueError(
            "`embedding_function` was not specified during initialization of vector store or the update call"
        )


def _get_single_value_from_list(data):
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def _validate_source_and_embedding_tensors(embedding_source_tensor, embedding_tensor):
    if isinstance(embedding_source_tensor, str) and isinstance(embedding_tensor, list):
        raise ValueError(
            "Multiple `embedding_tensor` were specified while a single `embedding_source_tensor` was given."
        )

    if (
        isinstance(embedding_source_tensor, list)
        and len(embedding_source_tensor) > 1
        and isinstance(embedding_tensor, str)
    ):
        raise ValueError(
            "Multiple `embedding_source_tensor` were specified while a single `embedding_tensor` was given."
        )


def _convert_to_embedder_list(embedding_function):
    if isinstance(embedding_function, list):
        return [DeepLakeEmbedder(embedding_function=fn) for fn in embedding_function]

    valid_function_types = (
        types.MethodType,
        types.FunctionType,
        types.LambdaType,
        functools.partial,
    )
    if isinstance(embedding_function, valid_function_types):
        return DeepLakeEmbedder(embedding_function=embedding_function)

    if embedding_function is not None:
        raise ValueError(
            "Invalid `embedding_function` type. It should be either a function or a list of functions."
        )


def parse_update_arguments(
    dataset,
    embedding_function=None,
    initial_embedding_function=None,
    embedding_source_tensor=None,
    embedding_tensor=None,
):
    _validate_embedding_functions(embedding_function, initial_embedding_function)

    embedding_tensor = get_embedding_tensor(
        embedding_tensor, embedding_source_tensor, dataset
    )
    embedding_tensor = _get_single_value_from_list(embedding_tensor)

    _validate_source_and_embedding_tensors(embedding_source_tensor, embedding_tensor)

    embedding_function = _convert_to_embedder_list(embedding_function)
    final_embedding_function = embedding_function or initial_embedding_function

    if isinstance(embedding_tensor, list) and not isinstance(
        final_embedding_function, list
    ):
        final_embedding_function = [final_embedding_function] * len(embedding_tensor)

    if isinstance(final_embedding_function, list):
        final_embedding_function = [
            fn.embed_documents for fn in final_embedding_function
        ]
    else:
        final_embedding_function = final_embedding_function.embed_documents

    if isinstance(embedding_tensor, list) and isinstance(embedding_source_tensor, list):
        assert len(embedding_tensor) == len(embedding_source_tensor), (
            "The length of the `embedding_tensor` does not match the length of "
            "`embedding_source_tensor`"
        )

    return (final_embedding_function, embedding_source_tensor, embedding_tensor)


def convert_embedding_source_tensor_to_embeddings(
    dataset,
    embedding_source_tensor,
    embedding_tensor,
    embedding_function,
    row_ids,
):
    embedding_tensor_data = {}
    if isinstance(embedding_source_tensor, list):
        for embedding_source_tensor_i, embedding_tensor_i, embedding_fn_i in zip(
            embedding_source_tensor, embedding_tensor, embedding_function
        ):
            embedding_data = dataset[row_ids][embedding_source_tensor_i].numpy()
            embedding_tensor_data[embedding_tensor_i] = embedding_fn_i(embedding_data)
            embedding_tensor_data[embedding_tensor_i] = np.array(
                embedding_tensor_data[embedding_tensor_i], dtype=np.float32
            )
    else:
        embedding_data = dataset[row_ids][embedding_source_tensor].numpy()
        embedding_tensor_data[embedding_tensor] = embedding_function(embedding_data)
        embedding_tensor_data[embedding_tensor] = np.array(
            embedding_tensor_data[embedding_tensor], dtype=np.float32
        )

    return embedding_tensor_data


def parse_add_arguments(
    dataset,
    embedding_function=None,
    initial_embedding_function=None,
    embedding_data=None,
    embedding_tensor=None,
    **tensors,
):
    """Parse the input argument to the Vector Store add function to infer whether they are a valid combination."""
    if embedding_data and not isinstance(next(iter(embedding_data)), list):
        embedding_data = [embedding_data]
    if embedding_tensor and not isinstance(embedding_tensor, list):
        embedding_tensor = [embedding_tensor]

    if embedding_function:
        (
            embedding_function,
            embedding_tensor,
        ) = check_embedding_function_embedding_tensor_consistency(
            embedding_tensor,
            embedding_function,
            embedding_data,
            tensors,
            dataset,
        )
        return (
            [fn.embed_documents for fn in embedding_function],
            embedding_data,
            embedding_tensor,
            tensors,
        )

    if initial_embedding_function:
        if not embedding_data:
            check_tensor_name_consistency(tensors, dataset.tensors, None)
            return (None, None, None, tensors)

        (
            initial_embedding_function,
            embedding_tensor,
        ) = check_embedding_function_embedding_tensor_consistency(
            embedding_tensor,
            initial_embedding_function,
            embedding_data,
            tensors,
            dataset,
        )
        return (
            [fn.embed_documents for fn in initial_embedding_function],
            embedding_data,
            embedding_tensor,
            tensors,
        )

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


def check_embedding_function_embedding_tensor_consistency(
    embedding_tensor,
    embedding_function,
    embedding_data,
    tensors,
    dataset,
):
    if not embedding_data:
        raise ValueError(
            f"embedding_data is not specified. When using embedding_function it is also necessary to specify the data that you want to embed"
        )

    # if single embedding function is specified, use it for all embedding data
    if not isinstance(embedding_function, list):
        embedding_function = [embedding_function] * len(embedding_data)

    embedding_tensor = get_embedding_tensors(embedding_tensor, tensors, dataset)

    assert len(embedding_function) == len(
        embedding_data
    ), "embedding_function and embedding_data must be of the same length"
    assert len(embedding_function) == len(
        embedding_tensor
    ), "embedding_function and embedding_tensor must be of the same length"

    check_tensor_name_consistency(tensors, dataset.tensors, embedding_tensor)
    return embedding_function, embedding_tensor


def check_tensor_name_consistency(tensors, dataset_tensors, embedding_tensor):
    """Check if the tensors specified in the add function are consistent with the tensors in the dataset and the automatically generated tensors (like id)"""
    id_str = "ids" if "ids" in dataset_tensors else "id"
    expected_tensor_length = len(dataset_tensors)
    if embedding_tensor is None:
        embedding_tensor = []
    allowed_missing_tensors = [id_str, *embedding_tensor]

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


def get_embedding_tensors(embedding_tensor, tensor_args, dataset) -> List[str]:
    """Get the embedding tensors to which embedding data should be uploaded."""
    if not embedding_tensor:
        embedding_tensor = find_embedding_tensors(dataset)

        if len(embedding_tensor) == 0:
            raise ValueError(
                f"embedding_function is specified but no embedding tensors were found in the Vector Store,"
                " so the embeddings cannot be added. Please specify the `embedding_tensor` parameter for storing the embeddings."
            )
        elif len(embedding_tensor) > 1:
            raise ValueError(
                f"embedding_function is specified but multiple embedding tensors were found in the Vector Store,"
                " so it is not clear to which tensor the embeddings should be added. Please specify the `embedding_tensor`"
                " parameter for storing the embeddings."
            )

    # if same tensor is specified in both embedding_tensor and tensors, raise error
    for tensor in embedding_tensor:
        if tensor_args.get(tensor):
            raise ValueError(
                f"{tensor} was specified as a tensor parameter for adding data, in addition to being specified as an `embedding_tensor' for storing embedding from the embedding_function."
                f"Either `embedding_function` or `embedding_data` shouldn't be specified or `{tensor}` shouldn't be specified as a tensor for appending data."
            )
    return embedding_tensor


def find_embedding_tensors(dataset) -> List[str]:
    """Find all the embedding tensors in a dataset."""
    matching_tensors = []
    for tensor in dataset.tensors.values():
        if is_embedding_tensor(tensor):
            matching_tensors.append(tensor.key)

    return matching_tensors


def is_embedding_tensor(tensor):
    """Check if a tensor is an embedding tensor."""

    valid_names = ["embedding", "embeddings"]

    return (
        tensor.htype == "embedding"
        or tensor.meta.name in valid_names
        or tensor.key in valid_names
    )


def index_used(exec_option):
    """Check if the index is used for the exec_option"""
    return exec_option in ("tensor_db", "compute_engine")


def create_embedding_function(embedding_function):
    if embedding_function:
        return DeepLakeEmbedder(
            embedding_function=embedding_function,
        )
    return None
