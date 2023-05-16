import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

import numpy as np

import random
import string
import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union


def check_indra_installation(exec_option, indra_installed):
    if exec_option == "compute_engine" and not indra_installed:
        raise raise_indra_installation_error(
            indra_import_error=False
        )  # pragma: no cover


def check_length_of_each_tensor(tensors):
    tensor_length = len(tensors["texts"])

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


def generate_random_json(integer):
    string = "abcdefg"
    return {string: integer}


def create_data(number_of_data, embedding_dim=100):
    embeddings = np.random.randint(0, 255, (number_of_data, embedding_dim))
    texts = [generate_random_string(1000) for i in range(number_of_data)]
    ids = [f"{i}" for i in range(number_of_data)]
    metadata = [generate_random_json(i) for i in range(number_of_data)]
    return texts, embeddings, ids, metadata
