import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

import numpy as np

import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union


def check_indra_installation(exec_option, indra_installed):
    if exec_option == "indra" and not indra_installed:
        raise raise_indra_installation_error(indra_import_error=False)


def check_length_of_each_tensor(tensors):
    tensor_length = len(tensors["texts"])

    for tensor_name in tensors:
        if len(tensors[f"{tensor_name}"]) != tensor_length:
            tensor_lengthes = create_tensor_to_length_str(tensors)

            raise Exception(
                f"All of the tensors should have equal length. Currently tensors have different length: {tensor_lengthes}"
            )


def create_tensor_to_length_str(tensors):
    tensor_lengthes = "\n"
    for tensor_name in tensors:
        tensor_lengthes += (
            f"length of {tensor_name} = {len(tensors[f'{tensor_name}'])}\n"
        )
    return tensor_lengthes
