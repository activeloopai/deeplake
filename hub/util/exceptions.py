from typing import Any


class ChunkSizeTooSmallError(Exception):
    def __init__(
        self,
        message="If the size of the last chunk is given, it must be smaller than the requested chunk size.",
    ):
        super().__init__(message)


class TensorNotFoundError(KeyError):
    def __init__(self, tensor_name: str, dataset_path: str):
        super().__init__(
            "Tensor {} not found in dataset {}".format(tensor_name, dataset_path)
        )


class InvalidKeyTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            "Item {} is of type {} is not a valid key".format(
                str(item), type(item).__name__
            )
        )


class UnsupportedTensorTypeError(TypeError):
    def __init__(self, item: Any):
        super().__init__(
            "Key of type {} is not currently supported to convert to a tensor.".format(
                type(item).__name__
            )
        )
