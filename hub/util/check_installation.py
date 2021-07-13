import pytest


def pytorch_installed():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True


def tensorflow_installed():
    try:
        import tensorflow  # type: ignore

        tensorflow.__version__
    except ImportError:
        return False
    return True


def _tfds_installed():
    try:
        import tensorflow_datasets  # type: ignore

        tensorflow_datasets.__version__
    except ImportError:
        return False
    return True


requires_torch = pytest.mark.skipif(
    not pytorch_installed(), reason="requires pytorch to be installed"
)

requires_tensorflow = pytest.mark.skipif(
    not tensorflow_installed(), reason="requires tensorflow to be installed"
)

requires_tfds = pytest.mark.skipif(
    not _tfds_installed(), reason="requires tensorflow_datasets to be installed"
)
