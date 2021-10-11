import pytest
from importlib.util import find_spec


def pytorch_installed():
    return find_spec("torch") != None


def tensorflow_installed():
    return find_spec("tensorflow") != None


def _tfds_installed():
    return find_spec("tensorflow_datasets") != None


def ray_installed():
    return find_spec("ray") != None


requires_torch = pytest.mark.skipif(
    not pytorch_installed(), reason="requires pytorch to be installed"
)

requires_tensorflow = pytest.mark.skipif(
    not tensorflow_installed(), reason="requires tensorflow to be installed"
)

requires_tfds = pytest.mark.skipif(
    not _tfds_installed(), reason="requires tensorflow_datasets to be installed"
)
