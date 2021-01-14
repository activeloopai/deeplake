from hub.utils import *
from hub.utils import _flatten


def test_flatten_array():
    expected_list = [1, 2, 3, 4, 5]
    flatten_list = _flatten([[1, 2], [3, 4, 5]])
    assert flatten_list == expected_list


def test_pytorch_loaded():
    result = pytorch_loaded()
    if result:
        import torch
    else:
        assert not result


def test_tensorflow_loaded():
    result = tensorflow_loaded()
    if result:
        import tensorflow
    else:
        assert not result


def test_ray_loaded():
    result = ray_loaded()
    if result:
        import ray
    else:
        assert not result


def test_dask_loaded():
    result = dask_loaded()
    if result:
        import dask
    else:
        assert not result


def test_tfdatasets_loaded():
    result = tfds_loaded()
    if result:
        import tensorflow_datasets
    else:
        assert not result


def test_pathos_loaded():
    result = pathos_loaded()
    if result:
        import pathos
    else:
        assert not result


def test_compute_lcm():
    assert compute_lcm([1, 2, 8, 3]) == 24
    assert compute_lcm([1]) == 1
    assert compute_lcm([]) is None
    assert compute_lcm([2, 2]) == 2


def test_batchify():
    actual = iter(batchify([1, 2, 3, 4], 2))
    assert next(actual) == [1, 2]
    assert next(actual) == [3, 4]
