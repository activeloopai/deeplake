from hub.utils import _flatten, batch, pytorch_loaded, tensorflow_loaded


def test_flatten_array():
    expected_list = [1, 2, 3, 4, 5]
    flatten_list = _flatten([[1, 2], [3, 4, 5]])
    assert flatten_list == expected_list


def test_pytorch_loaded():
    result = pytorch_loaded()
    assert not result


def test_tensorflow_loaded():
    result = tensorflow_loaded()
    assert not result


def test_batch():
    actual = batch([1, 2, 3, 4], 2)
    assert next(actual) == [1, 2]
    assert next(actual) == [3, 4]
