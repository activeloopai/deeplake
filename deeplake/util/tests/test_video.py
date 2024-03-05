from deeplake.util.video import normalize_index
import pytest


def test_normalize_index():
    assert normalize_index(None, 10) == (0, 10, 1, False)
    assert normalize_index(5, 10) == (5, 6, 1, False)
    assert normalize_index(slice(5, 10, 2), 10) == (5, 10, 2, False)
    assert normalize_index(slice(5, 10, 2), 10) == (5, 10, 2, False)
    assert normalize_index(slice(5, None, 2), 10) == (5, 10, 2, False)
    assert normalize_index(slice(None, 5, 2), 10) == (0, 5, 2, False)

    with pytest.raises(IndexError):
        normalize_index([5, 7, 8], 10)
