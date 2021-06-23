from hub.core.storage.provider import StorageProvider
import pytest
from hub.core.meta.encode.shape import ShapeEncoder


def test_fixed(memory_storage: StorageProvider):
    enc = ShapeEncoder(memory_storage)

    enc.add_shape((28, 28, 3), 1000)
    enc.add_shape((28, 28, 3), 1000)
    enc.add_shape((28, 28, 3), 3)
    enc.add_shape((28, 28, 3), 1000)
    enc.add_shape((28, 28, 3), 1000)

    assert enc.num_samples == 4003
    assert len(enc._encoded_shapes) == 1

    assert enc[0] == (28, 28, 3)
    assert enc[1999] == (28, 28, 3)
    assert enc[2000] == (28, 28, 3)
    assert enc[3000] == (28, 28, 3)
    assert enc[-1] == (28, 28, 3)


def test_dynamic(memory_storage: StorageProvider):
    enc = ShapeEncoder(memory_storage)

    enc.add_shape((28, 28, 3), 1000)
    enc.add_shape((28, 28, 3), 1000)
    enc.add_shape((30, 28, 3), 1000)
    enc.add_shape((28, 28, 4), 1000)
    enc.add_shape((28, 28, 3), 1)

    assert enc.num_samples == 4001
    assert len(enc._encoded_shapes) == 4

    assert enc[0] == (28, 28, 3)
    assert enc[1999] == (28, 28, 3)
    assert enc[2000] == (30, 28, 3)
    assert enc[3000] == (28, 28, 4)
    assert enc[-1] == (28, 28, 3)


def test_empty(memory_storage: StorageProvider):
    enc = ShapeEncoder(memory_storage)

    with pytest.raises(ValueError):
        enc.add_shape((5,), 0)

    with pytest.raises(ValueError):
        enc.add_shape((5, 5), 0)

    with pytest.raises(ValueError):
        enc.add_shape((100, 100, 3), 0)

    assert enc.num_samples == 0
    assert enc._encoded_shapes is None

    with pytest.raises(IndexError):
        enc[0]

    with pytest.raises(IndexError):
        enc[-1]


def test_scalars(memory_storage: StorageProvider):
    enc = ShapeEncoder(memory_storage)

    assert enc.num_samples == 0

    enc.add_shape((1,), 500)
    enc.add_shape((2,), 5)
    enc.add_shape((1,), 10)
    enc.add_shape((1,), 10)
    enc.add_shape((0,), 1)

    assert enc.num_samples == 526
    assert len(enc._encoded_shapes) == 4

    assert enc[0] == (1,)
    assert enc[499] == (1,)
    assert enc[500] == (2,)
    assert enc[504] == (2,)
    assert enc[505] == (1,)
    assert enc[524] == (1,)
    assert enc[-1] == (0,)

    with pytest.raises(IndexError):
        enc[526]


def test_failures(memory_storage: StorageProvider):
    enc = ShapeEncoder(memory_storage)

    with pytest.raises(ValueError):
        enc.add_shape((5,), 0)

    with pytest.raises(ValueError):
        enc.add_shape((28, 28, 3), 0)

    assert enc.num_samples == 0

    enc.add_shape((100, 100), 100)

    assert len(enc._encoded_shapes) == 1

    with pytest.raises(ValueError):
        enc.add_shape((100, 100, 1), 100)

    with pytest.raises(ValueError):
        enc.add_shape((100,), 100)

    assert enc.num_samples == 100
    assert len(enc._encoded_shapes) == 1

    assert enc[-1] == (100, 100)
