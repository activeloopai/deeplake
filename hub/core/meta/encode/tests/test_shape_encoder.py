import numpy as np
import pytest
from hub.core.meta.encode.shape import ShapeEncoder


# TODO: test update


def test_trivial():
    enc = ShapeEncoder()

    enc.register_samples((28, 28, 3), 4)
    assert enc[1] == (28, 28, 3)
    assert enc.num_samples == 4
    assert len(enc._encoded) == 1


def test_fixed():
    enc = ShapeEncoder()

    enc.register_samples((28, 28, 3), 1000)
    enc.register_samples((28, 28, 3), 1000)
    enc.register_samples((28, 28, 3), 3)
    enc.register_samples((28, 28, 3), 1000)
    enc.register_samples((28, 28, 3), 1000)

    assert enc.num_samples == 4003
    assert len(enc._encoded) == 1
    assert enc.num_samples_at(0) == 4003

    assert enc[0] == (28, 28, 3)
    assert enc[1999] == (28, 28, 3)
    assert enc[2000] == (28, 28, 3)
    assert enc[3000] == (28, 28, 3)
    assert enc[-1] == (28, 28, 3)


def test_dynamic():
    enc = ShapeEncoder()

    enc.register_samples((28, 28, 3), 1000)
    enc.register_samples((28, 28, 3), 1000)
    enc.register_samples((30, 28, 3), 1000)
    enc.register_samples((28, 28, 4), 1000)
    enc.register_samples((28, 28, 3), 1)

    assert enc.num_samples == 4001
    assert len(enc._encoded) == 4
    assert enc.num_samples_at(0) == 2000
    assert enc.num_samples_at(1) == 1000
    assert enc.num_samples_at(2) == 1000
    assert enc.num_samples_at(3) == 1

    assert enc[0] == (28, 28, 3)
    assert enc[1999] == (28, 28, 3)
    assert enc[2000] == (30, 28, 3)
    assert enc[3000] == (28, 28, 4)
    assert enc[-1] == (28, 28, 3)


def test_empty():
    enc = ShapeEncoder()

    with pytest.raises(ValueError):
        enc.register_samples((5,), 0)

    with pytest.raises(ValueError):
        enc.register_samples((5, 5), 0)

    with pytest.raises(ValueError):
        enc.register_samples((100, 100, 3), 0)

    assert enc.num_samples == 0
    np.testing.assert_array_equal(enc._encoded, np.array([], dtype=np.uint64))

    with pytest.raises(IndexError):
        enc[0]

    with pytest.raises(IndexError):
        enc[-1]


def test_scalars():
    enc = ShapeEncoder()

    assert enc.num_samples == 0

    enc.register_samples((1,), 500)
    enc.register_samples((2,), 5)
    enc.register_samples((1,), 10)
    enc.register_samples((1,), 10)
    enc.register_samples((0,), 1)

    assert enc.num_samples == 526
    assert len(enc._encoded) == 4

    assert enc[0] == (1,)
    assert enc[499] == (1,)
    assert enc[500] == (2,)
    assert enc[504] == (2,)
    assert enc[505] == (1,)
    assert enc[524] == (1,)
    assert enc[-1] == (0,)

    with pytest.raises(IndexError):
        enc[526]


def _assert_encoded(enc, expected_encoding):
    np.testing.assert_array_equal(enc._encoded, expected_encoding)


def test_update_simple():
    enc = ShapeEncoder(np.array([[100, 100, 0]]))
    enc[0] = (100, 101)
    assert enc.num_samples == 1


def test_update_no_change():
    enc = ShapeEncoder(np.array([[101, 100, 1], [100, 101, 5]]))

    enc[0] = (101, 100)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[1] = (101, 100)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[2] = (100, 101)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[3] = (100, 101)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[4] = (100, 101)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[5] = (100, 101)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    assert enc.num_samples == 6


def test_update_move_down():
    enc = ShapeEncoder(np.array([[101, 100, 0], [100, 101, 5]]))

    enc[1] = (101, 100)
    _assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[2] = (101, 100)
    _assert_encoded(enc, [[101, 100, 2], [100, 101, 5]])

    assert enc.num_samples == 6


def test_update_move_up():
    enc = ShapeEncoder(np.array([[101, 100, 5], [100, 101, 10]]))

    enc[5] = (100, 101)
    _assert_encoded(enc, [[101, 100, 4], [100, 101, 10]])

    enc[4] = (100, 101)
    _assert_encoded(enc, [[101, 100, 3], [100, 101, 10]])

    enc[3] = (100, 101)
    _assert_encoded(enc, [[101, 100, 2], [100, 101, 10]])

    assert enc.num_samples == 6


def test_update_split_first():
    enc = ShapeEncoder(np.array([[100, 101, 5]]))

    enc[0] = (101, 100)
    _assert_encoded(enc, [[101, 100, 0], [100, 101, 5]])


def test_update_split_last():
    enc = ShapeEncoder(np.array([[100, 101, 5]]))

    enc[5] = (101, 100)
    _assert_encoded(enc, [[100, 101, 4], [101, 100, 5]])


def test_update_split_squeeze():
    enc = ShapeEncoder(np.array([[28, 0, 5]]))
    _assert_encoded(enc, [[28, 0, 5]])

    enc[3] = (100, 100)
    _assert_encoded(enc, [[28, 0, 2], [100, 100, 3], [28, 0, 5]])

    enc[3] = (28, 0)
    _assert_encoded(enc, [[28, 0, 5]])

    assert enc.num_samples == 6


def test_failures():
    enc = ShapeEncoder()

    with pytest.raises(ValueError):
        enc.register_samples((5,), 0)

    with pytest.raises(ValueError):
        enc.register_samples((28, 28, 3), 0)

    assert enc.num_samples == 0

    enc.register_samples((100, 100), 100)

    assert len(enc._encoded) == 1

    with pytest.raises(ValueError):
        enc.register_samples((100, 100, 1), 100)

    with pytest.raises(ValueError):
        enc.register_samples((100,), 100)

    assert enc.num_samples == 100
    assert len(enc._encoded) == 1

    assert enc[-1] == (100, 100)

    with pytest.raises(IndexError):
        enc[101]

    with pytest.raises(IndexError):
        enc[101] = (1, 1)

    assert enc.num_samples == 100
