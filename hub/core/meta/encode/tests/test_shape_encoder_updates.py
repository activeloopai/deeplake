import numpy as np
from hub.core.meta.encode.shape import ShapeEncoder
from .common import assert_encoded


def test_update_no_change():
    enc = ShapeEncoder(np.array([[101, 100, 1], [100, 101, 5]]))

    enc[0] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[1] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[2] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[3] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[4] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[5] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    assert enc.num_samples == 6


def test_update_squeeze_trivial():
    enc = ShapeEncoder(np.array([[28, 0, 2], [100, 100, 3], [28, 0, 5]]))

    enc[3] = (28, 0)
    assert_encoded(enc, [[28, 0, 5]])

    assert enc.num_samples == 6


def test_update_squeeze_complex():
    enc = ShapeEncoder(
        np.array([[10, 10, 1], [28, 0, 2], [100, 100, 3], [28, 0, 5], [10, 10, 7]])
    )

    enc[3] = (28, 0)
    assert_encoded(enc, [[10, 10, 1], [28, 0, 5], [10, 10, 7]])

    assert enc.num_samples == 8


def test_update_move_up():
    enc = ShapeEncoder(np.array([[101, 100, 0], [100, 101, 5]]))

    enc[1] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])

    enc[2] = (101, 100)
    assert_encoded(enc, [[101, 100, 2], [100, 101, 5]])

    assert enc.num_samples == 6


def test_update_move_down():
    enc = ShapeEncoder(np.array([[101, 100, 5], [100, 101, 10]]))

    enc[5] = (100, 101)
    assert_encoded(enc, [[101, 100, 4], [100, 101, 10]])

    enc[4] = (100, 101)
    assert_encoded(enc, [[101, 100, 3], [100, 101, 10]])

    enc[3] = (100, 101)
    assert_encoded(enc, [[101, 100, 2], [100, 101, 10]])

    assert enc.num_samples == 11


def test_update_replace():
    enc = ShapeEncoder(np.array([[100, 100, 0]]))
    enc[0] = (100, 101)
    assert enc.num_samples == 1


def test_update_split_up():
    enc = ShapeEncoder(np.array([[100, 101, 5]]))

    enc[0] = (101, 100)
    assert_encoded(enc, [[101, 100, 0], [100, 101, 5]])


def test_update_split_down():
    enc = ShapeEncoder(np.array([[100, 101, 5]]))

    enc[5] = (101, 100)
    assert_encoded(enc, [[100, 101, 4], [101, 100, 5]])


def test_update_split_middle():
    enc = ShapeEncoder(np.array([[28, 0, 5]]))

    enc[3] = (100, 100)
    assert_encoded(enc, [[28, 0, 2], [100, 100, 3], [28, 0, 5]])
