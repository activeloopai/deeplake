import pytest
import hub
import numpy as np
import uuid
from hub.tests.common import LinkTransformTestContext


def test_linking(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds._link_tensors("x", "y", "append_test")
        ds.x.extend(list(range(10)))
        np.testing.assert_array_equal(ds.x.numpy(), np.arange(10).reshape(-1, 1))
        np.testing.assert_array_equal(ds.x.numpy(), ds.y.numpy())


def test_linking_sequence(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="sequence")
        ds.create_tensor("x_id")
        id_f = lambda _: 0
        with LinkTransformTestContext(id_f, "id"):
            ds._link_tensors("x", "x_id", "id", flatten_sequence=False)
            ds.x.extend(np.random.random((10, 5, 3, 2)))
            assert len(ds.x) == len(ds.x_id) == 10
            np.testing.assert_array_equal(ds.x_id.numpy(), np.zeros((10, 1)))


def test_linking_sequence_update(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="sequence")
        ds.create_tensor("x_id")
        id_f = lambda _: 0
        id_f2 = lambda *_: 1  # updated samples will have x_id=1
        with LinkTransformTestContext(id_f, "id"):
            with LinkTransformTestContext(id_f2, "id2"):
                ds._link_tensors(
                    "x", "x_id", append_f="id", update_f="id2", flatten_sequence=False
                )
                ds.x.extend(np.random.random((10, 5, 3, 2)))
                ds.x[0] += 1
                ds.x[3] += 1
                expected = np.zeros((10, 1))
                expected[0] = 1
                expected[3] = 1
                np.testing.assert_array_equal(ds.x_id.numpy(), expected)
