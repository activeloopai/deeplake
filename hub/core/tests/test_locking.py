from hub.util.exceptions import LockedException
import numpy as np
import pytest
import hub
import uuid
import time
import warnings


_counter = 0


class VM(object):
    """
    Emulates a different machine
    """

    def __init__(self):
        global _counter
        self.id = _counter
        _counter += 1

    def __enter__(self):
        self._getnode = uuid.getnode
        uuid.getnode = lambda: self.id

    def __exit__(self, *args, **kwargs):
        uuid.getnode = self._getnode


def test_dataset_locking(s3_ds_generator):
    ds = s3_ds_generator()
    ds.create_tensor("x")
    arr = np.random.random((32, 32))
    ds.x.append(arr)

    with VM():
        # Make sure read only warning is raised
        with pytest.warns(UserWarning):
            ds = s3_ds_generator()
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert ds.read_only == True

        # No warnings if user requests read only mode
        with warnings.catch_warnings(record=True) as ws:
            ds = s3_ds_generator(read_only=True)
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert not ws

        DATASET_LOCK_VALIDITY = hub.constants.DATASET_LOCK_VALIDITY
        # Temporarily set validity to 1 second so we dont have to wait too long.
        hub.constants.DATASET_LOCK_VALIDITY = 1
        # Wait for lock to expire.
        time.sleep(1.1)

        try:
            ds = s3_ds_generator()
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
            assert ds.read_only == False
        finally:
            hub.constants.DATASET_LOCK_VALIDITY = DATASET_LOCK_VALIDITY


def test_vc_locking(s3_ds_generator):
    ds = s3_ds_generator()
    ds.create_tensor("x")
    arr = np.random.random((32, 32))
    ds.x.append(arr)
    ds.commit()
    ds.checkout("branch", create=True)
    with VM():
        with warnings.catch_warnings(record=True) as ws:
            ds = s3_ds_generator()
        np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert not ws, str(ws[0])
