from hub.util.exceptions import LockedException
import numpy as np
import pytest
import hub
import uuid
import time
import warnings


def test_dataset_locking(s3_ds_generator):
    ds = s3_ds_generator()
    ds.create_tensor("x")
    arr = np.random.random((32, 32))
    ds.x.append(arr)

    # Emulate a different machine
    getnode = uuid.getnode
    uuid.getnode = lambda: getnode() + 1
    hub.core.lock._LOCKS.clear()

    try:

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
    finally:
        uuid.getnode = getnode
