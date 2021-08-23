from hub.util.exceptions import LockedException
import pytest
import hub
import uuid
import time


def test_dataset_locking(s3_ds_generator):
    ds = s3_ds_generator()
    ds.create_tensor("image")

    # Emulate a different machine
    getnode = uuid.getnode
    uuid.getnode = lambda: getnode() + 1
    hub.core.lock._LOCKS.clear()

    with pytest.warns(UserWarning):
        ds = s3_ds_generator()
    assert ds.read_only == True

    DATASET_LOCK_VALIDITY = hub.constants.DATASET_LOCK_VALIDITY
    # Temporarily set validity to 1 second so we dont have to wait too long.
    hub.constants.DATASET_LOCK_VALIDITY = 1
    # Wait for lock to expire.
    time.sleep(1.1)

    try:
        ds = s3_ds_generator()
        assert ds.read_only == False
    finally:
        uuid.getnode = getnode
        hub.constants.DATASET_LOCK_VALIDITY = DATASET_LOCK_VALIDITY
