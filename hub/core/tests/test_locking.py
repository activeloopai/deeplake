from hub.util.exceptions import LockedException
import pytest
import hub
import uuid


def test_dataset_locking(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("image")

    # Emulate a different machine
    getnode = uuid.getnode
    uuid.getnode = lambda: getnode() + 1
    hub.core.lock._LOCKS.clear()

    try:
        ds = local_ds_generator()
        assert ds.read_only == True
    finally:
        uuid.getnode = getnode
