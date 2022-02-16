from hub.util.exceptions import LockedException
import numpy as np
import pytest
import hub
import uuid
import time
import warnings
from hub.tests.dataset_fixtures import enabled_cloud_dataset_generators
from hub.core.storage import MemoryProvider


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
        self._locks = hub.core.lock._LOCKS.copy()
        hub.core.lock._LOCKS.clear()

    def __exit__(self, *args, **kwargs):
        uuid.getnode = self._getnode
        hub.core.lock._LOCKS.update(self._locks)


@enabled_cloud_dataset_generators
def test_dataset_locking(ds_generator):
    ds = ds_generator()
    ds.create_tensor("x")
    arr = np.random.random((32, 32))
    ds.x.append(arr)

    with VM():
        # Make sure read only warning is raised
        with pytest.warns(UserWarning):
            ds = ds_generator()
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert ds.read_only == True
        with pytest.raises(LockedException):
            ds.read_only = False
        # No warnings if user requests read only mode
        with warnings.catch_warnings(record=True) as ws:
            ds = ds_generator(read_only=True)
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert not ws

        DATASET_LOCK_VALIDITY = hub.constants.DATASET_LOCK_VALIDITY
        # Temporarily set validity to 1 second so we dont have to wait too long.
        hub.constants.DATASET_LOCK_VALIDITY = 1
        # Wait for lock to expire.
        time.sleep(1.1)

        try:
            ds = ds_generator()
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
            assert ds.read_only == False
        finally:
            hub.constants.DATASET_LOCK_VALIDITY = DATASET_LOCK_VALIDITY


@enabled_cloud_dataset_generators
def test_vc_locking(ds_generator):
    ds = ds_generator()
    ds.create_tensor("x")
    arr = np.random.random((32, 32))
    ds.x.append(arr)
    ds.commit()
    ds.checkout("branch", create=True)
    with VM():
        with warnings.catch_warnings(record=True) as ws:
            ds = ds_generator()
        np.testing.assert_array_equal(arr, ds.x[0].numpy())
        assert not ws, str(ws[0])


def test_lock_thread_leaking():
    locks = hub.core.lock._LOCKS
    refs = hub.core.lock._REFS
    nlocks_previous = len(locks)

    def nlocks():
        assert len(locks) == len(refs)
        return len(locks) - nlocks_previous

    path = "mem://abc"
    lockables = hub.core.dataset.dataset._LOCKABLE_STORAGES
    lockables.add(MemoryProvider)
    default_ds_lock_interval = hub.constants.DATASET_LOCK_GC_INTERVAL
    hub.constants.DATASET_LOCK_GC_INTERVAL = 1

    try:
        ds = hub.empty(path, overwrite=True)
        ds.create_tensor("x")
        assert nlocks() == 1

        del ds
        time.sleep(1.2)  # wait for gc
        assert nlocks() == 0

        ds = hub.empty(path, overwrite=True)
        ds.create_tensor("x")
        ds.x.extend(np.random.random((2, 32)))
        views = []
        for i in range(32):
            views.append(ds[i : i + 1])

        del ds
        time.sleep(1.2)

        assert nlocks() == 1  # 1 because views

        views.pop()
        time.sleep(1.2)
        assert nlocks() == 1  # deleting 1 view doesn't release locks

        del views
        time.sleep(1.2)
        assert nlocks() == 0  # 0 because dataset and all views deleted
    finally:
        hub.constants.DATASET_LOCK_GC_INTERVAL = default_ds_lock_interval
        lockables.remove(MemoryProvider)
