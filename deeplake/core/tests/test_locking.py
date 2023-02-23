from deeplake.util.exceptions import LockedException
import numpy as np
import pytest
import deeplake
import uuid
import time
import warnings
import json
from deeplake.tests.dataset_fixtures import (
    enabled_persistent_non_gdrive_dataset_generators,
)
from concurrent.futures import ThreadPoolExecutor

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
        self._locks = deeplake.core.lock._LOCKS.copy()
        deeplake.core.lock._LOCKS.clear()

    def __exit__(self, *args, **kwargs):
        uuid.getnode = self._getnode
        deeplake.core.lock._LOCKS.update(self._locks)


@enabled_persistent_non_gdrive_dataset_generators
def test_dataset_locking(ds_generator):
    DATASET_LOCK_VALIDITY = deeplake.constants.DATASET_LOCK_VALIDITY
    # Temporarily set validity to 1 second so we dont have to wait too long.
    deeplake.constants.DATASET_LOCK_VALIDITY = 1
    try:
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
            # Raise error if user explicitly asks for write access
            with pytest.raises(LockedException):
                ds = ds_generator(read_only=False)
            # No warnings if user requests read only mode
            with warnings.catch_warnings(record=True) as ws:
                ds = ds_generator(read_only=True)
                np.testing.assert_array_equal(arr, ds.x[0].numpy())
            assert not ws

            # Wait for lock to expire.
            time.sleep(1.1)

            ds = ds_generator()
            np.testing.assert_array_equal(arr, ds.x[0].numpy())
            assert ds.read_only == False
    finally:
        deeplake.constants.DATASET_LOCK_VALIDITY = DATASET_LOCK_VALIDITY


@enabled_persistent_non_gdrive_dataset_generators
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


def test_lock_thread_leaking(s3_ds_generator):
    locks = deeplake.core.lock._LOCKS
    refs = deeplake.core.lock._REFS
    nlocks_previous = len(locks)

    def nlocks():
        assert len(locks) == len(refs)
        return len(locks) - nlocks_previous

    ds = s3_ds_generator()
    ds.create_tensor("a")
    assert nlocks() == 1

    ds.__del__()  # Note: investigate why this doesnt happen automatically. (cyclic refs?)
    del ds
    assert nlocks() == 0

    ds = s3_ds_generator()
    ds.create_tensor("x")
    ds.x.extend(np.random.random((2, 32)))
    views = []
    for i in range(32):
        views.append(ds[i : i + 1])

    ds.__del__()
    del ds

    assert nlocks() == 1  # 1 because views

    views[-1].__del__()
    views.pop()
    assert nlocks() == 1  # deleting 1 view doesn't release locks

    for i in range(len(views)):
        views[i].__del__()
    del views
    assert nlocks() == 0  # 0 because dataset and all views deleted


def test_concurrent_locking(memory_ds):
    storage = memory_ds.base_storage

    def f(i):
        lock = deeplake.core.lock.Lock(storage, "lock.lock")
        with lock:
            byts = storage.get("meta.json")
            if byts is None:
                d = {"x": []}
            else:
                d = json.loads(byts.decode("utf-8"))
            d["x"].append(i)
            storage["meta.json"] = json.dumps(d).encode("utf-8")

    n = 10
    with ThreadPoolExecutor(max_workers=n) as executor:
        executor.map(f, range(n))
    assert set(json.loads(storage["meta.json"])["x"]) == set(range(n))
