import pytest
from hub.core.tests.common import parametrize_all_storages_and_caches
from hub.util.exceptions import ReadOnlyModeError
from hub import Dataset


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@parametrize_all_storages_and_caches
def test_readonly_del(storage):
    storage.enable_readonly()
    del storage["help!"]


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@parametrize_all_storages_and_caches
def test_readonly_set(storage):
    storage.enable_readonly()
    storage["help! im stuck in a storage provider!"] = "salvation"


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@parametrize_all_storages_and_caches
def test_readonly_clear(storage):
    storage.enable_readonly()
    storage.clear()


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@parametrize_all_storages_and_caches
def test_readonly_flush(storage):
    storage.enable_readonly()
    storage.flush()


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@parametrize_all_storages_and_caches
def test_readonly_ds_create_tensor(storage):
    ds = Dataset(read_only=True, storage=storage)
    ds.create_tensor("test")
