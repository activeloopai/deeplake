import pytest
from hub.util.exceptions import ReadOnlyModeError
from hub import Dataset
from hub.tests.storage_fixtures import enabled_storages


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@enabled_storages
def test_readonly_del(storage):
    storage.enable_readonly()
    del storage["help!"]


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@enabled_storages
def test_readonly_set(storage):
    storage.enable_readonly()
    storage["help! im stuck in a storage provider!"] = "salvation"


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@enabled_storages
def test_readonly_clear(storage):
    storage.enable_readonly()
    storage.clear()


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@enabled_storages
def test_readonly_flush(storage):
    storage.enable_readonly()
    storage.flush()


@pytest.mark.xfail(raises=ReadOnlyModeError, strict=True)
@enabled_storages
def test_readonly_ds_create_tensor(storage):
    ds = Dataset(read_only=True, storage=storage)
    ds.create_tensor("test")
