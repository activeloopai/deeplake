import pytest
from deeplake.util.exceptions import ReadOnlyModeError
from deeplake.tests.storage_fixtures import enabled_storages


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
