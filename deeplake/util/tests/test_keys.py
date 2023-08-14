import deeplake
from deeplake.util.keys import dataset_exists


def test_dataset_exists():
    ds = deeplake.dataset("mem://x")
    assert dataset_exists(ds.storage)

    # Single files missing is fine
    del ds.storage["version_control_info.json"]
    assert dataset_exists(ds.storage)

    ds = deeplake.dataset("mem://x")
    del ds.storage["dataset_meta.json"]
    assert dataset_exists(ds.storage)

    # Enough files are missing and it's no longer valid
    ds = deeplake.dataset("mem://x")
    del ds.storage["dataset_meta.json"]
    del ds.storage["version_control_info.json"]
    assert not dataset_exists(ds.storage)
