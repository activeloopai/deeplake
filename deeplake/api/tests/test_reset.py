from deeplake.util.exceptions import (
    DatasetCorruptError,
    ReadOnlyModeError,
    CheckoutError,
)
from deeplake.util.version_control import rebuild_version_info
from deeplake.util.testing import compare_version_info
from deeplake.util.keys import get_commit_info_key

import numpy as np

import deeplake
import pytest
import json


def corrupt_ds(ds, tensor, data):
    ds[tensor].append(data)
    ds[tensor].meta.length = 0
    ds[tensor].meta.is_dirty = True
    ds.flush()


def verify_reset_on_checkout(ds, branch, commit_id, old_head, data):
    assert ds.branch == branch
    assert ds.commit_id == commit_id
    assert ds.pending_commit_id != old_head

    for tensor in data:
        np.testing.assert_array_equal(ds[tensor].numpy(), data[tensor])


@pytest.mark.parametrize(
    "path",
    ["local_path", pytest.param("s3_path", marks=pytest.mark.slow)],
    indirect=True,
)
def test_load_corrupt_dataset(path):
    ds = deeplake.empty(path, overwrite=True)

    access_method = "local" if path.startswith("s3://") else "stream"

    with ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        first = ds.commit()

        ds.abc.append(2)
        second = ds.commit()

    ds = deeplake.load(path, access_method=access_method)

    corrupt_ds(ds, "abc", 3)
    save_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        deeplake.load(path, access_method=access_method)

    with pytest.raises(ReadOnlyModeError):
        deeplake.load(path, read_only=True, access_method=access_method, reset=True)

    ds = deeplake.load(
        path,
        reset=True,
        access_method=access_method,
    )
    verify_reset_on_checkout(ds, "main", second, save_head, {"abc": [[1], [2]]})


def test_load_corrupt_dataset_no_vc(local_path):
    ds = deeplake.empty(local_path)

    with ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        first = ds.commit()

        ds.abc.append(2)
        second = ds.commit()

    ds = deeplake.load(local_path)
    corrupt_ds(ds, "abc", 3)
    save_head = ds.pending_commit_id

    saved = json.loads(ds.storage["version_control_info.json"].decode("utf-8"))
    del ds.storage["version_control_info.json"]

    with pytest.raises(KeyError):
        ds.storage["version_control_info.json"]

    with pytest.raises(DatasetCorruptError):
        ds = deeplake.load(local_path)

    reloaded = json.loads(ds.storage["version_control_info.json"].decode("utf-8"))
    compare_version_info(saved, reloaded)

    ds = deeplake.load(local_path, reset=True)

    verify_reset_on_checkout(ds, "main", second, save_head, {"abc": [[1], [2]]})


def test_load_corrupted_branch(local_path):
    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("abc")
    ds.abc.append(1)
    main_1 = ds.commit()

    ds.abc.append(2)
    main_2 = ds.commit()

    ds.checkout("alt", create=True)

    corrupt_ds(ds, "abc", 3)
    save_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        deeplake.load(f"{local_path}@alt")

    ds = deeplake.load(f"{local_path}@alt", reset=True)
    verify_reset_on_checkout(ds, "alt", main_2, save_head, {"abc": [[1], [2]]})

    ds.abc.append(3)
    alt_1 = ds.commit()

    ds.abc.append(4)
    alt_2 = ds.commit()

    corrupt_ds(ds, "abc", 5)
    save_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        deeplake.load(f"{local_path}@alt")

    with pytest.raises(DatasetCorruptError):
        deeplake.load(f"{local_path}@{save_head}")

    ds = deeplake.load(f"{local_path}@alt", reset=True)
    verify_reset_on_checkout(ds, "alt", alt_2, save_head, {"abc": [[1], [2], [3], [4]]})


def test_checkout_corrupted_branch(local_path):
    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("abc")
    ds.abc.append(1)
    main_1 = ds.commit()

    ds.abc.append(2)
    main_2 = ds.commit()

    ds.checkout("alt", create=True)

    corrupt_ds(ds, "abc", 3)
    save_head = ds.pending_commit_id

    ds.checkout("main")

    with pytest.raises(DatasetCorruptError):
        ds.checkout("alt")

    ds.checkout("alt", reset=True)
    verify_reset_on_checkout(ds, "alt", main_2, save_head, {"abc": [[1], [2]]})

    ds.abc.append(3)

    ds.checkout("main")
    ds.checkout("alt")  # test reset persists correctly

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3]])

    alt_1 = ds.commit()

    corrupt_ds(ds, "abc", 4)
    save_head = ds.pending_commit_id

    ds.checkout("main")

    corrupt_ds(ds, "abc", 5)
    main_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        ds.checkout(save_head)

    ds.checkout(save_head, reset=True)
    verify_reset_on_checkout(ds, "alt", alt_1, save_head, {"abc": [[1], [2], [3]]})

    with pytest.raises(DatasetCorruptError):
        ds.checkout("main")

    ds.checkout("main", reset=True)
    verify_reset_on_checkout(ds, "main", main_2, main_head, {"abc": [[1], [2]]})


def test_load_corrupt_dataset_with_no_commits(local_path):
    ds = deeplake.dataset(local_path, overwrite=True)

    ds.create_tensor("abc")

    corrupt_ds(ds, "abc", 1)

    with pytest.raises(DatasetCorruptError):
        deeplake.load(local_path)

    with pytest.raises(ReadOnlyModeError):
        deeplake.load(local_path, read_only=True, reset=True)

    ds = deeplake.load(local_path, reset=True)

    assert set(ds._tensors()) == set()


def test_rebuild_vc_info(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        ds.commit()
        ds.checkout("alt1", create=True)
        ds.abc.append(2)
        ds.commit()
        ds.checkout("main")
        ds.abc.append(3)
        ds.commit()
        ds.checkout("alt2", create=True)
        ds.abc.append(4)
        ds.commit()
        ds.abc.append(5)
        ds.commit()
        ds.checkout("main")
        ds.merge("alt2")
        ds.merge("alt1")

    saved = json.loads(local_ds.storage["version_control_info.json"])
    del local_ds.storage["version_control_info.json"]

    with pytest.raises(KeyError):
        local_ds.storage["version_control_info.json"]

    rebuild_version_info(local_ds.storage)

    reloaded = json.loads(local_ds.storage["version_control_info.json"])

    compare_version_info(saved, reloaded)


def test_fix_vc(local_path):
    ds = deeplake.empty(local_path, overwrite=True)

    with ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        ds.commit()
        ds.checkout("alt", create=True)
        ds.abc.append(2)
        ds.commit()
        ds.checkout("main")
        ds.abc.append(3)
        ds.commit()

    saved = json.loads(ds.storage["version_control_info.json"].decode("utf-8"))

    for commit_id, commit in saved["commits"].items():
        commit["children"] = [
            c for c in commit["children"] if saved["commits"][c]["branch"] != "alt"
        ]
    alt_id = saved["branches"].pop("alt")
    del saved["commits"][alt_id]
    saved["commits"] = dict(
        filter(
            lambda x: x[0] != alt_id and x[1]["branch"] != "alt",
            saved["commits"].items(),
        )
    )

    ds.storage["version_control_info.json"] = json.dumps(saved).encode("utf-8")
    ds.storage.flush()

    ds = deeplake.load(local_path)

    with pytest.raises(CheckoutError):
        ds.checkout("alt")

    ds.fix_vc()

    ds.checkout("alt")

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2]])


def test_missing_commit_infos(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        a = ds.commit()
        ds.abc.append(2)
        b = ds.commit()
        ds.checkout("alt", create=True)
        ds.abc.append(3)
        c = ds.commit()
        ds.abc.append(4)
        d = ds.commit()
        ds.abc.append(5)

    del ds.storage["version_control_info.json"]
    del ds.storage[get_commit_info_key(d)]
    ds.storage.flush()

    ds = deeplake.load(local_ds.path)

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2]])

    ds.checkout("alt")

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3]])

    assert ds.commit_id == c


def test_dataset_with_no_commits_unaffected(local_path):
    ds = deeplake.empty(local_path, overwrite=True)

    ds.create_tensor("abc")
    ds.abc.append(1)

    del ds.storage["version_control_info.json"]
    ds.storage.flush()

    ds = deeplake.load(local_path)

    np.testing.assert_array_equal(ds.abc.numpy(), [[1]])


def test_load_corrupt_dataset_no_meta(local_path):
    ds = deeplake.empty(local_path, overwrite=True)

    with ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        a = ds.commit()

        ds.abc.append(2)
        b = ds.commit()

        del ds.storage["dataset_meta.json"]

    ds = deeplake.load(local_path)
    assert ds.commit_id == b
    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2]])

    with pytest.raises(CheckoutError):
        ds.checkout(a)
