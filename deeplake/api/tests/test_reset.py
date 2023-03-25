from deeplake.util.exceptions import DatasetCorruptError, ReadOnlyModeError

import numpy as np

import deeplake
import pytest


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


@pytest.mark.parametrize("path", ["local_path", "s3_path"], indirect=True)
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
        ds = deeplake.load(path, access_method=access_method)

    with pytest.raises(ReadOnlyModeError):
        ds = deeplake.load(
            path, read_only=True, access_method=access_method, reset=True
        )

    ds = deeplake.load(
        path,
        reset=True,
        access_method=access_method,
    )
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
        ds = deeplake.load(f"{local_path}@alt")

    ds = deeplake.load(f"{local_path}@alt", reset=True)
    verify_reset_on_checkout(ds, "alt", main_2, save_head, {"abc": [[1], [2]]})

    ds.abc.append(3)
    alt_1 = ds.commit()

    ds.abc.append(4)
    alt_2 = ds.commit()

    corrupt_ds(ds, "abc", 5)
    save_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        ds = deeplake.load(f"{local_path}@alt")

    with pytest.raises(DatasetCorruptError):
        ds = deeplake.load(f"{local_path}@{save_head}")

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
        ds = deeplake.load(local_path)

    with pytest.raises(ReadOnlyModeError):
        ds = deeplake.load(local_path, read_only=True, reset=True)

    ds = deeplake.load(local_path, reset=True)

    assert set(ds._tensors()) == set()
