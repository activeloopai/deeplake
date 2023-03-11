from deeplake.util.exceptions import DatasetCorruptError

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


def test_load_corrupt_dataset(local_path):
    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("abc")
    ds.abc.append(1)
    first = ds.commit()

    ds.abc.append(2)
    second = ds.commit()

    corrupt_ds(ds, "abc", 3)
    save_head = ds.pending_commit_id

    with pytest.raises(DatasetCorruptError):
        ds = deeplake.load(local_path)

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
