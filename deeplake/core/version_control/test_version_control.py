import glob
import json
from collections import OrderedDict

from deeplake.constants import FIRST_COMMIT_ID

import deeplake
import pytest
import numpy as np
from deeplake.util.diff import (
    get_all_changes_string,
    get_lowest_common_ancestor,
    sanitize_commit,
)
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.exceptions import (
    CheckoutError,
    CommitError,
    ReadOnlyModeError,
    InfoError,
    TensorModifiedError,
    EmptyCommitError,
    VersionControlError,
)

NO_COMMIT_PASSED_DIFF = ""
ONE_COMMIT_PASSED_DIFF = "The 2 diffs are calculated relative to the most recent common ancestor (%s) of the current state and the commit passed."
TWO_COMMIT_PASSED_DIFF = "The 2 diffs are calculated relative to the most recent common ancestor (%s) of the two commits passed."


def get_lca_id_helper(version_state, id_1, id_2=None):
    id_1 = sanitize_commit(id_1, version_state)
    id_2 = sanitize_commit(id_2, version_state) if id_2 else version_state["commit_id"]
    commit_node_1 = version_state["commit_node_map"][id_1]
    commit_node_2 = version_state["commit_node_map"][id_2]
    return get_lowest_common_ancestor(commit_node_1, commit_node_2)


def commit_details_helper(commits, ds):
    for commit in commits:
        assert ds.get_commit_details(commit["commit"]) == commit


def get_default_tensor_diff():
    return {
        "created": False,
        "cleared": False,
        "info_updated": False,
        "data_added": [0, 0],
        "data_updated": set(),
        "data_deleted": set(),
        "data_transformed_in_place": False,
    }


def get_default_dataset_diff(commit_id):
    return {
        "commit_id": commit_id,
        "info_updated": False,
        "renamed": OrderedDict(),
        "deleted": [],
    }


def get_diff_helper(
    ds_changes_1,
    ds_changes_2,
    tensor_changes_1,
    tensor_changes_2,
    version_state=None,
    commit_1=None,
    commit_2=None,
):
    if commit_1 and commit_2:
        lca_id = get_lca_id_helper(version_state, commit_1, commit_2)
        message0 = TWO_COMMIT_PASSED_DIFF % lca_id
        message1 = f"Diff in {commit_1} (target id 1):\n"
        message2 = f"Diff in {commit_2} (target id 2):\n"
    elif commit_1:
        lca_id = get_lca_id_helper(version_state, commit_1)
        message0 = ONE_COMMIT_PASSED_DIFF % lca_id
        message1 = "Diff in HEAD:\n"
        message2 = f"Diff in {commit_1} (target id):\n"
    else:
        message0 = NO_COMMIT_PASSED_DIFF
        message1 = "Diff in HEAD relative to the previous commit:\n"
        message2 = ""

    target = (
        get_all_changes_string(
            ds_changes_1,
            ds_changes_2,
            tensor_changes_1,
            tensor_changes_2,
            message0,
            message1,
            message2,
        )
        + "\n"
    )

    return target


def compare_tensor_dict(d1, d2):
    for key in d1:
        if key == "data_added" and d1[key] != d2[key]:
            assert d1[key][1] - d1[key][0] == 0
            assert d2[key][1] - d2[key][0] == 0
        else:
            assert d1[key] == d2[key]


def compare_tensor_diff(diff1, diff2):
    for commit_diff1, commit_diff2 in zip(diff1, diff2):
        for key in commit_diff1:
            if key == "commit_id":
                assert commit_diff1[key] == commit_diff2[key]
            else:
                compare_tensor_dict(commit_diff1[key], commit_diff2[key])


def compare_dataset_diff(diff1, diff2):
    ignore_keys = ["author", "date", "message"]
    assert len(diff1) == len(diff2)
    for commit_diff1, commit_diff2 in zip(diff1, diff2):
        for key in commit_diff1:
            if key not in ignore_keys:
                assert commit_diff1[key] == commit_diff2[key]


def test_commit(local_ds):
    with local_ds:
        local_ds.create_tensor("abc")
        local_ds.abc.append(1)
        local_ds.log()
        a = local_ds.commit("first")
        local_ds.abc[0] = 2
        b = local_ds.commit("second")
        local_ds.abc[0] = 3
        c = local_ds.commit("third")
        assert local_ds.abc[0].numpy() == 3
        local_ds.checkout(a)
        assert local_ds.commit_id == a
        assert local_ds.abc[0].numpy() == 1
        local_ds.checkout(b)
        assert local_ds.commit_id == b
        assert local_ds.abc[0].numpy() == 2
        local_ds.checkout(c)
        assert local_ds.commit_id == c
        assert local_ds.branch == "main"
        assert local_ds.abc[0].numpy() == 3
        with pytest.raises(CheckoutError):
            local_ds.checkout("main", create=True)
        with pytest.raises(CheckoutError):
            local_ds.checkout(a, create=True)


"""
test for checking unchanged dataset commits
"""


def test_unchanged_commit(local_ds):
    with local_ds:
        local_ds.create_tensor("abc")
        local_ds.abc.append(1)
        local_ds.log()
        a = local_ds.commit("first")
        local_ds.checkout(a)
        assert local_ds.commit_id == a
        with pytest.raises(EmptyCommitError):
            b = local_ds.commit("second")
        c = local_ds.commit("third", allow_empty=True)
        local_ds.checkout(c)
        assert local_ds.commit_id == c


def test_commit_checkout(local_ds):
    with local_ds:
        local_ds.create_tensor("img")
        local_ds.img.extend(np.ones((10, 100, 100, 3)))
        first_commit_id = local_ds.commit("stored all ones")

        for i in range(5):
            local_ds.img[i] *= 2
        second_commit_id = local_ds.commit("multiplied value of some images by 2")

        for i in range(5):
            assert (local_ds.img[i].numpy() == 2 * np.ones((100, 100, 3))).all()
        local_ds.checkout(first_commit_id)  # now all images are ones again

        for i in range(10):
            assert (local_ds.img[i].numpy() == np.ones((100, 100, 3))).all()

        local_ds.checkout("alternate", create=True)
        assert local_ds.branch == "alternate"

        for i in range(5):
            local_ds.img[i] *= 3
        local_ds.commit("multiplied value of some images by 3")

        for i in range(5):
            assert (local_ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()

        local_ds.checkout(second_commit_id)  # first 5 images are 2s, rest are 1s now
        assert local_ds.commit_id == second_commit_id
        assert local_ds.branch == "main"

        # we are not at the head of master but rather at the last commit, so we automatically get checked out to a new branch here
        for i in range(5, 10):
            local_ds.img[i] *= 2
        local_ds.commit("multiplied value of remaining images by 2")

        for i in range(10):
            assert (local_ds.img[i].numpy() == 2 * np.ones((100, 100, 3))).all()

        local_ds.checkout("alternate")

        for i in range(5, 10):
            local_ds.img[i] *= 3

        for i in range(10):
            assert (local_ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()
        local_ds.commit("multiplied value of remaining images by 3")
        for i in range(10):
            assert (local_ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()


def test_commit_checkout_2(local_ds):
    local_ds.create_tensor("abc")
    local_ds.create_tensor("img")
    for i in range(10):
        local_ds.img.append(i * np.ones((100, 100, 3)))
    a = local_ds.commit("first")

    local_ds.img[7] *= 2

    assert (local_ds.img[6].numpy() == 6 * np.ones((100, 100, 3))).all()
    assert (local_ds.img[7].numpy() == 2 * 7 * np.ones((100, 100, 3))).all()
    assert (local_ds.img[8].numpy() == 8 * np.ones((100, 100, 3))).all()
    assert (local_ds.img[9].numpy() == 9 * np.ones((100, 100, 3))).all()

    assert (local_ds.img[2].numpy() == 2 * np.ones((100, 100, 3))).all()

    b = local_ds.commit("second")

    # going back to first commit
    local_ds.checkout(a)

    assert (local_ds.img[7].numpy() == 7 * np.ones((100, 100, 3))).all()

    local_ds.checkout("another", create=True)

    local_ds.img[7] *= 3

    # and not 6 * 7 as it would have been, had we checked out from b
    assert (local_ds.img[7].numpy() == 3 * 7 * np.ones((100, 100, 3))).all()

    local_ds.commit("first2")

    local_ds.checkout("main")
    assert (local_ds.img[7].numpy() == 2 * 7 * np.ones((100, 100, 3))).all()
    local_ds.log()


def test_auto_checkout_bug(local_ds):
    local_ds.create_tensor("abc")
    local_ds.abc.extend([1, 2, 3, 4, 5])
    a = local_ds.commit("it is 1")
    local_ds.abc[0] = 2
    b = local_ds.commit("it is 2")
    c = local_ds.checkout(a)
    local_ds.checkout("other", True)
    d = local_ds.pending_commit_id
    local_ds.abc[0] = 3
    e = local_ds.commit("it is 3")
    local_ds.checkout(b)
    local_ds.abc[0] = 4
    f = local_ds.commit("it is 4")
    g = local_ds.checkout(a)
    local_ds.abc[0] = 5
    dsv = local_ds[0:3]
    h = local_ds.commit("it is 5")
    i = local_ds.checkout(e)
    local_ds.abc[0] = 6
    tsv = local_ds.abc[0:5]
    tsv[0] = 6
    j = local_ds.commit("it is 6")
    local_ds.log()
    local_ds.checkout(a)
    assert dsv.abc[0].numpy() == 5
    assert local_ds.abc[0].numpy() == 1
    local_ds.checkout(b)
    assert local_ds.abc[0].numpy() == 2
    local_ds.checkout(c)
    assert local_ds.abc[0].numpy() == 1
    local_ds.checkout(d)
    assert local_ds.abc[0].numpy() == 3
    local_ds.checkout(e)
    assert local_ds.abc[0].numpy() == 3
    local_ds.checkout(f)
    assert local_ds.abc[0].numpy() == 4
    local_ds.checkout(g)
    assert local_ds.abc[0].numpy() == 1
    local_ds.checkout(h)
    assert local_ds.abc[0].numpy() == 5
    local_ds.checkout(i)
    assert local_ds.abc[0].numpy() == 3
    local_ds.checkout(j)
    assert local_ds.abc[0].numpy() == 6
    local_ds.checkout("main")
    assert local_ds.abc[0].numpy() == 2
    local_ds.abc[0] = 7
    local_ds.checkout("copy", True)
    assert local_ds.abc[0].numpy() == 7
    local_ds.checkout("other")
    assert local_ds.abc[0].numpy() == 3


def test_read_mode(local_ds):
    base_storage = get_base_storage(local_ds.storage)
    base_storage.enable_readonly()
    ds = deeplake.Dataset(storage=local_ds.storage, read_only=True, verbose=False)
    with pytest.raises(ReadOnlyModeError):
        ds.commit("first")
    with pytest.raises(ReadOnlyModeError):
        ds.checkout("third", create=True)


def test_checkout_address_not_found(local_ds):
    with pytest.raises(CheckoutError):
        local_ds.checkout("second")


def test_dynamic(local_ds):
    local_ds.create_tensor("img")
    for i in range(10):
        local_ds.img.append(i * np.ones((100, 100, 3)))

    a = local_ds.commit("first")
    for i in range(10):
        local_ds.img[i] = 2 * i * np.ones((150, 150, 3))
    local_ds.checkout(a)

    for i in range(10):
        assert (local_ds.img[i].numpy() == i * np.ones((100, 100, 3))).all()

    local_ds.checkout("main")

    for i in range(10):
        assert (local_ds.img[i].numpy() == 2 * i * np.ones((150, 150, 3))).all()


def test_different_lengths(local_ds):
    with local_ds:
        local_ds.create_tensor("img")
        local_ds.create_tensor("abc")
        local_ds.img.extend(np.ones((5, 50, 50)))
        local_ds.abc.extend(np.ones((2, 10, 10)))
        first = local_ds.commit("stored 5 images, 2 abc")
        local_ds.img.extend(np.ones((3, 50, 50)))
        second = local_ds.commit("stored 3 more images")
        assert len(local_ds.tensors) == 2
        assert len(local_ds.img) == 8
        assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(local_ds.abc) == 2
        assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
        local_ds.checkout(first)
        assert len(local_ds.tensors) == 2
        assert len(local_ds.img) == 5
        assert (local_ds.img.numpy() == np.ones((5, 50, 50))).all()
        assert len(local_ds.abc) == 2
        assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()

        # will autocheckout to new branch
        local_ds.create_tensor("ghi")
        local_ds.ghi.extend(np.ones((2, 10, 10)))
        local_ds.img.extend(np.ones((2, 50, 50)))
        local_ds.abc.extend(np.ones((3, 10, 10)))
        assert len(local_ds.tensors) == 3
        assert len(local_ds.img) == 7
        assert (local_ds.img.numpy() == np.ones((7, 50, 50))).all()
        assert len(local_ds.abc) == 5
        assert (local_ds.abc.numpy() == np.ones((5, 10, 10))).all()
        assert len(local_ds.ghi) == 2
        assert (local_ds.ghi.numpy() == np.ones((2, 10, 10))).all()
        third = local_ds.commit(
            "stored 2 more images, 3 more abc in other branch, created ghi"
        )
        local_ds.checkout(first)
        assert len(local_ds.tensors) == 2
        assert len(local_ds.img) == 5
        assert (local_ds.img.numpy() == np.ones((5, 50, 50))).all()
        assert len(local_ds.abc) == 2
        assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
        local_ds.checkout(second)
        assert len(local_ds.tensors) == 2
        assert len(local_ds.img) == 8
        assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(local_ds.abc) == 2
        assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
        local_ds.checkout(third)
        assert len(local_ds.tensors) == 3
        assert len(local_ds.img) == 7
        assert (local_ds.img.numpy() == np.ones((7, 50, 50))).all()
        assert len(local_ds.abc) == 5
        assert (local_ds.abc.numpy() == np.ones((5, 10, 10))).all()
        local_ds.checkout("main")
        assert len(local_ds.tensors) == 2
        assert len(local_ds.img) == 8
        assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(local_ds.abc) == 2
        assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()

    path = local_ds.path
    if path.startswith("mem://"):
        # memory datasets are not persistent
        return

    # reloading the dataset to check persistence

    local_ds = deeplake.dataset(path)
    assert len(local_ds.tensors) == 2
    assert len(local_ds.img) == 8
    assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(local_ds.abc) == 2
    assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
    local_ds.checkout(first)
    assert len(local_ds.tensors) == 2
    assert len(local_ds.img) == 5
    assert (local_ds.img.numpy() == np.ones((5, 50, 50))).all()
    assert len(local_ds.abc) == 2
    assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
    local_ds.checkout(second)
    assert len(local_ds.tensors) == 2
    assert len(local_ds.img) == 8
    assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(local_ds.abc) == 2
    assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()
    local_ds.checkout(third)
    assert len(local_ds.tensors) == 3
    assert len(local_ds.img) == 7
    assert (local_ds.img.numpy() == np.ones((7, 50, 50))).all()
    assert len(local_ds.abc) == 5
    assert (local_ds.abc.numpy() == np.ones((5, 10, 10))).all()
    local_ds.checkout("main")
    assert len(local_ds.tensors) == 2
    assert len(local_ds.img) == 8
    assert (local_ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(local_ds.abc) == 2
    assert (local_ds.abc.numpy() == np.ones((2, 10, 10))).all()


def test_auto_checkout(local_ds):
    # auto checkout happens when write operations are performed on non head commits
    local_ds.create_tensor("abc")
    first = local_ds.commit("created abc")

    local_ds.checkout(first)
    assert local_ds.branch == "main"
    local_ds.create_tensor("def")
    assert local_ds.branch != "main"

    local_ds.checkout(first)
    assert local_ds.branch == "main"
    local_ds.abc.append(1)
    assert local_ds.branch != "main"

    local_ds.checkout(first)
    assert local_ds.branch == "main"
    local_ds.abc.extend([1])
    assert local_ds.branch != "main"

    local_ds.checkout(first)
    assert local_ds.branch == "main"

    with pytest.raises(InfoError):
        local_ds.info[5] = 5

    assert local_ds.branch == "main"


def test_auto_commit(local_ds):
    initial_commit_id = local_ds.pending_commit_id
    # auto commit as head of main branch and has changes (due to creation of dataset)
    local_ds.checkout("pqr", create=True)
    local_ds.checkout("main")
    second_commit_id = local_ds.pending_commit_id
    assert second_commit_id != initial_commit_id
    assert local_ds.commit_id == initial_commit_id
    local_ds.create_tensor("abc")
    local_ds.abc.append(1)
    # auto commit as head of main and data present
    local_ds.checkout("xyz", create=True)
    local_ds.checkout("main")

    assert local_ds.pending_commit_id != second_commit_id
    assert local_ds.commit_id == second_commit_id

    with local_ds:
        local_ds.abc.append(1)

    third_commit_id = local_ds.pending_commit_id

    # auto commit as head of main and data present
    local_ds.checkout("tuv", create=True)
    local_ds.checkout("main")

    assert local_ds.pending_commit_id != third_commit_id
    assert local_ds.commit_id == third_commit_id


def test_dataset_info(local_ds):
    assert len(local_ds.info) == 0
    local_ds.info.key = "value"
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "value"

    a = local_ds.commit("added key, value", allow_empty=True)
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "value"

    local_ds.info.key2 = "value2"
    assert len(local_ds.info) == 2
    assert local_ds.info.key == "value"
    assert local_ds.info.key2 == "value2"

    b = local_ds.commit("added key2, value2", allow_empty=True)
    assert len(local_ds.info) == 2
    assert local_ds.info.key == "value"
    assert local_ds.info.key2 == "value2"

    local_ds.checkout(a)
    assert local_ds.info.key == "value"

    local_ds.checkout("alt", create=True)
    local_ds.info.key = "notvalue"
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "notvalue"
    c = local_ds.commit("changed key to notvalue", allow_empty=True)

    local_ds.checkout(a)
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "value"

    local_ds.checkout(b)
    assert len(local_ds.info) == 2
    assert local_ds.info.key == "value"
    assert local_ds.info.key2 == "value2"

    local_ds.checkout("alt")
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "notvalue"

    local_ds.checkout(c)
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "notvalue"


def test_tensor_info(local_ds):
    local_ds.create_tensor("abc")
    assert len(local_ds.abc.info) == 0
    local_ds.abc.info.key = "value"
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "value"

    a = local_ds.commit("added key, value")
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "value"

    local_ds.abc.info.key2 = "value2"
    assert len(local_ds.abc.info) == 2
    assert local_ds.abc.info.key == "value"
    assert local_ds.abc.info.key2 == "value2"

    b = local_ds.commit("added key2, value2")
    assert len(local_ds.abc.info) == 2
    assert local_ds.abc.info.key == "value"
    assert local_ds.abc.info.key2 == "value2"

    local_ds.checkout(a)
    assert local_ds.abc.info.key == "value"

    local_ds.checkout("alt", create=True)
    local_ds.abc.info.key = "notvalue"
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "notvalue"
    c = local_ds.commit("changed key to notvalue")

    local_ds.checkout(a)
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "value"

    local_ds.checkout(b)
    assert len(local_ds.abc.info) == 2
    assert local_ds.abc.info.key == "value"
    assert local_ds.abc.info.key2 == "value2"

    local_ds.checkout("alt")
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "notvalue"

    local_ds.checkout(c)
    assert len(local_ds.abc.info) == 1
    assert local_ds.abc.info.key == "notvalue"


def test_delete(local_ds):
    with local_ds:
        local_ds.create_tensor("abc")
        local_ds.abc.append(1)
        a = local_ds.commit("first")
        local_ds.delete_tensor("abc")
        b = local_ds.commit("second", allow_empty=True)
        local_ds.checkout(a)
        assert local_ds.abc[0].numpy() == 1
        local_ds.checkout(b)
        assert local_ds.tensors == {}

        local_ds.create_tensor("x/y/z")
        local_ds["x/y/z"].append(1)
        c = local_ds.commit("third")
        local_ds["x"].delete_tensor("y/z")
        d = local_ds.commit("fourth", allow_empty=True)
        local_ds.checkout(c)
        assert local_ds["x/y/z"][0].numpy() == 1
        local_ds.checkout(d)
        assert local_ds.tensors == {}
        assert list(local_ds.groups) == ["x"]
        local_ds.delete_group("x")
        assert list(local_ds.groups) == []

        local_ds.checkout(c)
        local_ds["x"].delete_group("y")
        assert local_ds.tensors == {}
        assert list(local_ds.groups) == ["x"]

        local_ds.checkout(c)
        local_ds.delete_group("x/y")
        assert local_ds.tensors == {}
        assert list(local_ds.groups) == ["x"]


def test_tensor_rename(local_ds):
    with local_ds:
        local_ds.create_tensor("x/y/z")
        local_ds["x/y/z"].append(1)
        local_ds["x/y"].rename_tensor("z", "a")
        a = local_ds.commit("first")

        assert local_ds["x/y/a"][0].numpy() == 1
        local_ds["x/y/a"].append(2)
        local_ds["x"].rename_tensor("y/a", "y/z")
        b = local_ds.commit("second")

        assert local_ds["x/y/z"][1].numpy() == 2
        local_ds.create_tensor("x/y/a")
        local_ds["x/y/a"].append(3)
        local_ds["x/y"].rename_tensor("z", "b")
        c = local_ds.commit("third")

        local_ds.checkout(a)
        assert local_ds["x/y/a"][0].numpy() == 1

        local_ds.checkout(b)
        assert local_ds["x/y/z"][1].numpy() == 2

        local_ds.checkout(c)
        assert local_ds["x/y/a"][0].numpy() == 3
        assert local_ds["x/y/b"][1].numpy() == 2


def test_dataset_diff(local_ds, capsys):
    local_ds.create_tensor("abc")
    a = local_ds.commit()
    expected_tensor_diff_from_a = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds.rename_tensor("abc", "xyz")
    expected_dataset_diff_from_a["renamed"] = OrderedDict({"abc": "xyz"})
    local_ds.info["hello"] = "world"
    expected_dataset_diff_from_a["info_updated"] = True

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [expected_tensor_diff_from_a]
    expected_dataset_diff = [expected_dataset_diff_from_a]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)
    local_ds.diff()
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    b = local_ds.commit()
    expected_tensor_diff_from_b = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_b = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds.delete_tensor("xyz")
    expected_tensor_diff_from_b.pop("xyz")
    expected_dataset_diff_from_b["deleted"].append("xyz")

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = (
        [expected_tensor_diff_from_b, expected_tensor_diff_from_a],
        [],
    )
    expected_dataset_diff = (
        [expected_dataset_diff_from_b, expected_dataset_diff_from_a],
        [],
    )
    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    # cover DatasetDiff.frombuffer
    ds = deeplake.load(local_ds.path)
    ds.diff(a)
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()
    diff = ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])


def test_clear_diff(local_ds, capsys):
    with local_ds:
        expected_tensor_diff_from_start = {
            "commit_id": local_ds.pending_commit_id,
            "abc": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_start = get_default_dataset_diff(
            local_ds.pending_commit_id
        )
        local_ds.create_tensor("abc")
        expected_tensor_diff_from_start["abc"]["created"] = True
        local_ds.abc.append([1, 2, 3])
        expected_tensor_diff_from_start["abc"]["data_added"] = [0, 1]
        local_ds.abc.clear()
        expected_tensor_diff_from_start["abc"]["cleared"] = True
        expected_tensor_diff_from_start["abc"]["data_added"] = [0, 0]
        local_ds.abc.append([4, 5, 6])
        expected_tensor_diff_from_start["abc"]["data_added"] = [0, 1]
        local_ds.abc.append([1, 2, 3])
        expected_tensor_diff_from_start["abc"]["data_added"] = [0, 2]

    target_dataset_diff = [expected_dataset_diff_from_start]
    target_tensor_diff = [expected_tensor_diff_from_start]
    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    assert tensor_diff == target_tensor_diff
    compare_dataset_diff(dataset_diff, target_dataset_diff)

    local_ds.diff()
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)

    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    a = local_ds.commit()
    expected_tensor_diff_from_a = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
    }

    expected_dataset_diff_from_a = get_default_dataset_diff(local_ds.pending_commit_id)

    with local_ds:
        local_ds.abc.append([3, 4, 5])
        expected_tensor_diff_from_a["abc"]["data_added"] = [2, 3]
        local_ds.create_tensor("xyz")
        expected_tensor_diff_from_a["xyz"] = get_default_tensor_diff()
        expected_tensor_diff_from_a["xyz"]["created"] = True
        local_ds.abc.clear()
        expected_tensor_diff_from_a["abc"]["cleared"] = True
        expected_tensor_diff_from_a["abc"]["data_added"] = [0, 0]
        local_ds.abc.append([1, 0, 0])
        expected_tensor_diff_from_a["abc"]["data_added"] = [0, 1]
        local_ds.xyz.append([0, 1, 0])
        expected_tensor_diff_from_a["xyz"]["data_added"] = [0, 1]

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = ([expected_tensor_diff_from_a], [])
    expected_dataset_diff = ([expected_dataset_diff_from_a], [])

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    b = local_ds.commit()
    expected_tensor_diff_from_b = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_b = get_default_dataset_diff(local_ds.pending_commit_id)

    with local_ds:
        local_ds.xyz.append([0, 0, 1])
        expected_tensor_diff_from_b["xyz"]["data_added"] = [1, 2]
        local_ds.xyz.clear()
        expected_tensor_diff_from_b["xyz"]["cleared"] = True
        expected_tensor_diff_from_b["xyz"]["data_added"] = [0, 0]
        local_ds.xyz.append([1, 2, 3])
        expected_tensor_diff_from_b["xyz"]["data_added"] = [0, 1]
        local_ds.abc.append([3, 4, 2])
        expected_tensor_diff_from_b["abc"]["data_added"] = [1, 2]
    c = local_ds.commit()

    diff = local_ds.diff(c, a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = (
        [expected_tensor_diff_from_b, expected_tensor_diff_from_a],
        [],
    )
    expected_dataset_diff = (
        [expected_dataset_diff_from_b, expected_dataset_diff_from_a],
        [],
    )

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(c, a)

    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        c,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()


def test_delete_diff(local_ds, capsys):
    local_ds.create_tensor("x/y/z")
    local_ds["x/y/z"].append([4, 5, 6])
    a = local_ds.commit()
    expected_tensor_diff_from_a = {
        "commit_id": local_ds.pending_commit_id,
        "x/y/z": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds.create_tensor("a/b/c")
    expected_tensor_diff_from_a["a/b/c"] = get_default_tensor_diff()
    expected_tensor_diff_from_a["a/b/c"]["created"] = True
    b = local_ds.commit()
    expected_tensor_diff_from_b = {
        "commit_id": local_ds.pending_commit_id,
        "a/b/c": get_default_tensor_diff(),
        "x/y/z": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_b = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds["a/b/c"].append([1, 2, 3])
    expected_tensor_diff_from_b["a/b/c"]["data_added"] = [0, 1]
    c = local_ds.commit()
    expected_tensor_diff_from_c = {
        "commit_id": local_ds.pending_commit_id,
        "a/b/c": get_default_tensor_diff(),
        "x/y/z": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_c = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds.delete_tensor("a/b/c")
    expected_dataset_diff_from_c["deleted"].append("a/b/c")
    expected_tensor_diff_from_c.pop("a/b/c")

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = (
        [
            expected_tensor_diff_from_c,
            expected_tensor_diff_from_b,
            expected_tensor_diff_from_a,
        ],
        [],
    )
    expected_dataset_diff = (
        [
            expected_dataset_diff_from_c,
            expected_dataset_diff_from_b,
            expected_dataset_diff_from_a,
        ],
        [],
    )

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    d = local_ds.commit()
    expected_tensor_diff_from_d = {
        "commit_id": local_ds.pending_commit_id,
        "x/y/z": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_d = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds["x/y/z"][0] = [1, 3, 4]
    expected_tensor_diff_from_d["x/y/z"]["data_updated"] = {0}
    e = local_ds.commit()
    expected_tensor_diff_from_e = {
        "commit_id": local_ds.pending_commit_id,
        "x/y/z": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_e = get_default_dataset_diff(local_ds.pending_commit_id)
    local_ds.create_tensor("a/b/c")
    expected_tensor_diff_from_e["a/b/c"] = get_default_tensor_diff()
    expected_tensor_diff_from_e["a/b/c"]["created"] = True
    local_ds.delete_tensor("x/y/z")
    expected_dataset_diff_from_e["deleted"].append("x/y/z")
    expected_tensor_diff_from_e.pop("x/y/z")

    diff = local_ds.diff(c, as_dict=True)

    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = (
        [
            expected_tensor_diff_from_e,
            expected_tensor_diff_from_d,
            expected_tensor_diff_from_c,
        ],
        [],
    )
    expected_dataset_diff = (
        [
            expected_dataset_diff_from_e,
            expected_dataset_diff_from_d,
            expected_dataset_diff_from_c,
        ],
        [],
    )

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(c)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        c,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = (
        [
            expected_tensor_diff_from_e,
            expected_tensor_diff_from_d,
            expected_tensor_diff_from_c,
            expected_tensor_diff_from_b,
            expected_tensor_diff_from_a,
        ],
        [],
    )
    expected_dataset_diff = (
        [
            expected_dataset_diff_from_e,
            expected_dataset_diff_from_d,
            expected_dataset_diff_from_c,
            expected_dataset_diff_from_b,
            expected_dataset_diff_from_a,
        ],
        [],
    )

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [expected_tensor_diff_from_e]
    expected_dataset_diff = [expected_dataset_diff_from_e]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)


def test_rename_diff_single(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("abc")
        expect_tensor_diff_from_start = {
            "commit_id": local_ds.pending_commit_id,
            "abc": get_default_tensor_diff(),
        }
        expect_tensor_diff_from_start["abc"]["created"] = True
        expect_dataset_diff_from_start = get_default_dataset_diff(
            local_ds.pending_commit_id
        )
        local_ds.abc.append([1, 2, 3])
        expect_tensor_diff_from_start["abc"]["data_added"] = [0, 1]
        local_ds.rename_tensor("abc", "xyz")
        expect_tensor_diff_from_start["xyz"] = expect_tensor_diff_from_start.pop("abc")
        local_ds.xyz.append([2, 3, 4])
        expect_tensor_diff_from_start["xyz"]["data_added"] = [0, 2]
        local_ds.rename_tensor("xyz", "efg")
        expect_tensor_diff_from_start["efg"] = expect_tensor_diff_from_start.pop("xyz")
        local_ds.create_tensor("red")
        expect_tensor_diff_from_start["red"] = get_default_tensor_diff()
        expect_tensor_diff_from_start["red"]["created"] = True

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [expect_tensor_diff_from_start]
    expected_dataset_diff = [expect_dataset_diff_from_start]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    local_ds.diff()
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    a = local_ds.commit()
    expected_tensor_diff_from_a = {
        "commit_id": local_ds.pending_commit_id,
        "efg": get_default_tensor_diff(),
        "red": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a = get_default_dataset_diff(local_ds.pending_commit_id)

    with local_ds:
        local_ds.rename_tensor("red", "blue")
        expected_dataset_diff_from_a["renamed"]["red"] = "blue"
        expected_tensor_diff_from_a["blue"] = expected_tensor_diff_from_a.pop("red")

        local_ds.efg.append([3, 4, 5])
        expected_tensor_diff_from_a["efg"]["data_added"] = [2, 3]
        local_ds.rename_tensor("efg", "bcd")
        expected_dataset_diff_from_a["renamed"]["efg"] = "bcd"
        expected_tensor_diff_from_a["bcd"] = expected_tensor_diff_from_a.pop("efg")
        local_ds.bcd[1] = [2, 5, 4]
        expected_tensor_diff_from_a["bcd"]["data_updated"].add(1)
        local_ds.rename_tensor("bcd", "red")
        expected_dataset_diff_from_a["renamed"]["efg"] = "red"
        expected_tensor_diff_from_a["red"] = expected_tensor_diff_from_a.pop("bcd")
        local_ds.red.append([1, 3, 4])
        expected_tensor_diff_from_a["red"]["data_added"] = [2, 4]
        local_ds.blue.append([2, 3, 4])
        expected_tensor_diff_from_a["blue"]["data_added"] = [0, 1]
        local_ds.rename_tensor("blue", "efg")
        expected_dataset_diff_from_a["renamed"]["red"] = "efg"
        expected_tensor_diff_from_a["efg"] = expected_tensor_diff_from_a.pop("blue")

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [[expected_tensor_diff_from_a], []]
    expected_dataset_diff = [[expected_dataset_diff_from_a], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()


def test_rename_diff_linear(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("abc")
        local_ds.abc.append([1, 2, 3])
        local_ds.create_tensor("xyz")

    a = local_ds.commit()
    expected_tensor_diff_from_a = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a = get_default_dataset_diff(local_ds.pending_commit_id)
    with local_ds:
        local_ds.create_tensor("red")
        expected_tensor_diff_from_a["red"] = get_default_tensor_diff()
        expected_tensor_diff_from_a["red"]["created"] = True
        local_ds.xyz.append([3, 4, 5])
        expected_tensor_diff_from_a["xyz"]["data_added"] = [0, 1]
        local_ds.rename_tensor("xyz", "efg")
        expected_dataset_diff_from_a["renamed"]["xyz"] = "efg"
        expected_tensor_diff_from_a["efg"] = expected_tensor_diff_from_a.pop("xyz")
        local_ds.rename_tensor("abc", "xyz")
        expected_dataset_diff_from_a["renamed"]["abc"] = "xyz"
        expected_tensor_diff_from_a["xyz"] = expected_tensor_diff_from_a.pop("abc")
        local_ds.xyz[0] = [2, 3, 4]
        expected_tensor_diff_from_a["xyz"]["data_updated"].add(0)

    b = local_ds.commit()
    expected_tensor_diff_from_b = {
        "commit_id": local_ds.pending_commit_id,
        "efg": get_default_tensor_diff(),
        "xyz": get_default_tensor_diff(),
        "red": get_default_tensor_diff(),
    }

    expected_dataset_diff_from_b = get_default_dataset_diff(local_ds.pending_commit_id)
    with local_ds:
        local_ds.rename_tensor("red", "blue")
        expected_dataset_diff_from_b["renamed"]["red"] = "blue"
        expected_tensor_diff_from_b["blue"] = expected_tensor_diff_from_b.pop("red")
        local_ds.xyz.append([5, 6, 7])
        expected_tensor_diff_from_b["xyz"]["data_added"] = [1, 2]
        local_ds.xyz.info["hello"] = "world"
        expected_tensor_diff_from_b["xyz"]["info_updated"] = True
        local_ds.rename_tensor("efg", "abc")
        expected_dataset_diff_from_b["renamed"]["efg"] = "abc"
        expected_tensor_diff_from_b["abc"] = expected_tensor_diff_from_b.pop("efg")
        local_ds.abc.append([6, 7, 8])
        expected_tensor_diff_from_b["abc"]["data_added"] = [1, 2]

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [expected_tensor_diff_from_b, expected_tensor_diff_from_a],
        [],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_b, expected_dataset_diff_from_a],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    c = local_ds.commit()
    expected_tensor_diff_from_c = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
        "xyz": get_default_tensor_diff(),
        "blue": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_c = get_default_dataset_diff(local_ds.pending_commit_id)
    with local_ds:
        local_ds.rename_tensor("abc", "bcd")
        expected_dataset_diff_from_c["renamed"]["abc"] = "bcd"
        expected_tensor_diff_from_c["bcd"] = expected_tensor_diff_from_c.pop("abc")
        local_ds.rename_tensor("xyz", "abc")
        expected_dataset_diff_from_c["renamed"]["xyz"] = "abc"
        expected_tensor_diff_from_c["abc"] = expected_tensor_diff_from_c.pop("xyz")
        local_ds.delete_tensor("blue")
        expected_dataset_diff_from_c["deleted"].append("blue")
        expected_tensor_diff_from_c.pop("blue")

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [
            expected_tensor_diff_from_c,
            expected_tensor_diff_from_b,
            expected_tensor_diff_from_a,
        ],
        [],
    ]
    expected_dataset_diff = [
        [
            expected_dataset_diff_from_c,
            expected_dataset_diff_from_b,
            expected_dataset_diff_from_a,
        ],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    d = local_ds.commit()
    expected_tensor_diff_from_d = {
        "commit_id": local_ds.pending_commit_id,
        "bcd": get_default_tensor_diff(),
        "abc": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_d = get_default_dataset_diff(local_ds.pending_commit_id)
    with local_ds:
        local_ds.delete_tensor("bcd")
        expected_dataset_diff_from_d["deleted"].append("bcd")
        expected_tensor_diff_from_d.pop("bcd")

        local_ds.rename_tensor("abc", "bcd")
        expected_dataset_diff_from_d["renamed"]["abc"] = "bcd"
        expected_tensor_diff_from_d["bcd"] = expected_tensor_diff_from_d.pop("abc")
        local_ds.bcd.append([4, 5, 6])
        expected_tensor_diff_from_d["bcd"]["data_added"] = [2, 3]

    diff = local_ds.diff(b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [
            expected_tensor_diff_from_d,
            expected_tensor_diff_from_c,
            expected_tensor_diff_from_b,
        ],
        [],
    ]
    expected_dataset_diff = [
        [
            expected_dataset_diff_from_d,
            expected_dataset_diff_from_c,
            expected_dataset_diff_from_b,
        ],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(b)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        b,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    e = local_ds.commit()
    expected_tensor_diff_from_e = {
        "commit_id": local_ds.pending_commit_id,
        "bcd": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_e = get_default_dataset_diff(local_ds.pending_commit_id)
    with local_ds:
        local_ds.rename_tensor("bcd", "abc")
        expected_dataset_diff_from_e["renamed"]["bcd"] = "abc"
        expected_tensor_diff_from_e["abc"] = expected_tensor_diff_from_e.pop("bcd")

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [
            expected_tensor_diff_from_e,
            expected_tensor_diff_from_d,
            expected_tensor_diff_from_c,
            expected_tensor_diff_from_b,
            expected_tensor_diff_from_a,
        ],
        [],
    ]
    expected_dataset_diff = [
        [
            expected_dataset_diff_from_e,
            expected_dataset_diff_from_d,
            expected_dataset_diff_from_c,
            expected_dataset_diff_from_b,
            expected_dataset_diff_from_a,
        ],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()


def test_rename_diff_branch(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("abc")
        local_ds.abc.append([1, 2, 3])

    a = local_ds.commit()
    local_ds.checkout("alt", create=True)
    expected_tensor_diff_from_a_on_alt = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_alt = get_default_dataset_diff(
        local_ds.pending_commit_id
    )

    with local_ds:
        local_ds.rename_tensor("abc", "xyz")
        expected_dataset_diff_from_a_on_alt["renamed"]["abc"] = "xyz"
        expected_tensor_diff_from_a_on_alt[
            "xyz"
        ] = expected_tensor_diff_from_a_on_alt.pop("abc")
        local_ds.xyz.append([4, 5, 6])
        expected_tensor_diff_from_a_on_alt["xyz"]["data_added"] = [1, 2]

    b = local_ds.commit()
    local_ds.checkout("main")
    expected_tensor_diff_from_a_on_main = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_main = get_default_dataset_diff(
        local_ds.pending_commit_id
    )

    with local_ds:
        local_ds.abc.append([2, 3, 4])
        expected_tensor_diff_from_a_on_main["abc"]["data_added"] = [1, 2]
        local_ds.create_tensor("red")
        expected_tensor_diff_from_a_on_main["red"] = get_default_tensor_diff()
        expected_tensor_diff_from_a_on_main["red"]["created"] = True

    c = local_ds.commit()
    local_ds.checkout("alt2", create=True)
    expected_tensor_diff_from_c_on_alt2 = {
        "commit_id": local_ds.pending_commit_id,
        "abc": get_default_tensor_diff(),
        "red": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_c_on_alt2 = get_default_dataset_diff(
        local_ds.pending_commit_id
    )

    with local_ds:
        local_ds.rename_tensor("abc", "efg")
        expected_dataset_diff_from_c_on_alt2["renamed"]["abc"] = "efg"
        expected_tensor_diff_from_c_on_alt2[
            "efg"
        ] = expected_tensor_diff_from_c_on_alt2.pop("abc")
        local_ds.efg.append([5, 6, 7])
        expected_tensor_diff_from_c_on_alt2["efg"]["data_added"] = [2, 3]
        local_ds.efg.info["hello"] = "world"
        expected_tensor_diff_from_c_on_alt2["efg"]["info_updated"] = True
        local_ds.rename_tensor("red", "blue")
        expected_dataset_diff_from_c_on_alt2["renamed"]["red"] = "blue"
        expected_tensor_diff_from_c_on_alt2[
            "blue"
        ] = expected_tensor_diff_from_c_on_alt2.pop("red")

    d = local_ds.commit()
    expected_tensor_diff_from_d_on_alt2 = {
        "commit_id": local_ds.pending_commit_id,
        "blue": get_default_tensor_diff(),
        "efg": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_d_on_alt2 = get_default_dataset_diff(
        local_ds.pending_commit_id
    )

    local_ds.delete_tensor("blue")
    expected_dataset_diff_from_d_on_alt2["deleted"].append("blue")
    expected_tensor_diff_from_d_on_alt2.pop("blue")

    e = local_ds.commit()

    diff = local_ds.diff(b, e, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [expected_tensor_diff_from_a_on_alt],
        [
            expected_tensor_diff_from_d_on_alt2,
            expected_tensor_diff_from_c_on_alt2,
            expected_tensor_diff_from_a_on_main,
        ],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_a_on_alt],
        [
            expected_dataset_diff_from_d_on_alt2,
            expected_dataset_diff_from_c_on_alt2,
            expected_dataset_diff_from_a_on_main,
        ],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(b, e)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        b,
        e,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()


def test_rename_group(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("g1/g2/g3/t1")
        local_ds.create_tensor("g1/g2/t2")
        a = local_ds.commit()
        expected_tensor_diff_from_a = {
            "commit_id": local_ds.pending_commit_id,
            "g1/g2/g3/t1": get_default_tensor_diff(),
            "g1/g2/t2": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_a = get_default_dataset_diff(
            local_ds.pending_commit_id
        )
        local_ds.rename_group("g1/g2", "g1/g4")
        expected_dataset_diff_from_a["renamed"]["g1/g2/g3/t1"] = "g1/g4/g3/t1"
        expected_dataset_diff_from_a["renamed"]["g1/g2/t2"] = "g1/g4/t2"
        expected_tensor_diff_from_a["g1/g4/g3/t1"] = expected_tensor_diff_from_a.pop(
            "g1/g2/g3/t1"
        )
        expected_tensor_diff_from_a["g1/g4/t2"] = expected_tensor_diff_from_a.pop(
            "g1/g2/t2"
        )

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [expected_tensor_diff_from_a]
    expected_dataset_diff = [expected_dataset_diff_from_a]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    local_ds.diff()
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()


def test_diff_linear(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])
        local_ds.create_tensor("pqr")
        local_ds.pqr.extend([4, 5, 6])
    a = local_ds.commit()
    with local_ds:
        expected_tensor_diff_from_a = {
            "commit_id": local_ds.pending_commit_id,
            "xyz": get_default_tensor_diff(),
            "pqr": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_a = get_default_dataset_diff(
            local_ds.pending_commit_id
        )
        local_ds.xyz[0] = 10
        expected_tensor_diff_from_a["xyz"]["data_updated"] = {0}
        local_ds.xyz.info["hello"] = "world"
        expected_tensor_diff_from_a["xyz"]["info_updated"] = True
        local_ds.pqr[2] = 20
        expected_tensor_diff_from_a["pqr"]["data_updated"] = {2}
        local_ds.create_tensor("abc")
        expected_tensor_diff_from_a["abc"] = get_default_tensor_diff()
        expected_tensor_diff_from_a["abc"]["created"] = True
        local_ds.abc.extend([1, 2, 3])
        expected_tensor_diff_from_a["abc"]["data_added"] = [0, 3]

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [expected_tensor_diff_from_a]
    expected_dataset_diff = [expected_dataset_diff_from_a]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    local_ds.diff()
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    b = local_ds.commit()
    expected_tensor_diff_from_b = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
        "pqr": get_default_tensor_diff(),
        "abc": get_default_tensor_diff(),
    }

    expected_dataset_diff_from_b = get_default_dataset_diff(local_ds.pending_commit_id)

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [expected_tensor_diff_from_b]
    expected_dataset_diff = [expected_dataset_diff_from_b]
    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    local_ds.diff()
    target = get_diff_helper(dataset_diff, None, tensor_diff, None)
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [
        [expected_tensor_diff_from_b, expected_tensor_diff_from_a],
        [],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_b, expected_dataset_diff_from_a],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[expected_tensor_diff_from_b], []]
    expected_dataset_diff = [[expected_dataset_diff_from_b], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(b)

    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        b,
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(a, b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[], [expected_tensor_diff_from_a]]
    expected_dataset_diff = [[], [expected_dataset_diff_from_a]]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(a, b)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        a,
        b,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    diff = local_ds.diff(b, a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[expected_tensor_diff_from_a], []]
    expected_dataset_diff = [[expected_dataset_diff_from_a], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    local_ds.diff(b, a)
    target = get_diff_helper(
        dataset_diff[0],
        dataset_diff[1],
        tensor_diff[0],
        tensor_diff[1],
        local_ds.version_state,
        b,
        a,
    )
    captured = capsys.readouterr()
    assert captured.out.strip() == target.strip()

    local_ds.checkout(b)
    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [expected_tensor_diff_from_a]
    expected_dataset_diff = [expected_dataset_diff_from_a]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)


def test_diff_branch(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])

    a = local_ds.commit()
    local_ds.checkout("alt", create=True)
    expected_tensor_diff_from_a_on_alt = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_alt = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    with local_ds:
        local_ds.xyz.extend([4, 5, 6])
        expected_tensor_diff_from_a_on_alt["xyz"]["data_added"] = [3, 6]
        local_ds.create_tensor("pqr")
        expected_tensor_diff_from_a_on_alt["pqr"] = get_default_tensor_diff()
        expected_tensor_diff_from_a_on_alt["pqr"]["created"] = True
        local_ds.pqr.extend([7, 8, 9])
        expected_tensor_diff_from_a_on_alt["pqr"]["data_added"] = [0, 3]
        local_ds.xyz[2] = 6
        local_ds.xyz[3] = 8
        local_ds.pqr[1] = 8
        expected_tensor_diff_from_a_on_alt["xyz"]["data_updated"] = {2}

    b = local_ds.commit()
    local_ds.checkout("main")
    expected_tensor_diff_from_a_on_main = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_main = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    with local_ds:
        local_ds.xyz.extend([0, 0])
        expected_tensor_diff_from_a_on_main["xyz"]["data_added"] = [3, 5]
        local_ds.xyz[2] = 10
        local_ds.xyz[3] = 11
        local_ds.xyz[0] = 11
        expected_tensor_diff_from_a_on_main["xyz"]["data_updated"] = {0, 2}

    diff = local_ds.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [expected_tensor_diff_from_a_on_main]
    expected_dataset_diff = [expected_dataset_diff_from_a_on_main]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    c = local_ds.commit()
    expected_tensor_diff_from_c_on_main = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_c_on_main = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    diff = local_ds.diff(as_dict=True)

    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [expected_tensor_diff_from_c_on_main]
    expected_dataset_diff = [expected_dataset_diff_from_c_on_main]

    compare_tensor_diff(tensor_diff, expected_tensor_diff)
    compare_dataset_diff(dataset_diff, expected_dataset_diff)

    diff = local_ds.diff(a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [
        [expected_tensor_diff_from_c_on_main, expected_tensor_diff_from_a_on_main],
        [],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_c_on_main, expected_dataset_diff_from_a_on_main],
        [],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [
        [expected_tensor_diff_from_c_on_main, expected_tensor_diff_from_a_on_main],
        [expected_tensor_diff_from_a_on_alt],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_c_on_main, expected_dataset_diff_from_a_on_main],
        [expected_dataset_diff_from_a_on_alt],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(c, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[expected_tensor_diff_from_c_on_main], []]
    expected_dataset_diff = [[expected_dataset_diff_from_c_on_main], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(a, b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[], [expected_tensor_diff_from_a_on_alt]]
    expected_dataset_diff = [[], [expected_dataset_diff_from_a_on_alt]]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(b, a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = expected_tensor_diff[::-1]
    expected_dataset_diff = expected_dataset_diff[::-1]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(b, c, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [
        [expected_tensor_diff_from_a_on_alt],
        [expected_tensor_diff_from_a_on_main],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_a_on_alt],
        [expected_dataset_diff_from_a_on_main],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(c, b, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = expected_tensor_diff[::-1]
    expected_dataset_diff = expected_dataset_diff[::-1]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(c, a, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = [[expected_tensor_diff_from_a_on_main], []]
    expected_dataset_diff = [[expected_dataset_diff_from_a_on_main], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(a, c, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    expected_tensor_diff = expected_tensor_diff[::-1]
    expected_dataset_diff = expected_dataset_diff[::-1]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])


def test_complex_diff(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])
    a = local_ds.commit()
    b = local_ds.checkout("alt", create=True)
    expected_tensor_diff_from_a_on_alt = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_alt = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    with local_ds:
        local_ds.xyz.extend([4, 5, 6])
        expected_tensor_diff_from_a_on_alt["xyz"]["data_added"] = [3, 6]
    local_ds.commit()
    expected_tensor_diff_from_b_on_alt = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_b_on_alt = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    c = local_ds.pending_commit_id
    with local_ds:
        local_ds.xyz[4] = 7
        local_ds.xyz[0] = 0
        expected_tensor_diff_from_b_on_alt["xyz"]["data_updated"] = {0, 4}
    local_ds.checkout("main")
    expected_tensor_diff_from_a_on_main = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_a_on_main = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    d = local_ds.pending_commit_id
    with local_ds:
        local_ds.xyz[1] = 10
        expected_tensor_diff_from_a_on_main["xyz"]["data_updated"] = {1}
        local_ds.create_tensor("pqr")
        expected_tensor_diff_from_a_on_main["pqr"] = get_default_tensor_diff()
        expected_tensor_diff_from_a_on_main["pqr"]["created"] = True

    local_ds.commit()
    local_ds.checkout("another", create=True)
    expected_tensor_diff_from_d_on_another = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
        "pqr": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_d_on_another = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    with local_ds:
        local_ds.create_tensor("tuv")
        expected_tensor_diff_from_d_on_another["tuv"] = get_default_tensor_diff()
        expected_tensor_diff_from_d_on_another["tuv"]["created"] = True
        local_ds.tuv.extend([1, 2, 3])
        expected_tensor_diff_from_d_on_another["tuv"]["data_added"] = [0, 3]
        local_ds.pqr.append(5)
        expected_tensor_diff_from_d_on_another["pqr"]["data_added"] = [0, 1]
    local_ds.commit()
    expected_tensor_diff_from_f_on_another = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
        "tuv": get_default_tensor_diff(),
        "pqr": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_f_on_another = get_default_dataset_diff(
        local_ds.pending_commit_id
    )
    g = local_ds.pending_commit_id
    local_ds.checkout("main")
    e = local_ds.pending_commit_id
    expected_tensor_diff_from_d_on_main = {
        "commit_id": local_ds.pending_commit_id,
        "xyz": get_default_tensor_diff(),
        "pqr": get_default_tensor_diff(),
    }
    expected_dataset_diff_from_d_on_main = get_default_dataset_diff(
        local_ds.pending_commit_id
    )

    diff = local_ds.diff(c, g, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [expected_tensor_diff_from_b_on_alt, expected_tensor_diff_from_a_on_alt],
        [
            expected_tensor_diff_from_f_on_another,
            expected_tensor_diff_from_d_on_another,
            expected_tensor_diff_from_a_on_main,
        ],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_b_on_alt, expected_dataset_diff_from_a_on_alt],
        [
            expected_dataset_diff_from_f_on_another,
            expected_dataset_diff_from_d_on_another,
            expected_dataset_diff_from_a_on_main,
        ],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(e, d, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [[expected_tensor_diff_from_d_on_main], []]

    expected_dataset_diff = [[expected_dataset_diff_from_d_on_main], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(e, e, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [[], []]
    expected_dataset_diff = [[], []]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff(c, "main", as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = [
        [expected_tensor_diff_from_b_on_alt, expected_tensor_diff_from_a_on_alt],
        [expected_tensor_diff_from_d_on_main, expected_tensor_diff_from_a_on_main],
    ]
    expected_dataset_diff = [
        [expected_dataset_diff_from_b_on_alt, expected_dataset_diff_from_a_on_alt],
        [expected_dataset_diff_from_d_on_main, expected_dataset_diff_from_a_on_main],
    ]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])

    diff = local_ds.diff("main", c, as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]

    expected_tensor_diff = expected_tensor_diff[::-1]
    expected_dataset_diff = expected_dataset_diff[::-1]

    compare_tensor_diff(tensor_diff[0], expected_tensor_diff[0])
    compare_tensor_diff(tensor_diff[1], expected_tensor_diff[1])
    compare_dataset_diff(dataset_diff[0], expected_dataset_diff[0])
    compare_dataset_diff(dataset_diff[1], expected_dataset_diff[1])


def test_diff_not_exists(local_ds):
    with pytest.raises(KeyError):
        local_ds.diff("12345", "5678")


def test_branches(local_ds_generator):
    local_ds = local_ds_generator()
    assert local_ds.branches == ["main"]
    local_ds.checkout("alt", create=True)
    assert local_ds.branches == ["main", "alt"]
    local_ds.checkout("main")
    assert local_ds.branches == ["main", "alt"]

    local_ds = local_ds_generator()
    assert local_ds.branches == ["main", "alt"]
    local_ds.checkout("alt")
    assert local_ds.branches == ["main", "alt"]
    local_ds.checkout("other", create=True)
    assert local_ds.branches == ["main", "alt", "other"]


def test_commits(local_ds):
    commits = local_ds.commits
    assert len(commits) == 0
    local_ds.commit()
    commits = local_ds.commits
    assert len(commits) == 1
    commit_details_helper(commits, local_ds)
    local_ds.checkout("alt", create=True)
    commits = local_ds.commits
    assert len(commits) == 1
    commit_details_helper(commits, local_ds)
    local_ds.checkout("main")
    commits = local_ds.commits
    assert len(commits) == 1
    commit_details_helper(commits, local_ds)
    local_ds.create_tensor("xyz")
    local_ds.checkout("other", create=True)
    commits = local_ds.commits
    assert len(commits) == 2
    commit_details_helper(commits, local_ds)
    local_ds.commit(allow_empty=True)
    commits = local_ds.commits
    assert len(commits) == 3
    commit_details_helper(commits, local_ds)
    local_ds.checkout("main")
    commits = local_ds.commits
    assert len(commits) == 2
    commit_details_helper(commits, local_ds)
    local_ds.commit(allow_empty=True)
    commits = local_ds.commits
    assert len(commits) == 3
    commit_details_helper(commits, local_ds)


def test_clear(local_ds):
    local_ds.create_tensor("abc")
    local_ds.abc.append([1, 2, 3])
    a = local_ds.commit("first")
    local_ds.abc.clear()
    b = local_ds.commit("second")
    local_ds.abc.append([4, 5, 6, 7])
    c = local_ds.commit("third")
    local_ds.abc.clear()

    assert len(local_ds.abc.numpy()) == 0

    local_ds.checkout(a)

    np.testing.assert_array_equal(local_ds.abc.numpy(), np.array([[1, 2, 3]]))

    local_ds.checkout(b)

    assert len(local_ds.abc.numpy()) == 0

    local_ds.checkout(c)

    np.testing.assert_array_equal(local_ds.abc.numpy(), np.array([[4, 5, 6, 7]]))


def test_custom_commit_hash(local_ds):
    commits = local_ds.commits
    assert len(commits) == 0
    local_ds._commit(hash="abcd")
    assert local_ds.version_state["commit_id"] == "abcd"
    with pytest.raises(CommitError):
        local_ds._commit(hash="abcd")
    with pytest.raises(CommitError):
        local_ds._checkout("xyz", create=True, hash="abcd")
    local_ds._checkout("xyz", create=True, hash="efgh")
    assert local_ds.version_state["commit_id"] == "efgh"
    assert set(local_ds.version_state["branch_commit_map"].keys()) == set(
        ("main", "xyz")
    )
    assert local_ds.version_state["branch_commit_map"]["xyz"] == "efgh"


def test_read_only_checkout(local_ds):
    with local_ds:
        local_ds.create_tensor("x")
        local_ds.x.append([1, 2, 3])
        local_ds.checkout("branch", create=True)
        local_ds.checkout("main")
    assert local_ds.storage.autoflush == True
    local_ds.read_only = True
    local_ds.checkout("main")


def test_modified_samples(memory_ds):
    with memory_ds:
        memory_ds.create_tensor("image")
        memory_ds.image.extend(np.array(list(range(5))))

        img, indexes = memory_ds.image.modified_samples(return_indexes=True)
        assert indexes == list(range(5))
        assert len(img) == 5
        for i in range(5):
            np.testing.assert_array_equal(img[i].numpy(), i)
        first_commit = memory_ds.commit()

        img, indexes = memory_ds.image.modified_samples(return_indexes=True)
        assert indexes == []
        assert len(img) == 0

        memory_ds.image.extend(np.array(list(range(5, 8))))
        img, indexes = memory_ds.image.modified_samples(return_indexes=True)
        assert indexes == list(range(5, 8))
        assert len(img) == 3
        for i in range(3):
            np.testing.assert_array_equal(img[i].numpy(), i + 5)

        memory_ds.image[2] = -1
        img, indexes = memory_ds.image.modified_samples(return_indexes=True)
        assert indexes == [2, 5, 6, 7]
        assert len(img) == 4
        np.testing.assert_array_equal(img[0].numpy(), -1)
        for i in range(3):
            np.testing.assert_array_equal(img[i + 1].numpy(), i + 5)

        memory_ds.image[4] = 8
        img, indexes = memory_ds.image.modified_samples(return_indexes=True)
        assert indexes == [2, 4, 5, 6, 7]
        assert len(img) == 5
        np.testing.assert_array_equal(img[0].numpy(), -1)
        np.testing.assert_array_equal(img[1].numpy(), 8)
        for i in range(3):
            np.testing.assert_array_equal(img[i + 2].numpy(), i + 5)

        second_commit = memory_ds.commit()
        img = memory_ds.image.modified_samples()
        assert len(img) == 0

        img, indexes = memory_ds.image.modified_samples(
            first_commit, return_indexes=True
        )
        assert indexes == [2, 4, 5, 6, 7]
        assert len(img) == 5
        np.testing.assert_array_equal(img[0].numpy(), -1)
        np.testing.assert_array_equal(img[1].numpy(), 8)
        for i in range(3):
            np.testing.assert_array_equal(img[i + 2].numpy(), i + 5)

        memory_ds.checkout(first_commit)
        memory_ds.checkout("alt", create=True)
        alt_commit = memory_ds.commit(allow_empty=True)

        memory_ds.checkout("main")

        with pytest.raises(TensorModifiedError):
            memory_ds.image.modified_samples(alt_commit)


def test_reset(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")
        for i in range(10):
            ds.abc.append(i)
            ds.xyz.append(np.ones((100, 100, 3)) * i)
        ds.info.hello = "world"
        ds.abc.info.hello = "world"
        assert list(local_ds.tensors) == ["abc", "xyz"]
        assert local_ds.info.hello == "world"
        for i in range(10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), i)
            np.testing.assert_array_equal(ds.xyz[i].numpy(), np.ones((100, 100, 3)) * i)

        ds.reset()
        assert not list(local_ds.tensors)
        assert "hello" not in local_ds.info

        ds.create_tensor("abc")
        for i in range(10):
            ds.abc.append(i)
        ds.info.hello = "world"
        ds.abc.info.hello = "world"
        assert list(local_ds.tensors) == ["abc"]
        assert local_ds.info.hello == "world"
        assert local_ds.abc.info.hello == "world"
        assert len(ds) == 10

        for i in range(10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), i)

        ds.commit()
        ds.reset()

        assert list(local_ds.tensors) == ["abc"]
        assert local_ds.info.hello == "world"
        assert local_ds.abc.info.hello == "world"
        for i in range(10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), i)

        for i in range(10):
            ds.abc[i] = 0
        for i in range(10, 15):
            ds.abc.append(i)

        ds.info.hello = "new world"
        ds.abc.info.hello1 = "world1"

        assert ds.info.hello == "new world"
        assert local_ds.abc.info.hello == "world"
        assert ds.abc.info.hello1 == "world1"
        assert len(ds) == 15

        for i in range(10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), 0)

        for i in range(10, 15):
            np.testing.assert_array_equal(ds.abc[i].numpy(), i)

        ds.reset()

        assert list(local_ds.tensors) == ["abc"]
        assert local_ds.info.hello == "world"
        assert local_ds.abc.info.hello == "world"
        assert "hello1" not in ds.abc.info
        assert len(ds) == 10

        for i in range(10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), i)


def test_reset_create_delete_tensors(local_ds):
    with local_ds as ds:
        local_ds.create_tensor("one")
        local_ds.create_tensor("two")
        assert set(ds.tensors.keys()) == {"one", "two"}
        ds.commit()
        assert set(ds.tensors.keys()) == {"one", "two"}
        ds.create_tensor("three")
        ds.delete_tensor("two")
        assert set(ds.tensors.keys()) == {"one", "three"}
        ds.reset()
        assert set(ds.tensors.keys()) == {"one", "two"}


@pytest.mark.parametrize(
    "ds_generator",
    [
        "local_ds_generator",
        pytest.param("s3_ds_generator", marks=pytest.mark.slow),
        pytest.param("gcs_ds_generator", marks=pytest.mark.slow),
        pytest.param("hub_cloud_ds_generator", marks=pytest.mark.slow),
    ],
    indirect=True,
)
def test_reset_bug(ds_generator):
    ds = ds_generator()
    ds.create_tensor("abc")
    ds.abc.append([1, 2, 3])
    assert len(ds.abc) == 1
    a = ds.commit()

    ds = ds_generator()
    ds.abc.append([3, 4, 5])
    assert len(ds.abc) == 2
    ds.reset()
    assert len(ds.abc) == 1


def test_reset_delete_group(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc/x")
        ds.commit()
        ds.delete_group("abc")
        assert ds.has_head_changes


def test_load_to_version(local_path):
    with deeplake.empty(local_path, overwrite=True) as ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        main_1 = ds.commit()
        ds.create_tensor("xyz")
        main_2 = ds.commit()

        ds.checkout("alt", create=True)
        ds.abc.append(2)
        alt_1 = ds.commit()
        ds.xyz.append(1)
        ds.xyz.append(2)
        alt_2 = ds.commit()

    ds = deeplake.load(f"{local_path}@{main_1}")
    set(ds.tensors.keys()) == {"abc"}
    np.testing.assert_array_equal(ds.abc.numpy(), [[1]])

    for address in ("main", main_2):
        ds = deeplake.load(f"{local_path}@{address}")
        set(ds.tensors.keys()) == {"abc", "xyz"}
        np.testing.assert_array_equal(ds.abc.numpy(), [[1]])
        np.testing.assert_array_equal(ds.xyz.numpy(), [])

    ds = deeplake.load(f"{local_path}@{alt_1}")
    set(ds.tensors.keys()) == {"abc", "xyz"}
    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2]])
    np.testing.assert_array_equal(ds.xyz.numpy(), [])

    for address in ("alt", alt_2):
        ds = deeplake.load(f"{local_path}@{address}")
        set(ds.tensors.keys()) == {"abc", "xyz"}
        np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2]])
        np.testing.assert_array_equal(ds.xyz.numpy(), [[1], [2]])


def test_version_in_path(local_path):
    with pytest.raises(ValueError):
        deeplake.empty(f"{local_path}@main")

    with pytest.raises(ValueError):
        deeplake.delete(f"{local_path}@main")

    with pytest.raises(ValueError):
        deeplake.dataset(f"{local_path}@main")

    with pytest.raises(ValueError):
        deeplake.exists(f"{local_path}@main")


@pytest.mark.slow
def test_branch_delete(local_ds_generator):
    local_ds = local_ds_generator()
    local_ds.create_tensor("test")

    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("main")
    assert "Cannot delete the currently checked out branch: main" in str(e.value)

    # Add commits to main
    local_ds.test.append("main 1")
    local_ds.test.append("main 2")
    local_ds.commit("first main commit")
    local_ds.test.append("main 3")
    local_ds.commit("second main commit")
    local_ds.test.append("main 4")
    local_ds.commit("third main commit")

    assert len(local_ds.branches) == 1
    original_version_count = len(glob.glob(local_ds.path + "/versions/*"))

    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("invalid_branch")
    assert "Branch invalid_branch does not exist" in str(e.value)

    # Create a simple branch to delete with commits
    local_ds.checkout("alt1", create=True)
    assert len(local_ds.branches) == 2

    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("alt1")
    assert "Cannot delete the currently checked out branch: alt1" in str(e.value)

    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("main")
    assert "Cannot delete the main branch" in str(e.value)

    # Simple branch can be deleted and it's correctly cleaned out
    local_ds.checkout("main")
    local_ds.delete_branch("alt1")
    assert len(local_ds.branches) == 1
    with open(local_ds.path + "/version_control_info.json", "r") as f:
        assert '"alt1"' not in f.read()
    assert original_version_count == len(glob.glob(local_ds.path + "/versions/*"))

    # Branches with children cannot be deleted until children are deleted
    local_ds.checkout("alt1", create=True)
    local_ds.test.append("alt1 4")
    local_ds.commit("first alt1 commit")

    local_ds.checkout("alt1_sub1", create=True)
    local_ds.test.append("alt1_sub1 5")
    local_ds.commit("first alt1_sub1 commit")

    local_ds.checkout("alt1")
    local_ds.checkout("alt1_sub2", create=True)
    local_ds.test.append("alt1_sub2 5")
    local_ds.commit("first alt1_sub2 commit")

    local_ds.checkout("main")
    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("alt1")
    assert "Cannot delete branch alt1 because it has sub-branches" in str(e.value)

    assert len(local_ds.branches) == 4

    local_ds.delete_branch("alt1_sub1")
    assert len(local_ds.branches) == 3
    with open(local_ds.path + "/version_control_info.json", "r") as f:
        content = f.read()
        assert '"alt1_sub1"' not in content
        assert '"alt1_sub2"' in content
        assert '"alt1"' in content

    local_ds.delete_branch("alt1_sub2")
    assert len(local_ds.branches) == 2
    with open(local_ds.path + "/version_control_info.json", "r") as f:
        content = f.read()
        assert '"alt1_sub2"' not in content
        assert '"alt1"' in content

    local_ds.delete_branch("alt1")
    assert len(local_ds.branches) == 1
    with open(local_ds.path + "/version_control_info.json", "r") as f:
        assert '"alt1"' not in f.read()

    assert original_version_count == len(glob.glob(local_ds.path + "/versions/*"))

    # Branches that have been merged into other branches cannot be merged
    local_ds.checkout("alt1", create=True)
    local_ds.test.append("alt1 4")
    local_ds.commit("first alt1 commit")

    local_ds.checkout("main")
    local_ds.merge("alt1")

    with pytest.raises(VersionControlError) as e:
        local_ds.delete_branch("alt1")
    assert (
        "Cannot delete branch alt1 because it has been previously merged into main"
        in str(e.value)
    )

    local_ds.checkout("alt1")
    local_ds.checkout("alt1_sub1", create=True)
    local_ds.test.append("alt1_sub1 5")

    local_ds.checkout("main")

    local_ds.delete_branch("alt1_sub1")
    assert len(local_ds.branches) == 2


def test_squash_main_has_branch(local_ds_generator):
    local_ds = local_ds_generator()
    local_ds.create_tensor("test")
    with local_ds:
        local_ds.test.append("main 1")
        local_ds.commit("first main commit")
    local_ds.checkout("alt", create=True)

    with pytest.raises(VersionControlError) as e:
        local_ds._squash_main()
    assert "Cannot squash commits if there are multiple branches" in str(e.value)


def test_squash_main_has_view(local_ds_generator):
    local_ds = local_ds_generator()
    local_ds.create_tensor("test")
    with local_ds:
        local_ds.test.append("main 1")
        local_ds.commit("first main commit")
    query = local_ds.filter("test == 'a'")
    query.save_view("test_view")

    with pytest.raises(VersionControlError) as e:
        local_ds._squash_main()
    assert "Cannot squash commits if there are views present" in str(e.value)


def test_squash_main(local_ds_generator):
    local_ds = local_ds_generator()
    local_ds.create_tensor("test")

    with local_ds:
        # Add commits to main
        local_ds.test.append("main 1")
        local_ds.test.append("main 2")
        local_ds.commit("first main commit")
        local_ds.test.append("main 3")
        local_ds.commit("second main commit")
        local_ds.test.append("main 4")
        local_ds.commit("third main commit")
        local_ds.test.append("main uncommitted")

    assert len(local_ds.branches) == 1
    assert len(glob.glob(local_ds.path + "/versions/*")) > 0
    assert len(local_ds.test) == 5
    assert [i.data()["value"] for i in local_ds.test] == [
        "main 1",
        "main 2",
        "main 3",
        "main 4",
        "main uncommitted",
    ]
    assert [i["message"] for i in local_ds.commits] == [
        "third main commit",
        "second main commit",
        "first main commit",
    ]

    local_ds._squash_main()

    assert len(local_ds.branches) == 1
    assert len(glob.glob(local_ds.path + "/versions/*")) == 1
    assert [commit["message"] for commit in local_ds.commits] == ["Squashed commits"]
    assert local_ds.pending_commit_id != FIRST_COMMIT_ID

    with open(local_ds.path + "/version_control_info.json", "r") as f:
        data = json.load(f)
        assert len(data["commits"]) == 1
        assert data["commits"][FIRST_COMMIT_ID]["commit_message"] == None
        assert data["commits"][FIRST_COMMIT_ID]["commit_time"] == None
        assert data["commits"][FIRST_COMMIT_ID]["commit_user_name"] == None
        assert len(data["commits"][FIRST_COMMIT_ID]["children"]) == 0
        assert data["commits"][FIRST_COMMIT_ID]["parent"] == None

    assert [i.data()["value"] for i in local_ds.test] == [
        "main 1",
        "main 2",
        "main 3",
        "main 4",
        "main uncommitted",
    ]
    assert [i["message"] for i in local_ds.commits] == ["Squashed commits"]
    assert local_ds.pending_commit_id != FIRST_COMMIT_ID
