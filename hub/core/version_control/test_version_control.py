import hub
import pytest
import numpy as np
from hub.util.diff import (
    get_all_changes_string,
    get_lowest_common_ancestor,
    sanitize_commit,
)
from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import (
    CheckoutError,
    CommitError,
    ReadOnlyModeError,
    InfoError,
    TensorModifiedError,
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
    dsv.abc[0] = 5
    h = local_ds.commit("it is 5")
    i = local_ds.checkout(e)
    local_ds.abc[0] = 6
    tsv = local_ds.abc[0:5]
    tsv[0] = 6
    j = local_ds.commit("it is 6")
    local_ds.log()
    local_ds.checkout(a)
    assert dsv.abc[0].numpy() == 1
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
    ds = hub.Dataset(storage=local_ds.storage, read_only=True, verbose=False)
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

    local_ds = hub.dataset(path)
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

    a = local_ds.commit("added key, value")
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "value"

    local_ds.info.key2 = "value2"
    assert len(local_ds.info) == 2
    assert local_ds.info.key == "value"
    assert local_ds.info.key2 == "value2"

    b = local_ds.commit("added key2, value2")
    assert len(local_ds.info) == 2
    assert local_ds.info.key == "value"
    assert local_ds.info.key2 == "value2"

    local_ds.checkout(a)
    assert local_ds.info.key == "value"

    local_ds.checkout("alt", create=True)
    local_ds.info.key = "notvalue"
    assert len(local_ds.info) == 1
    assert local_ds.info.key == "notvalue"
    c = local_ds.commit("changed key to notvalue")

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
        b = local_ds.commit("second")
        local_ds.checkout(a)
        assert local_ds.abc[0].numpy() == 1
        local_ds.checkout(b)
        assert local_ds.tensors == {}

        local_ds.create_tensor("x/y/z")
        local_ds["x/y/z"].append(1)
        c = local_ds.commit("third")
        local_ds["x"].delete_tensor("y/z")
        d = local_ds.commit("fourth")
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


def test_diff_linear(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])
        local_ds.create_tensor("pqr")
        local_ds.pqr.extend([4, 5, 6])
    a = local_ds.commit()
    with local_ds:
        local_ds.xyz[0] = 10
        local_ds.xyz.info["hello"] = "world"
        local_ds.pqr[2] = 20
        local_ds.create_tensor("abc")
        local_ds.abc.extend([1, 2, 3])

    local_ds.diff()
    changes_b_from_a = {
        "xyz": {
            "data_added": [3, 3],
            "data_updated": {0},
            "created": False,
            "info_updated": True,
            "data_transformed_in_place": False,
        },
        "pqr": {
            "data_added": [3, 3],
            "data_updated": {2},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "abc": {
            "data_added": [0, 3],
            "data_updated": set(),
            "created": True,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    message0 = NO_COMMIT_PASSED_DIFF
    message1 = "Diff in HEAD relative to the previous commit:\n"
    # ds_changes_1, ds_changes_2, tensor_changes_1, tensor_changes_2, msg_0, msg_1, msg_2
    target = (
        get_all_changes_string({}, {}, changes_b_from_a, None, message0, message1, None)
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(as_dict=True)
    assert diff == changes_b_from_a

    b = local_ds.commit()
    local_ds.diff()
    changes_empty = {}
    target = (
        get_all_changes_string({}, {}, changes_empty, None, message0, message1, None)
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(as_dict=True)
    assert diff == changes_empty

    local_ds.diff(a)
    lca_id = get_lca_id_helper(local_ds.version_state, a)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message1 = "Diff in HEAD:\n"
    message2 = f"Diff in {a} (target id):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_b_from_a, changes_empty, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_b_from_a
    assert diff[1] == changes_empty

    local_ds.diff(b)
    lca_id = get_lca_id_helper(local_ds.version_state, b)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message2 = f"Diff in {b} (target id):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_empty, changes_empty, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(b, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_empty
    assert diff[1] == changes_empty

    local_ds.diff(a, b)
    lca_id = get_lca_id_helper(local_ds.version_state, a, b)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {a} (target id 1):\n"
    message2 = f"Diff in {b} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_empty, changes_b_from_a, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, b, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_empty
    assert diff[1] == changes_b_from_a

    local_ds.diff(b, a)
    lca_id = get_lca_id_helper(local_ds.version_state, b, a)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {b} (target id 1):\n"
    message2 = f"Diff in {a} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_b_from_a, changes_empty, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(b, a, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_b_from_a
    assert diff[1] == changes_empty

    local_ds.checkout(b)
    local_ds.diff()
    message0 = NO_COMMIT_PASSED_DIFF
    message1 = f"Diff in {b} (current commit) relative to the previous commit:\n"
    target = (
        get_all_changes_string({}, {}, changes_b_from_a, None, message0, message1, None)
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(as_dict=True)
    assert diff == changes_b_from_a

    local_ds.diff(a)
    lca_id = get_lca_id_helper(local_ds.version_state, a)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {b} (current commit):\n"
    message2 = f"Diff in {a} (target id):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_b_from_a, changes_empty, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, as_dict=True)
    assert isinstance(diff, tuple)


def test_diff_branch(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])

    a = local_ds.commit()
    local_ds.checkout("alt", create=True)
    with local_ds:
        local_ds.xyz.extend([4, 5, 6])
        local_ds.create_tensor("pqr")
        local_ds.pqr.extend([7, 8, 9])
        local_ds.xyz[2] = 6
        local_ds.xyz[3] = 8
        local_ds.pqr[1] = 8

    b = local_ds.commit()
    local_ds.checkout("main")
    with local_ds:
        local_ds.xyz.extend([0, 0])
        local_ds.xyz[2] = 10
        local_ds.xyz[3] = 11
        local_ds.xyz[0] = 11

    local_ds.diff()
    changes_b_from_branch_off = {
        "xyz": {
            "data_added": [3, 6],
            "data_updated": {2},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "pqr": {
            "data_added": [0, 3],
            "data_updated": set(),
            "created": True,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    changes_main_from_branch_off = {
        "xyz": {
            "data_added": [3, 5],
            "data_updated": {0, 2},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    message0 = NO_COMMIT_PASSED_DIFF
    message1 = "Diff in HEAD relative to the previous commit:\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_main_from_branch_off, None, message0, message1, None
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(as_dict=True)
    assert diff == changes_main_from_branch_off

    c = local_ds.commit()

    local_ds.diff()
    empty_changes = {}
    target = (
        get_all_changes_string({}, {}, empty_changes, None, message0, message1, None)
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(as_dict=True)
    assert diff == empty_changes

    local_ds.diff(a)
    lca_id = get_lca_id_helper(local_ds.version_state, a)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message1 = "Diff in HEAD:\n"
    message2 = f"Diff in {a} (target id):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_main_from_branch_off,
            empty_changes,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_main_from_branch_off
    assert diff[1] == empty_changes

    local_ds.diff(b)
    lca_id = get_lca_id_helper(local_ds.version_state, b)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message2 = f"Diff in {b} (target id):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_main_from_branch_off,
            changes_b_from_branch_off,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(b, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_main_from_branch_off
    assert diff[1] == changes_b_from_branch_off

    local_ds.diff(c)
    lca_id = get_lca_id_helper(local_ds.version_state, c)
    message0 = ONE_COMMIT_PASSED_DIFF % lca_id
    message2 = f"Diff in {c} (target id):\n"
    target = (
        get_all_changes_string(
            {}, {}, empty_changes, empty_changes, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(c, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == empty_changes
    assert diff[1] == empty_changes

    local_ds.diff(a, b)
    lca_id = get_lca_id_helper(local_ds.version_state, a)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {a} (target id 1):\n"
    message2 = f"Diff in {b} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            empty_changes,
            changes_b_from_branch_off,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, b, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == empty_changes
    assert diff[1] == changes_b_from_branch_off

    local_ds.diff(b, a)
    lca_id = get_lca_id_helper(local_ds.version_state, a, b)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {b} (target id 1):\n"
    message2 = f"Diff in {a} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_b_from_branch_off,
            empty_changes,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(b, a, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_b_from_branch_off
    assert diff[1] == empty_changes

    local_ds.diff(b, c)
    lca_id = get_lca_id_helper(local_ds.version_state, b, c)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {b} (target id 1):\n"
    message2 = f"Diff in {c} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_b_from_branch_off,
            changes_main_from_branch_off,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target

    local_ds.diff(c, b)
    lca_id = get_lca_id_helper(local_ds.version_state, c, b)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {c} (target id 1):\n"
    message2 = f"Diff in {b} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_main_from_branch_off,
            changes_b_from_branch_off,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(c, b, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_main_from_branch_off
    assert diff[1] == changes_b_from_branch_off

    local_ds.diff(c, a)
    lca_id = get_lca_id_helper(local_ds.version_state, c, a)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {c} (target id 1):\n"
    message2 = f"Diff in {a} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            changes_main_from_branch_off,
            empty_changes,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(c, a, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_main_from_branch_off
    assert diff[1] == empty_changes

    local_ds.diff(a, c)
    lca_id = get_lca_id_helper(local_ds.version_state, a, c)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {a} (target id 1):\n"
    message2 = f"Diff in {c} (target id 2):\n"
    target = (
        get_all_changes_string(
            {},
            {},
            empty_changes,
            changes_main_from_branch_off,
            message0,
            message1,
            message2,
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(a, c, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == empty_changes
    assert diff[1] == changes_main_from_branch_off


def test_complex_diff(local_ds, capsys):
    with local_ds:
        local_ds.create_tensor("xyz")
        local_ds.xyz.extend([1, 2, 3])
    a = local_ds.commit()
    b = local_ds.checkout("alt", create=True)
    with local_ds:
        local_ds.xyz.extend([4, 5, 6])
    local_ds.commit()
    c = local_ds.pending_commit_id
    with local_ds:
        local_ds.xyz[4] = 7
        local_ds.xyz[0] = 0
    local_ds.checkout("main")
    d = local_ds.pending_commit_id
    with local_ds:
        local_ds.xyz[1] = 10
        local_ds.create_tensor("pqr")
    local_ds.commit()
    f = local_ds.checkout("another", create=True)
    with local_ds:
        local_ds.create_tensor("tuv")
        local_ds.tuv.extend([1, 2, 3])
        local_ds.pqr.append(5)
    local_ds.commit()
    g = local_ds.pending_commit_id
    e = local_ds.checkout("main")

    # x is LCA of a and g
    changes_c_from_x = {
        "xyz": {
            "data_added": [3, 6],
            "data_updated": {0},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    changes_g_from_x = {
        "pqr": {
            "data_added": [0, 1],
            "data_updated": set(),
            "created": True,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "tuv": {
            "data_added": [0, 3],
            "data_updated": set(),
            "created": True,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "xyz": {
            "data_added": [3, 3],
            "data_updated": {1},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    empty_changes = {}

    local_ds.diff(c, g)
    lca_id = get_lca_id_helper(local_ds.version_state, c, g)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {c} (target id 1):\n"
    message2 = f"Diff in {g} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_c_from_x, changes_g_from_x, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(c, g, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_c_from_x
    assert diff[1] == changes_g_from_x

    local_ds.diff(e, d)
    lca_id = get_lca_id_helper(local_ds.version_state, e, d)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {e} (target id 1):\n"
    message2 = f"Diff in {d} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, empty_changes, empty_changes, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(e, d, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == empty_changes
    assert diff[1] == empty_changes

    local_ds.diff(e, e)
    lca_id = get_lca_id_helper(local_ds.version_state, e, e)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {e} (target id 1):\n"
    message2 = f"Diff in {e} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, empty_changes, empty_changes, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(e, e, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == empty_changes
    assert diff[1] == empty_changes

    changes_main_from_x = {
        "xyz": {
            "data_added": [3, 3],
            "data_updated": {1},
            "created": False,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "pqr": {
            "data_added": [0, 0],
            "data_updated": set(),
            "created": True,
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }

    local_ds.diff(c, "main")
    lca_id = get_lca_id_helper(local_ds.version_state, c, "main")
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = f"Diff in {c} (target id 1):\n"
    message2 = "Diff in main (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_c_from_x, changes_main_from_x, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff(c, "main", as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_c_from_x
    assert diff[1] == changes_main_from_x

    local_ds.diff("main", c)
    lca_id = get_lca_id_helper(local_ds.version_state, "main", c)
    message0 = TWO_COMMIT_PASSED_DIFF % lca_id
    message1 = "Diff in main (target id 1):\n"
    message2 = f"Diff in {c} (target id 2):\n"
    target = (
        get_all_changes_string(
            {}, {}, changes_main_from_x, changes_c_from_x, message0, message1, message2
        )
        + "\n"
    )
    captured = capsys.readouterr()
    assert captured.out == target
    diff = local_ds.diff("main", c, as_dict=True)
    assert isinstance(diff, tuple)
    assert diff[0] == changes_main_from_x
    assert diff[1] == changes_c_from_x


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
    local_ds.commit()
    commits = local_ds.commits
    assert len(commits) == 3
    commit_details_helper(commits, local_ds)
    local_ds.checkout("main")
    commits = local_ds.commits
    assert len(commits) == 2
    commit_details_helper(commits, local_ds)
    local_ds.commit()
    commits = local_ds.commits
    assert len(commits) == 3
    commit_details_helper(commits, local_ds)


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
        alt_commit = memory_ds.commit()

        memory_ds.checkout("main")

        with pytest.raises(TensorModifiedError):
            memory_ds.image.modified_samples(alt_commit)
