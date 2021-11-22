import hub
import pytest
import numpy as np
from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import CheckoutError, ReadOnlyModeError


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
    d = local_ds.checkout("other", True)
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
    local_ds.info[5] = 5
    assert local_ds.branch != "main"

    local_ds.checkout(first)
    assert local_ds.branch == "main"
    local_ds.info.update(list=[1, 2, "apple"])
    assert local_ds.branch != "main"


def test_auto_commit(local_ds):
    initial_commit_id = local_ds.commit_id
    local_ds.checkout("pqr", create=True)
    local_ds.checkout("main")
    # no auto commit as there was no data
    assert local_ds.commit_id == initial_commit_id
    local_ds.create_tensor("abc")
    local_ds.abc.append(1)
    # auto commit as there was data
    local_ds.checkout("xyz", create=True)
    local_ds.checkout("main")

    assert local_ds.commit_id != initial_commit_id

    with local_ds:
        local_ds.abc.append(1)

    current_commit_id = local_ds.commit_id

    # auto commit as there was data
    local_ds.checkout("tuv", create=True)
    local_ds.checkout("main")

    assert local_ds.commit_id != current_commit_id
