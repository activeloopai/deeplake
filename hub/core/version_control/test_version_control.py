import hub
import pytest
import numpy as np
from hub.tests.dataset_fixtures import enabled_datasets
from hub.util.exceptions import CheckoutError, ReadOnlyModeError


@enabled_datasets
def test_commit(ds):
    with ds:
        ds.create_tensor("abc")
        ds.abc.append(1)
        a = ds.commit("first")
        ds.abc[0] = 2
        b = ds.commit("second")
        ds.abc[0] = 3
        c = ds.commit("third")
        assert ds.abc[0].numpy() == 3
        ds.checkout(a)
        assert ds.abc[0].numpy() == 1
        ds.checkout(b)
        assert ds.abc[0].numpy() == 2
        ds.checkout(c)
        assert ds.abc[0].numpy() == 3
        with pytest.raises(CheckoutError):
            ds.checkout("main", create=True)
        with pytest.raises(CheckoutError):
            ds.checkout(a, create=True)


@enabled_datasets
def test_commit_checkout(ds):
    with ds:
        ds.create_tensor("img")
        ds.img.extend(np.ones((10, 100, 100, 3)))
        first_commit_id = ds.commit("stored all ones")

        for i in range(5):
            ds.img[i] *= 2
        second_commit_id = ds.commit("multiplied value of some images by 2")

        for i in range(5):
            assert (ds.img[i].numpy() == 2 * np.ones((100, 100, 3))).all()
        ds.checkout(first_commit_id)  # now all images are ones again

        for i in range(10):
            assert (ds.img[i].numpy() == np.ones((100, 100, 3))).all()

        ds.checkout("alternate", create=True)

        for i in range(5):
            ds.img[i] *= 3
        ds.commit("multiplied value of some images by 3")

        for i in range(5):
            assert (ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()

        ds.checkout(second_commit_id)  # first 5 images are 2s, rest are 1s now

        # we are not at the head of master but rather at the last commit, so we automatically get checkouted out to a new branch here
        for i in range(5, 10):
            ds.img[i] *= 2
        ds.commit("multiplied value of remaining images by 2")

        for i in range(10):
            assert (ds.img[i].numpy() == 2 * np.ones((100, 100, 3))).all()

        ds.checkout("alternate")

        for i in range(5, 10):
            ds.img[i] *= 3

        for i in range(10):
            assert (ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()
        ds.commit("multiplied value of remaining images by 3")
        for i in range(10):
            assert (ds.img[i].numpy() == 3 * np.ones((100, 100, 3))).all()


@enabled_datasets
def test_commit_checkout_2(ds):
    ds.create_tensor("abc")
    ds.create_tensor("img")
    for i in range(10):
        ds.img.append(i * np.ones((100, 100, 3)))
    a = ds.commit("first")

    ds.img[7] *= 2

    assert (ds.img[6].numpy() == 6 * np.ones((100, 100, 3))).all()
    assert (ds.img[7].numpy() == 2 * 7 * np.ones((100, 100, 3))).all()
    assert (ds.img[8].numpy() == 8 * np.ones((100, 100, 3))).all()
    assert (ds.img[9].numpy() == 9 * np.ones((100, 100, 3))).all()

    assert (ds.img[2].numpy() == 2 * np.ones((100, 100, 3))).all()

    b = ds.commit("second")

    # going back to first commit
    ds.checkout(a)

    assert (ds.img[7].numpy() == 7 * np.ones((100, 100, 3))).all()

    ds.checkout("another", create=True)

    ds.img[7] *= 3

    # and not 6 * 7 as it would have been, had we checked out from b
    assert (ds.img[7].numpy() == 3 * 7 * np.ones((100, 100, 3))).all()

    ds.commit("first2")

    ds.checkout("main")
    assert (ds.img[7].numpy() == 2 * 7 * np.ones((100, 100, 3))).all()
    ds.log()


@enabled_datasets
def test_auto_checkout_bug(ds):
    ds.create_tensor("abc")
    ds.abc.extend([1, 2, 3, 4, 5])
    a = ds.commit("it is 1")
    ds.abc[0] = 2
    b = ds.commit("it is 2")
    c = ds.checkout(a)
    d = ds.checkout("other", True)
    ds.abc[0] = 3
    e = ds.commit("it is 3")
    ds.checkout(b)
    ds.abc[0] = 4
    f = ds.commit("it is 4")
    g = ds.checkout(a)
    ds.abc[0] = 5
    dsv = ds[0:3]
    dsv.abc[0] = 5
    h = ds.commit("it is 5")
    i = ds.checkout(e)
    ds.abc[0] = 6
    tsv = ds.abc[0:5]
    tsv[0] = 6
    j = ds.commit("it is 6")
    ds.log()
    ds.checkout(a)
    assert dsv.abc[0].numpy() == 1
    assert ds.abc[0].numpy() == 1
    ds.checkout(b)
    assert ds.abc[0].numpy() == 2
    ds.checkout(c)
    assert ds.abc[0].numpy() == 1
    ds.checkout(d)
    assert ds.abc[0].numpy() == 3
    ds.checkout(e)
    assert ds.abc[0].numpy() == 3
    ds.checkout(f)
    assert ds.abc[0].numpy() == 4
    ds.checkout(g)
    assert ds.abc[0].numpy() == 1
    ds.checkout(h)
    assert ds.abc[0].numpy() == 5
    ds.checkout(i)
    assert ds.abc[0].numpy() == 3
    ds.checkout(j)
    assert ds.abc[0].numpy() == 6
    ds.checkout("main")
    assert ds.abc[0].numpy() == 2
    ds.abc[0] = 7
    ds.checkout("copy", True)
    assert ds.abc[0].numpy() == 7
    ds.checkout("other")
    assert ds.abc[0].numpy() == 3


# @enabled_datasets
# def test_read_mode(ds):
#     ds.create_tensor("abc")
#     ds.checkout("second", create=True)
#     with pytest.raises(ReadOnlyModeError):
#         ds.commit("first")
#     with pytest.raises(ReadOnlyModeError):
#         ds.checkout("third", create=True)
#     with pytest.raises(ReadOnlyModeError):
#         ds.abc.append(10)


@enabled_datasets
def test_checkout_address_not_found(ds):
    with pytest.raises(CheckoutError):
        ds.checkout("second")


@enabled_datasets
def test_dynamic(ds):
    ds.create_tensor("img")
    for i in range(10):
        ds.img.append(i * np.ones((100, 100, 3)))

    a = ds.commit("first")
    for i in range(10):
        ds.img[i] = 2 * i * np.ones((150, 150, 3))
    ds.checkout(a)

    for i in range(10):
        assert (ds.img[i].numpy() == i * np.ones((100, 100, 3))).all()

    ds.checkout("main")

    for i in range(10):
        assert (ds.img[i].numpy() == 2 * i * np.ones((150, 150, 3))).all()


@enabled_datasets
def test_different_lengths(ds):
    with ds:
        ds.create_tensor("img")
        ds.create_tensor("abc")
        ds.img.extend(np.ones((5, 50, 50)))
        ds.abc.extend(np.ones((2, 10, 10)))
        first = ds.commit("stored 5 images, 2 abc")
        ds.img.extend(np.ones((3, 50, 50)))
        second = ds.commit("stored 3 more images")
        assert len(ds.tensors) == 2
        assert len(ds.img) == 8
        assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(ds.abc) == 2
        assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
        ds.checkout(first)
        assert len(ds.tensors) == 2
        assert len(ds.img) == 5
        assert (ds.img.numpy() == np.ones((5, 50, 50))).all()
        assert len(ds.abc) == 2
        assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()

        # will autocheckout to new branch
        ds.create_tensor("ghi")
        ds.ghi.extend(np.ones((2, 10, 10)))
        ds.img.extend(np.ones((2, 50, 50)))
        ds.abc.extend(np.ones((3, 10, 10)))
        assert len(ds.tensors) == 3
        assert len(ds.img) == 7
        assert (ds.img.numpy() == np.ones((7, 50, 50))).all()
        assert len(ds.abc) == 5
        assert (ds.abc.numpy() == np.ones((5, 10, 10))).all()
        assert len(ds.ghi) == 2
        assert (ds.ghi.numpy() == np.ones((2, 10, 10))).all()
        third = ds.commit(
            "stored 2 more images, 3 more abc in other branch, created ghi"
        )
        ds.checkout(first)
        assert len(ds.tensors) == 2
        assert len(ds.img) == 5
        assert (ds.img.numpy() == np.ones((5, 50, 50))).all()
        assert len(ds.abc) == 2
        assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
        ds.checkout(second)
        assert len(ds.tensors) == 2
        assert len(ds.img) == 8
        assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(ds.abc) == 2
        assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
        ds.checkout(third)
        assert len(ds.tensors) == 3
        assert len(ds.img) == 7
        assert (ds.img.numpy() == np.ones((7, 50, 50))).all()
        assert len(ds.abc) == 5
        assert (ds.abc.numpy() == np.ones((5, 10, 10))).all()
        ds.checkout("main")
        assert len(ds.tensors) == 2
        assert len(ds.img) == 8
        assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
        assert len(ds.abc) == 2
        assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()

    path = ds.path
    if path.startswith("mem://"):
        # memory datasets are not persistent
        return

    # reloading the dataset to check persistence
    ds = hub.dataset(path)
    assert len(ds.tensors) == 2
    assert len(ds.img) == 8
    assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(ds.abc) == 2
    assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
    ds.checkout(first)
    assert len(ds.tensors) == 2
    assert len(ds.img) == 5
    assert (ds.img.numpy() == np.ones((5, 50, 50))).all()
    assert len(ds.abc) == 2
    assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
    ds.checkout(second)
    assert len(ds.tensors) == 2
    assert len(ds.img) == 8
    assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(ds.abc) == 2
    assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
    ds.checkout(third)
    assert len(ds.tensors) == 3
    assert len(ds.img) == 7
    assert (ds.img.numpy() == np.ones((7, 50, 50))).all()
    assert len(ds.abc) == 5
    assert (ds.abc.numpy() == np.ones((5, 10, 10))).all()
    ds.checkout("main")
    assert len(ds.tensors) == 2
    assert len(ds.img) == 8
    assert (ds.img.numpy() == np.ones((8, 50, 50))).all()
    assert len(ds.abc) == 2
    assert (ds.abc.numpy() == np.ones((2, 10, 10))).all()
