from hub.exceptions import (
    AddressNotFound,
    ReadModeException,
    VersioningNotSupportedException,
)
from hub.schema import Image
import hub
import numpy as np
import pytest


def test_commit():
    my_schema = {"abc": "uint32"}
    ds = hub.Dataset(
        "./data/test_versioning/eg_1", shape=(10,), schema=my_schema, mode="w"
    )
    ds["abc", 0] = 1
    a = ds.commit("first")
    ds["abc", 0] = 2
    b = ds.commit("second")
    ds["abc", 0] = 3
    c = ds.commit("third")
    assert ds["abc", 0].compute() == 3
    ds.checkout(a)
    assert ds["abc", 0].compute() == 1
    ds.checkout(b)
    assert ds["abc", 0].compute() == 2
    ds.checkout(c)
    assert ds["abc", 0].compute() == 3


def test_commit_checkout():
    my_schema = {"img": hub.schema.Tensor((1000, 1000, 3))}
    ds = hub.Dataset("./data/eg_1", shape=(10,), schema=my_schema, mode="w")

    for i in range(10):
        ds["img", i] = np.ones((1000, 1000, 3))

    first_commit_id = ds.commit("stored all ones")

    for i in range(5):
        ds["img", i] = ds["img", i].compute() * 2

    second_commit_id = ds.commit("multiplied value of some images by 2")

    assert (ds["img", 4].compute() == 2 * np.ones((1000, 1000, 3))).all()

    ds.checkout(first_commit_id)  # now all images are ones again

    for i in range(10):
        assert (ds["img", i].compute() == np.ones((1000, 1000, 3))).all()

    ds.checkout(
        "alternate", create=True
    )  # creating a new branch as we are currently not on the head of master

    for i in range(5):
        ds["img", i] = ds["img", i].compute() * 3

    #  if we had not checked out to "alternate" branch earlier here it would auto checkout to a new branch
    ds.commit("multiplied value of some images by 3")

    assert (ds["img", 4].compute() == 3 * np.ones((1000, 1000, 3))).all()

    ds.checkout(second_commit_id)  # first 5 images are 2s, rest are 1s now

    for i in range(5, 10):
        ds["img", i] = ds["img", i].compute() * 2

    # we are not at the head of master but rather at the last commit, so we automatically get checkouted out to a new branch here
    # this happens any time we try to commit when we are not at the head of the branch
    ds.commit("multiplied value of remaining images by 2")

    for i in range(10):
        assert (ds["img", i].compute() == 2 * np.ones((1000, 1000, 3))).all()

    ds.checkout("alternate")

    for i in range(5, 10):
        ds["img", i] = ds["img", i].compute() * 3

    for i in range(10):
        assert (ds["img", i].compute() == 3 * np.ones((1000, 1000, 3))).all()

    # we are already at the head of alternate so it does not check us out to a new branch, rather we commit on the alternate branch itself
    ds.commit("multiplied value of remaining images by 3")


def test_commit_checkout_2():
    my_schema = {
        "abc": "uint32",
        "img": Image((1000, 1000, 3), dtype="uint16"),
    }
    ds = hub.Dataset(
        "./data/test_versioning/eg_3", shape=(100,), schema=my_schema, mode="w"
    )
    for i in range(100):
        ds["img", i] = i * np.ones((1000, 1000, 3))
    a = ds.commit("first")

    # chunk 7.0.0.0 gets rewritten
    ds["img", 21] = 2 * ds["img", 21].compute()

    # the rest part of the chunk stays intact
    assert (ds["img", 21].compute() == 2 * 21 * np.ones((1000, 1000, 3))).all()
    assert (ds["img", 22].compute() == 22 * np.ones((1000, 1000, 3))).all()
    assert (ds["img", 23].compute() == 23 * np.ones((1000, 1000, 3))).all()

    # other chunks are still accessed from original chunk, for eg chunk 11 that contains 35th sample has single copy
    assert (ds["img", 35].compute() == 35 * np.ones((1000, 1000, 3))).all()

    b = ds.commit("second")

    # going back to first commit
    ds.checkout(a)

    # sanity check
    assert (ds["img", 21].compute() == 21 * np.ones((1000, 1000, 3))).all()

    ds.checkout("another", create=True)

    ds["img", 21] = 3 * ds["img", 21].compute()
    assert (
        ds["img", 21].compute() == 3 * 21 * np.ones((1000, 1000, 3))
    ).all()  # and not 6 * 21 as it would have been, had we checked out from b

    ds.commit("first2")

    ds.checkout("master")
    assert (ds["img", 21].compute() == 2 * 21 * np.ones((1000, 1000, 3))).all()
    ds.log()


def test_auto_checkout_bug():
    my_schema = {"abc": "uint8"}
    ds = hub.Dataset(
        "./data/test_versioning/branch_bug", shape=(10,), schema=my_schema, mode="w"
    )
    ds["abc", 0] = 1
    a = ds.commit("it is 1")
    ds["abc", 0] = 2
    b = ds.commit("it is 2")
    c = ds.checkout(a)
    d = ds.checkout("other", True)
    ds["abc", 0] = 3
    e = ds.commit("it is 3")
    ds.checkout(b)
    ds["abc", 0] = 4
    f = ds.commit("it is 4")
    g = ds.checkout(a)
    dsv = ds[0:3]
    dsv["abc", 0] = 5
    h = ds.commit("it is 5")
    i = ds.checkout(e)
    tsv = ds[0:5, "abc"]
    tsv[0] = 6
    j = ds.commit("it is 6")
    ds.log()
    ds.checkout(a)
    assert dsv["abc", 0].compute() == 1
    assert ds["abc", 0].compute() == 1
    ds.checkout(b)
    assert ds["abc", 0].compute() == 2
    ds.checkout(c)
    assert ds["abc", 0].compute() == 1
    ds.checkout(d)
    assert ds["abc", 0].compute() == 3
    ds.checkout(e)
    assert ds["abc", 0].compute() == 3
    ds.checkout(f)
    assert ds["abc", 0].compute() == 4
    ds.checkout(g)
    assert ds["abc", 0].compute() == 1
    ds.checkout(h)
    assert ds["abc", 0].compute() == 5
    ds.checkout(i)
    assert ds["abc", 0].compute() == 3
    ds.checkout(j)
    assert ds["abc", 0].compute() == 6
    ds.checkout("master")
    assert ds["abc", 0].compute() == 2
    ds["abc", 0] = 7
    ds.checkout("copy", True)
    assert ds["abc", 0].compute() == 7
    ds.checkout("other")
    assert ds["abc", 0].compute() == 3


def test_read_mode():
    my_schema = {"abc": "uint8"}
    ds = hub.Dataset("./data/test_versioning/read_ds", schema=my_schema, shape=(10,))
    ds.checkout("second", create=True)
    ds2 = hub.Dataset("./data/test_versioning/read_ds", mode="r")
    with pytest.raises(ReadModeException):
        ds2.commit("first")
    with pytest.raises(ReadModeException):
        ds2.checkout("third", create=True)
    with pytest.raises(ReadModeException):
        ds2["abc", 4] = 10


def test_checkout_address_not_found():
    my_schema = {"abc": "uint8"}
    ds = hub.Dataset("./data/test_versioning/ds_address", schema=my_schema, shape=(10,))
    with pytest.raises(AddressNotFound):
        ds.checkout("second")


def test_old_datasets():
    ds = hub.Dataset("activeloop/mnist")
    with pytest.raises(VersioningNotSupportedException):
        ds.checkout("third")
    with pytest.raises(VersioningNotSupportedException):
        ds.checkout("third", create=True)
    with pytest.raises(VersioningNotSupportedException):
        ds.log()


def test_dynamic_version_control():
    my_schema = {"img": Image((None, None, 3), max_shape=(1000, 1000, 3))}
    ds = hub.Dataset(
        "./data/dynamic_versioning", shape=(10,), schema=my_schema, mode="w"
    )
    for i in range(10):
        ds["img", i] = i * np.ones((100, 100, 3))

    a = ds.commit("first")
    for i in range(10):
        ds["img", i] = 2 * i * np.ones((150, 150, 3))
    ds.checkout(a)

    for i in range(10):
        assert (ds["img", i].compute() == i * np.ones((100, 100, 3))).all()

    ds.checkout("master")

    for i in range(10):
        assert (ds["img", i].compute() == 2 * i * np.ones((150, 150, 3))).all()


if __name__ == "__main__":
    test_commit()
    test_commit_checkout()
    test_commit_checkout_2()
    test_auto_checkout_bug()
    test_read_mode()
    test_old_datasets()
    test_checkout_address_not_found()
