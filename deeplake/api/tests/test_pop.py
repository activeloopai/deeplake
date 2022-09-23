import numpy as np
import hub
from hub.api.tests.test_api_tiling import compressions_paremetrized
import pytest


def pop_helper_basic(ds, pop_count):
    for i in range(len(ds.xyz)):
        ofs = 1 if i < 5 else 1 + pop_count
        target = i + ofs
        assert ds.xyz[i].shape == ds.xyz[i].numpy().shape == (target, target)
        np.testing.assert_array_equal(
            ds.xyz[i].numpy(), target * np.ones((target, target))
        )


def pop_helper_link(ds):
    assert len(ds.xyz) == 9
    for i in range(9):
        target = (513, 464, 4) if i % 2 == 0 else (900, 900, 3)
        assert ds.xyz[i].numpy().shape == ds.xyz[i].shape == target


def test_multiple(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("xyz")
        for i in range(1, 11):
            ds.xyz.append(i * np.ones((i, i)))
        for pop_count in range(1, 6):
            ds.xyz.pop(5)
            assert len(ds.xyz) == 10 - pop_count
            pop_helper_basic(ds, pop_count)

    ds = local_ds_generator()
    pop_helper_basic(ds, 5)

    with ds:
        ds.xyz.append(20 * np.ones((20, 20)))

    assert len(ds) == 6
    np.testing.assert_array_equal(ds.xyz[5].numpy(), 20 * np.ones((20, 20)))

    for i in range(6):
        ds.xyz.pop(0)
    assert len(ds) == 0
    assert ds.xyz.meta.max_shape == []
    assert ds.xyz.meta.min_shape == []

    ds = local_ds_generator()
    assert len(ds) == 0
    assert ds.xyz.meta.max_shape == []
    assert ds.xyz.meta.min_shape == []

    with ds:
        ds.xyz.append(30 * np.ones((30, 30)))

    assert len(ds) == 1
    np.testing.assert_array_equal(ds.xyz[0].numpy(), 30 * np.ones((30, 30)))


def test_link_pop(local_ds_generator, cat_path, flower_path):
    with local_ds_generator() as ds:
        ds.create_tensor("xyz", htype="link[image]")
        for i in range(10):
            url = cat_path if i % 2 == 0 else flower_path
            ds.xyz.append(hub.link(url))
        assert ds.xyz[0].numpy().shape == ds.xyz[0].shape == (900, 900, 3)
        ds.xyz.pop(0)
        assert len(ds.xyz) == 9
        pop_helper_link(ds)

    ds = local_ds_generator()
    assert len(ds.xyz) == 9
    pop_helper_link(ds)

    ds.xyz.append(hub.link(cat_path))
    assert ds.xyz[9].numpy().shape == ds.xyz[9].shape == (900, 900, 3)


def test_tiling_pop(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("xyz")
        ds.xyz.append(np.ones((1000, 1000, 3)))
        ds.xyz.append(2 * np.ones((2000, 2000, 3)))
        ds.xyz.append(3 * np.ones((3000, 3000, 3)))
        ds.xyz.pop(1)
        assert len(ds.xyz) == 2

        assert ds.xyz[0].numpy().shape == ds.xyz[0].shape == (1000, 1000, 3)
        np.testing.assert_array_equal(ds.xyz[0].numpy(), np.ones((1000, 1000, 3)))
        assert ds.xyz[1].numpy().shape == ds.xyz[1].shape == (3000, 3000, 3)
        np.testing.assert_array_equal(ds.xyz[1].numpy(), 3 * np.ones((3000, 3000, 3)))


@compressions_paremetrized
def test_compressions_pop(local_ds_generator, compression):
    ds = local_ds_generator()
    rint = np.random.randint
    ls = [
        rint(0, 255, (rint(50, 100), rint(50, 100), 3), dtype=np.uint8)
        for _ in range(6)
    ]
    with ds:
        ds.create_tensor("xyz", **compression)
        for i in range(5):
            ds.xyz.append(ls[i])

        ds.xyz.pop(2)
        assert len(ds.xyz) == 4
        np.testing.assert_array_equal(ds.xyz[0].numpy(), ls[0])
        np.testing.assert_array_equal(ds.xyz[1].numpy(), ls[1])
        np.testing.assert_array_equal(ds.xyz[2].numpy(), ls[3])
        np.testing.assert_array_equal(ds.xyz[3].numpy(), ls[4])

        ds.xyz.pop(0)
        assert len(ds.xyz) == 3
        np.testing.assert_array_equal(ds.xyz[0].numpy(), ls[1])
        np.testing.assert_array_equal(ds.xyz[1].numpy(), ls[3])
        np.testing.assert_array_equal(ds.xyz[2].numpy(), ls[4])

    ds = local_ds_generator()
    assert len(ds.xyz) == 3
    np.testing.assert_array_equal(ds.xyz[0].numpy(), ls[1])
    np.testing.assert_array_equal(ds.xyz[1].numpy(), ls[3])
    np.testing.assert_array_equal(ds.xyz[2].numpy(), ls[4])

    with ds:
        ds.xyz.append(ls[5])
        assert len(ds.xyz) == 4
        np.testing.assert_array_equal(ds.xyz[0].numpy(), ls[1])
        np.testing.assert_array_equal(ds.xyz[1].numpy(), ls[3])
        np.testing.assert_array_equal(ds.xyz[2].numpy(), ls[4])
        np.testing.assert_array_equal(ds.xyz[3].numpy(), ls[5])


def test_sequence_pop(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("xyz", htype="sequence")
        ds.xyz.append([[1, 2, 3], [4, 5, 6, 7]])
        ds.xyz.append([[8, 9], [11, 12, 13, 14]])
        ds.xyz.pop(0)
        assert len(ds.xyz) == 1
        val = ds.xyz[0].numpy(aslist=True)
        assert len(val) == 2
        assert (val[0] == [8, 9]).all()
        assert (val[1] == [11, 12, 13, 14]).all()

    ds = local_ds_generator()
    assert len(ds.xyz) == 1
    val = ds.xyz[0].numpy(aslist=True)
    assert len(val) == 2
    assert (val[0] == [8, 9]).all()
    assert (val[1] == [11, 12, 13, 14]).all()


def test_diff_pop(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc")
        a = ds.commit("first commit")
        for i in range(5):
            ds.abc.append(i)

        b = ds.commit("added 5 samples")

        ds.abc[2] = -2
        ds.abc[3] = -3
        ds.abc.pop(3)
        ds.abc.append(5)
        ds.abc.append(6)

        c = ds.commit("second commit")
        ds.abc.pop(2)

        diff1, diff2 = ds.diff(b, as_dict=True)["tensor"]
        assert diff1 == {
            "abc": {
                "created": False,
                "cleared": False,
                "info_updated": False,
                "data_transformed_in_place": False,
                "data_added": [3, 5],
                "data_updated": set(),
                "data_deleted": {2, 3},
            }
        }
        assert diff2 == {}


def test_ds_pop(local_ds):
    with local_ds as ds:
        ds.create_tensor("images")
        ds.create_tensor("labels")

        with pytest.raises(IndexError):
            ds.pop()

        for i in range(100):
            ds.images.append(i * np.ones((i + 1, i + 1, 3)))
            if i < 50:
                ds.labels.append(i)

        ds.pop(80)  # doesn't pop from tensors shorter than length 80
        assert len(ds.images) == 99
        assert len(ds.labels) == 50

        ds.pop(20)
        assert len(ds.images) == 98
        assert len(ds.labels) == 49

        ds.pop()  # only pops from the longest tensor
        assert len(ds.images) == 97
        assert len(ds.labels) == 49

        with pytest.raises(IndexError):
            ds.pop(-5)
