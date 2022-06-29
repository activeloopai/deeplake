import numpy as np
import hub


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
        ds.create_tensor("xyz", htype="link")
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
