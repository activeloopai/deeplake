import numpy as np


def pop_heler(ds, pop_count):
    for i in range(len(ds.xyz)):
        ofs = 1 if i < 5 else 1 + pop_count
        target = i + ofs
        assert ds.xyz[i].shape == ds.xyz[i].numpy().shape == (target, target)
        np.testing.assert_array_equal(
            ds.xyz[i].numpy(), target * np.ones((target, target))
        )


def test_multiple(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("xyz")
        for i in range(1, 11):
            ds.xyz.append(i * np.ones((i, i)))
        for pop_count in range(1, 6):
            ds.xyz.pop(5)
            assert len(ds.xyz) == 10 - pop_count
            pop_heler(ds, pop_count)

    ds = local_ds_generator()
    pop_heler(ds, 5)
