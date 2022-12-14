import numpy as np
import deeplake
from deeplake.api.tests.test_api_tiling import compressions_paremetrized
import pytest

from deeplake.core.version_control.test_version_control import (
    compare_dataset_diff,
    compare_tensor_diff,
    get_default_tensor_diff,
    get_default_dataset_diff,
)


@deeplake.compute
def pop_fn(sample_in, samples_out):
    samples_out.x.append(sample_in)


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
        ds.create_tensor("xyz", htype="link[image]", sample_compression="jpg")
        for i in range(10):
            url = cat_path if i % 2 == 0 else flower_path
            ds.xyz.append(deeplake.link(url))
        assert ds.xyz[0].numpy().shape == ds.xyz[0].shape == (900, 900, 3)
        ds.xyz.pop(0)
        assert len(ds.xyz) == 9
        pop_helper_link(ds)

    ds = local_ds_generator()
    assert len(ds.xyz) == 9
    pop_helper_link(ds)

    ds.xyz.append(deeplake.link(cat_path))
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
        expected_tensor_diff_from_a = {
            "commit_id": ds.pending_commit_id,
            "abc": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_a = get_default_dataset_diff(ds.pending_commit_id)
        for i in range(5):
            ds.abc.append(i)
        expected_tensor_diff_from_a["abc"]["data_added"] = [0, 5]

        b = ds.commit("added 5 samples")
        expected_tensor_diff_from_b = {
            "commit_id": ds.pending_commit_id,
            "abc": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_b = get_default_dataset_diff(ds.pending_commit_id)

        ds.abc[2] = -2
        ds.abc[3] = -3
        expected_tensor_diff_from_b["abc"]["data_updated"] = {2, 3}
        ds.abc.pop(3)
        expected_tensor_diff_from_b["abc"]["data_deleted"] = {3}
        expected_tensor_diff_from_b["abc"]["data_updated"] = {2}
        ds.abc.append(5)
        ds.abc.append(6)
        expected_tensor_diff_from_b["abc"]["data_added"] = [4, 6]

        c = ds.commit("second commit")
        expected_tensor_diff_from_c = {
            "commit_id": ds.pending_commit_id,
            "abc": get_default_tensor_diff(),
        }
        expected_dataset_diff_from_c = get_default_dataset_diff(ds.pending_commit_id)
        ds.abc.pop(2)
        expected_tensor_diff_from_c["abc"]["data_deleted"] = {2}

        diff = ds.diff(a, as_dict=True)
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

        compare_tensor_diff(expected_tensor_diff[0], tensor_diff[0])
        compare_tensor_diff(expected_tensor_diff[1], tensor_diff[1])
        compare_dataset_diff(expected_dataset_diff[0], dataset_diff[0])
        compare_dataset_diff(expected_dataset_diff[1], dataset_diff[1])


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


def test_pop_bug(local_ds_generator):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("x")

    pop_fn().eval([1, 2, 3, 4], ds, num_workers=4),

    np.testing.assert_array_equal(ds.x.numpy().squeeze(), [1, 2, 3, 4])
    a = ds.commit()
    ds.pop(2)
    np.testing.assert_array_equal(ds.x.numpy().squeeze(), [1, 2, 4])
    ds.checkout(a)
    np.testing.assert_array_equal(ds.x.numpy().squeeze(), [1, 2, 3, 4])

    ds = local_ds_generator()
    np.testing.assert_array_equal(ds.x.numpy().squeeze(), [1, 2, 4])
    ds.checkout(a)
    np.testing.assert_array_equal(ds.x.numpy().squeeze(), [1, 2, 3, 4])


def test_pop_tiled(local_ds_generator):
    ds = local_ds_generator()
    arr1 = np.random.random((3, 1, 2))
    arr2 = np.random.random((2, 1, 3))
    arr3 = np.random.random((50, 5, 1))
    arr4 = np.random.random((1, 2, 3))
    arrs = [arr1, arr2, arr3, arr4]
    with ds:
        ds.create_tensor("x", max_chunk_size=1024, tiling_threshold=1024)
        ds.x.append(arr1)
        ds.x.append(arr2)
        ds.x.append(arr3)
        ds.x.append(arr4)
    assert not ds.x.chunk_engine._is_tiled_sample(0)
    assert not ds.x.chunk_engine._is_tiled_sample(1)
    assert ds.x.chunk_engine._is_tiled_sample(2)
    assert not ds.x.chunk_engine._is_tiled_sample(3)
    with ds:
        ds.pop(1)
    arrs.pop(1)
    for arr, sample in zip(arrs, ds.x):
        np.testing.assert_array_equal(sample, arr)

    assert not ds.x.chunk_engine._is_tiled_sample(0)
    assert ds.x.chunk_engine._is_tiled_sample(1)
    assert not ds.x.chunk_engine._is_tiled_sample(2)
