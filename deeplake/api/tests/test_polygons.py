import deeplake
import pytest
import numpy as np


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_polygons(local_ds, ndim, args):
    with local_ds as ds:
        ds.create_tensor("polygons", htype="polygon", **args)
        samples = []
        samples.append(np.random.randint(0, 10, (5, 7, 2)))
        num_samples = 10
        for _ in range(1, num_samples):
            num_polygons = np.random.randint(1, 10)
            polygons = []
            for _ in range(num_polygons):
                num_points = np.random.randint(3, 10)
                polygon = np.random.randint(0, 100, (num_points, ndim))
                polygons.append(polygon)
            samples.append(polygons)
        for i in range(num_samples // 2):
            ds.polygons.append(samples[i])
        ds.polygons.extend(samples[num_samples // 2 :])
        samples2 = ds.polygons.numpy()
        assert len(samples) == len(samples2)
        for s1, s2 in zip(samples, samples2):
            assert len(s1) == len(s2)
            assert type(s2) == list
            for p1, p2 in zip(s1, s2):
                assert isinstance(p2, np.ndarray)
                np.testing.assert_array_equal(p1, p2)
    for i, sample in enumerate(ds.pytorch(num_workers=2)):
        assert len(samples[i]) == len(sample["polygons"])
        for p1, p2 in zip(samples[i], sample["polygons"]):
            np.testing.assert_array_equal(p1, p2[0])
    idxs = [2, 2, 6, 4, 6, 7]
    view = ds[idxs]
    ds.commit()
    view.save_view()
    materialized = deeplake.empty("mem://")
    deeplake.copy(view, materialized)


def test_fixed_shape_bug(memory_ds):
    arr = np.random.randint(
        0,
        10,
        (
            5,
            7,
            2,
        ),
    )
    with memory_ds as ds:
        ds.create_tensor("polygons", htype="polygon")
        ds.polygons.append(arr)
    np.testing.assert_array_equal(ds.polygons[0], arr)


def test_polygon_disabled_cache(memory_ds):
    arr1 = np.random.randint(0, 10, (3, 3, 2))
    arr2 = np.random.randint(0, 10, (3, 3, 2))
    with memory_ds as ds:
        ds.create_tensor("polygons", htype="polygon")
        ds.polygons.append(arr1)
        ds.polygons.append(arr2)
    np.testing.assert_array_equal(ds.polygons.numpy()[0], arr1)
    np.testing.assert_array_equal(ds.polygons.numpy()[1], arr2)
