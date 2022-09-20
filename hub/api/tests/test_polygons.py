import hub
import pytest
import numpy as np


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}])
def test_polygons(memory_ds, ndim, args):
    with memory_ds as ds:
        ds.create_tensor("x", htype="polygon", **args)
        samples = []
        num_samples = 10
        for _ in range(num_samples):
            num_polygons = np.random.randint(10, 100)
            polygons = []
            for _ in range(num_polygons):
                num_points = np.random.randint(10, 100)
                polygon = np.random.randint(0, 1000, (num_points, ndim))
                polygons.append(polygon)
            samples.append(polygons)
        for i in range(num_samples // 2):
            ds.x.append(samples[i])
        ds.x.extend(samples[num_samples // 2 :])
        samples2 = ds.x.numpy()
        assert len(samples) == len(samples2)
        for s1, s2 in zip(samples, samples2):
            assert len(s1) == len(s2)
            assert type(s2) == list
            for p1, p2 in zip(s1, s2):
                assert isinstance(p2, np.ndarray)
                np.testing.assert_array_equal(p1, p2)
