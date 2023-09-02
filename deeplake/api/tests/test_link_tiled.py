import deeplake
import itertools
import numpy as np
import pytest

from deeplake.util.exceptions import SampleUpdateError


@deeplake.compute
def add_link_tiled(sample_in, samples_out):
    arr = np.empty((10, 10), dtype=object)
    for j, i in itertools.product(range(10), range(10)):
        arr[j, i] = sample_in
    samples_out.image.append(deeplake.link_tiled(arr))


def check_data(actual_data, ds, index, downsampled=False):
    deeplake_data = ds.image[index]
    shape = deeplake_data.shape
    assert shape == (9000, 9000, 3)

    if downsampled:
        downsampled_data = ds._image_downsampled_2[index]
        assert downsampled_data.shape == (4500, 4500, 3)
        downsampled_numpy = ds._image_downsampled_2[index].numpy()
        assert downsampled_numpy.shape == (4500, 4500, 3)

    deeplake_numpy = ds.image[index].numpy()
    assert deeplake_numpy.shape == (9000, 9000, 3)

    for j, i in itertools.product(range(10), range(10)):
        deeplake_numpy_sliced = deeplake_numpy[
            j * 900 : (j + 1) * 900, i * 900 : (i + 1) * 900, :
        ]
        deeplake_data_sliced = deeplake_data[
            j * 900 : (j + 1) * 900, i * 900 : (i + 1) * 900, :
        ].numpy()

        np.testing.assert_array_equal(deeplake_numpy_sliced, actual_data)
        np.testing.assert_array_equal(deeplake_data_sliced, actual_data)

    deeplake_data_sliced = ds.image[0, 450:1350, 450:1350, :].numpy()
    np.testing.assert_array_equal(
        deeplake_data_sliced[:450, :450, :], actual_data[450:, 450:, :]
    )
    np.testing.assert_array_equal(
        deeplake_data_sliced[450:, 450:, :], actual_data[:450, :450, :]
    )


@pytest.mark.slow
def test_link_tiled(local_ds_generator, cat_path):
    arr = np.empty((10, 10), dtype=object)
    for j, i in itertools.product(range(10), range(10)):
        arr[j, i] = cat_path
    linked_sample = deeplake.link_tiled(arr)

    with local_ds_generator() as ds:
        ds.create_tensor(
            "image",
            htype="link[image]",
            sample_compression="jpeg",
            downsampling=[2, 1],
            create_shape_tensor=False,
        )
        ds.image.append(linked_sample)

    actual_data = deeplake.read(cat_path).array
    ds = local_ds_generator()
    index = 0
    check_data(actual_data, ds, index, downsampled=True)
    with pytest.raises(SampleUpdateError):
        ds.image[index][100:1000, 100:1000, :] = deeplake.link(cat_path)

    with ds:
        ds.image.extend([linked_sample, linked_sample])
    check_data(actual_data, ds, 1, downsampled=True)
    check_data(actual_data, ds, 2, downsampled=True)

    sample = ds.image[0]._linked_sample()
    np.testing.assert_array_equal(sample.path_array.flatten(), arr.flatten())
    assert sample.creds_key is None


@pytest.mark.slow
def test_link_tiled_transform(local_ds_generator, cat_path):
    data_in = [cat_path] * 2
    with local_ds_generator() as ds:
        ds.create_tensor("image", htype="link[image]", sample_compression="jpeg")
        add_link_tiled().eval(data_in, ds, num_workers=2)

    actual_data = deeplake.read(cat_path).array
    ds = local_ds_generator()
    for index in range(2):
        check_data(actual_data, ds, index)
        with pytest.raises(SampleUpdateError):
            ds.image[index][100:1000, 100:1000, :] = deeplake.link(cat_path)
