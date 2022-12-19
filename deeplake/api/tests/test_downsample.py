import deeplake
import numpy as np


def test_downsample(local_ds_generator, cat_path):
    with local_ds_generator() as ds:
        ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", downsampling=(2, 7)
        )
        tensors = set(ds._tensors(include_hidden=True).keys())
        downsampled_tensors = {
            "_image_downsampled_2",
            "_image_downsampled_4",
            "_image_downsampled_8",
            "_image_downsampled_16",
            "_image_downsampled_32",
            "_image_downsampled_64",
            "_image_downsampled_128",
        }
        assert tensors.issuperset(downsampled_tensors)
        downsampled_tensors.remove("_image_downsampled_128")
        for tensor in downsampled_tensors:
            assert ds[tensor].info["downsampling_factor"] == 2
        ds.image.append(deeplake.read(cat_path))
        assert ds["_image_downsampled_128"][0].numpy().shape == (0, 0, 0)
        cats = [ds[tensor][0].numpy() for tensor in downsampled_tensors]
    ds = local_ds_generator()
    ds.image[0] = np.random.randint(0, 255, size=(813, 671, 3), dtype=np.uint8)
    assert ds["_image_downsampled_128"][0].numpy().shape == (0, 0, 0)
    arrs = [ds[tensor][0].numpy() for tensor in downsampled_tensors]
    for cat, arr in zip(cats, arrs):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(cat, arr)


def test_downsample_tiled(memory_ds):
    with memory_ds as ds:
        ds.create_tensor(
            "image",
            htype="image",
            sample_compression="jpeg",
            tiling_threshold=1024 * 1024,
            downsampling=(3, 5),
        )
        ds.image.append(deeplake.tiled(sample_shape=(3648, 5472 * 4, 3)))
        arr = np.zeros((3648, 5472, 3), dtype=np.uint8)
        for i in range(4):
            x = i * 5472
            ds.image[0][0:3648, x : x + 5472, :] = arr
