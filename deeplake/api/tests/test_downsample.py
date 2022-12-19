import deeplake
import numpy as np


def test_downsample(local_ds_generator, cat_path, flower_path):
    with local_ds_generator() as ds:
        ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", downsampling=(2, 3)
        )
        tensors = set(ds._tensors(include_hidden=True).keys())
        downsampled_tensors = {
            "_image_downsampled_2",
            "_image_downsampled_4",
            "_image_downsampled_8",
        }
        assert tensors.issuperset(downsampled_tensors)
        for tensor in downsampled_tensors:
            assert ds[tensor].info["downsampling_factor"] == 2
        ds.image.append(deeplake.read(cat_path))
        cats = []
        cats.append(ds.image[0].numpy())
        cats.append(ds["_image_downsampled_2"][0].numpy())
        cats.append(ds["_image_downsampled_4"][0].numpy())
        cats.append(ds["_image_downsampled_8"][0].numpy())

    ds = local_ds_generator()
    ds.image[0] = np.random.randint(0, 255, size=(813, 671, 3), dtype=np.uint8)
    arrs = []
    arrs.append(ds.image[0].numpy())
    arrs.append(ds["_image_downsampled_2"][0].numpy())
    arrs.append(ds["_image_downsampled_4"][0].numpy())
    arrs.append(ds["_image_downsampled_8"][0].numpy())

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
