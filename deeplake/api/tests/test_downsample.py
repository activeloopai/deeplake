import deeplake
import numpy as np
import pytest


def test_downsample(local_ds_generator, cat_path):
    with local_ds_generator() as ds:
        ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", downsampling=(2, 7)
        )
        tensors = set(ds._tensors(include_hidden=True).keys())
        downsampled_tensors = [
            "image",
            "_image_downsampled_2",
            "_image_downsampled_4",
            "_image_downsampled_8",
            "_image_downsampled_16",
            "_image_downsampled_32",
            "_image_downsampled_64",
            "_image_downsampled_128",
        ]
        assert tensors.issuperset(downsampled_tensors)
        for tensor in downsampled_tensors[1:]:
            assert ds[tensor].info["downsampling_factor"] == 2
        ds.image.extend([deeplake.read(cat_path), deeplake.read(cat_path)])
        cats = [ds[tensor][0].numpy() for tensor in downsampled_tensors]
        expected_shapes = [
            (900, 900, 3),
            (450, 450, 3),
            (225, 225, 3),
            (112, 112, 3),
            (56, 56, 3),
            (28, 28, 3),
            (14, 14, 3),
            (0, 0, 0),
        ]
        for cat, shape in zip(cats, expected_shapes):
            assert cat.shape == shape

    ds = local_ds_generator()
    ds.image[0] = np.random.randint(0, 255, size=(813, 671, 3), dtype=np.uint8)
    arrs = [ds[tensor][0].numpy() for tensor in downsampled_tensors]
    expected_shapes = [
        (813, 671, 3),
        (406, 335, 3),
        (203, 167, 3),
        (101, 83, 3),
        (50, 41, 3),
        (25, 20, 3),
        (12, 10, 3),
        (0, 0, 0),
    ]
    for arr, shape in zip(arrs, expected_shapes):
        assert arr.shape == shape
    for cat, arr in zip(cats[:-1], arrs[-1]):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(cat, arr)


def test_downsample_link(local_ds, cat_path):
    with local_ds as ds:
        ds.create_tensor(
            "image", htype="link[image]", sample_compression="jpeg", downsampling=(2, 7)
        )
        tensors = set(ds._tensors(include_hidden=True).keys())
        downsampled_tensors = [
            "image",
            "_image_downsampled_2",
            "_image_downsampled_4",
            "_image_downsampled_8",
            "_image_downsampled_16",
            "_image_downsampled_32",
            "_image_downsampled_64",
            "_image_downsampled_128",
        ]
        assert tensors.issuperset(downsampled_tensors)
        for tensor in downsampled_tensors[1:]:
            assert ds[tensor].info["downsampling_factor"] == 2
        ds.image.append(deeplake.link(cat_path))
        cats = [ds[tensor][0].numpy() for tensor in downsampled_tensors]
        expected_shapes = [
            (900, 900, 3),
            (450, 450, 3),
            (225, 225, 3),
            (112, 112, 3),
            (56, 56, 3),
            (28, 28, 3),
            (14, 14, 3),
            (0, 0, 0),
        ]
        for cat, shape in zip(cats, expected_shapes):
            assert cat.shape == shape


@pytest.mark.slow
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


@pytest.mark.parametrize(
    "sample_compression", [None]
)  # add back apng when bug is fixed
def test_downsample_binary_mask(memory_ds, sample_compression):
    with memory_ds as ds:
        ds.create_tensor(
            "mask",
            htype="binary_mask",
            sample_compression=sample_compression,
            downsampling=(2, 5),
        )
        binary_masks = [
            np.ones((1000, 1000, 3), dtype=bool),
            np.zeros((1000, 1000, 3), dtype=bool),
        ]
        ds.mask.extend(binary_masks)

        for i in range(1, 6):
            tensor = ds[f"_mask_downsampled_{2 ** i}"]
            assert len(tensor) == 2
            for j in range(2):
                np.testing.assert_array_equal(
                    tensor[j], binary_masks[j][:: 2**i, :: 2**i, :]
                )


def test_downsample_group_bug(memory_ds):
    with memory_ds as ds:
        ds.create_group("stuff")
        ds.create_tensor(
            "mask", htype="binary_mask", sample_compression="lz4", downsampling=(2, 2)
        )
        ds.create_tensor(
            "stuff/mask",
            htype="binary_mask",
            sample_compression="lz4",
            downsampling=(2, 2),
        )


def test_downsample_image(memory_ds):
    with memory_ds as ds:
        ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", downsampling=(2, 2)
        )
        ds.image.append(np.ones((100, 100, 3), dtype="uint8"))
        ds.image.append(np.ones((100, 100, 1), dtype="uint8"))
        ds.image.append(np.ones((100, 100, 0), dtype="uint8"))
        ds.image.append(np.ones((100, 0, 3), dtype="uint8"))
        ds.image.append(np.ones((100, 100), dtype="uint8"))

        target_shapes = {
            "image": [
                (100, 100, 3),
                (100, 100, 1),
                (100, 100, 0),
                (100, 0, 3),
                (100, 100, 1),
            ],
            "_image_downsampled_2": [
                (50, 50, 3),
                (50, 50, 1),
                (0, 0, 0),
                (0, 0, 0),
                (50, 50, 1),
            ],
            "_image_downsampled_4": [
                (25, 25, 3),
                (25, 25, 1),
                (0, 0, 0),
                (0, 0, 0),
                (25, 25, 1),
            ],
        }
        for tensor, target_shape in target_shapes.items():
            shapes = [ds[tensor][i].shape for i in range(5)]
            numpy_shapes = [ds[tensor][i].numpy().shape for i in range(5)]
            assert shapes == target_shape == numpy_shapes
