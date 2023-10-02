from deeplake.constants import KB, MB
from deeplake.util.exceptions import SampleUpdateError, TensorDoesNotExistError
import pytest
from typing import Callable
from deeplake.tests.common import assert_array_lists_equal
import numpy as np
import deeplake


def _add_dummy_mnist(ds, **kwargs):
    compression = kwargs.get(
        "compression", {"image_compression": {"sample_compression": None}}
    )
    ds.create_tensor("images", htype="image", **compression["image_compression"])
    ds.create_tensor(
        "labels", htype="class_label", **compression.get("label_compression", {})
    )

    ds.images.extend(np.ones((10, 28, 28), dtype=np.uint8))
    ds.labels.extend(np.ones(10, dtype=np.uint8))

    return ds


def _make_update_assert_equal(
    ds_generator: Callable,
    tensor_name: str,
    index,
    value,
    check_persistence: bool = True,
):
    """Updates a tensor and checks that the data is as expected.

    Example update:
        >>> ds.tensor[0:5] = [1, 2, 3, 4, 5]

    Args:
        ds_generator (Callable): Function that returns a new dataset object with each call.
        tensor_name (str): Name of the tensor to be updated.
        index (Any): Any value that can be used as an index for updating (`ds.tensor[index] = value`).
        value (Any): Any value that can be used as a value for updating (`ds.tensor[index] = value`).
        check_persistence (bool): If True, the update will be tested to make sure it can be serialized/deserialized.
    """

    ds = ds_generator()
    assert len(ds) == 10

    tensor = ds[tensor_name]
    expected = tensor.numpy(aslist=True)

    # this is necessary because `expected` uses `aslist=True` to handle dynamic cases.
    # with `aslist=False`, this wouldn't be necessary.
    expected_value = value
    if hasattr(value, "__len__") and len(value) == 1:
        expected_value = value[0]

    # make updates
    tensor[index] = value
    expected[index] = expected_value

    # non-persistence check
    actual = tensor.numpy(aslist=True)
    assert_array_lists_equal(actual, expected)
    assert len(ds) == 10

    if check_persistence:
        ds = ds_generator()
        tensor = ds[tensor_name]
        actual = tensor.numpy(aslist=True)
        assert_array_lists_equal(actual, expected)

        # make sure no new values are recorded
        ds = ds_generator()
        assert len(ds) == 10


@pytest.mark.parametrize(
    "compression",
    [
        {
            "image_compression": {"sample_compression": None},
        },
        {
            "image_compression": {"sample_compression": None},
            "label_compression": {"sample_compression": "lz4"},
        },
        {
            "image_compression": {"sample_compression": None},
            "label_compression": {"chunk_compression": "lz4"},
        },
        {"image_compression": {"sample_compression": "png"}},
        {"image_compression": {"chunk_compression": "png"}},
        {"image_compression": {"sample_compression": "lz4"}},
        {"image_compression": {"chunk_compression": "lz4"}},
    ],
)
def test(local_ds_generator, compression):
    gen = local_ds_generator

    _add_dummy_mnist(gen(), **compression)

    # update single sample
    _make_update_assert_equal(
        gen, "images", -1, np.ones((28, 28), dtype="uint8") * 75
    )  # same shape
    _make_update_assert_equal(
        gen, "images", -1, np.ones((28, 28), dtype="uint8") * 75
    )  # same shape
    _make_update_assert_equal(
        gen, "images", 0, np.ones((28, 25), dtype="uint8") * 5
    )  # new shape
    _make_update_assert_equal(
        gen, "images", 0, np.ones((32, 32), dtype="uint8") * 5
    )  # new shape
    _make_update_assert_equal(
        gen, "images", -1, np.ones((0, 0), dtype="uint8")
    )  # empty sample (new shape)
    _make_update_assert_equal(gen, "labels", -5, np.uint8(99))
    _make_update_assert_equal(gen, "labels", 0, np.uint8(5))

    # update a range of samples
    x = np.arange(3 * 28 * 28).reshape((3, 28, 28)).astype("uint8")
    _make_update_assert_equal(gen, "images", slice(0, 3), x)  # same shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype="uint8")
    )  # new shapes
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype=int).tolist()
    )  # test downcasting python scalars
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 5, 28), dtype=np.ubyte).tolist()
    )  # test upcasting
    _make_update_assert_equal(
        gen, "images", slice(3, 5), np.zeros((2, 0, 0), dtype="uint8")
    )  # empty samples (new shape)
    _make_update_assert_equal(gen, "labels", slice(0, 5), [1, 2, 3, 4, 5])

    # update a range of samples with dynamic samples
    _make_update_assert_equal(
        gen,
        "images",
        slice(7, 10),
        [
            np.ones((28, 50), dtype="uint8") * 5,
            np.ones((0, 5), dtype="uint8"),
            np.ones((1, 1), dtype="uint8") * 10,
        ],
    )

    ds = gen()
    assert ds.images.shape_interval.lower == (10, 0, 0)
    assert ds.images.shape_interval.upper == (10, 32, 50)


@pytest.mark.parametrize("images_compression", [None, "png"])
def test_deeplake_read(local_ds_generator, images_compression, cat_path, flower_path):
    gen = local_ds_generator

    ds = gen()
    ds.create_tensor("images", htype="image", sample_compression=images_compression)
    ds.images.extend(np.zeros((10, 0, 0, 0), dtype=np.uint8))

    ds.images[0] = deeplake.read(cat_path)
    np.testing.assert_array_equal(ds.images[0].numpy(), deeplake.read(cat_path).array)

    ds.images[1] = deeplake.read(flower_path)
    np.testing.assert_array_equal(
        ds.images[1].numpy(), deeplake.read(flower_path).array
    )

    ds.images[8:10] = [deeplake.read(cat_path), deeplake.read(flower_path)]
    assert_array_lists_equal(
        ds.images[8:10].numpy(aslist=True),
        [deeplake.read(cat_path).array, deeplake.read(flower_path).array],
    )

    assert ds.images.shape_interval.lower == (10, 0, 0, 0)
    assert ds.images.shape_interval.upper == (10, 900, 900, 4)

    assert len(ds.images) == 10


def test_pre_indexed_tensor(memory_ds):
    """A pre-indexed tensor update means the tensor was already indexed into, and an update is being made to that tensor view."""

    tensor = memory_ds.create_tensor("tensor")

    tensor.append([0, 1, 2])
    tensor.append([3, 4, 5, 6, 7])
    tensor.append([8, 5])
    tensor.append([9, 10, 11])
    tensor.append([12, 13, 14, 15, 16])
    tensor.append([17, 18, 19, 20, 21])

    tensor[0:5][0] = [99, 98, 97]
    tensor[5:10][0] = [44, 44, 44, 44]
    tensor[4:10][0:2] = [[44, 44, 44, 44], [33]]

    np.testing.assert_array_equal([99, 98, 97], tensor[0])
    np.testing.assert_array_equal([44, 44, 44, 44], tensor[4])
    np.testing.assert_array_equal([33], tensor[5])

    assert tensor.shape_interval.lower == (6, 1)
    assert tensor.shape_interval.upper == (6, 5)
    assert len(tensor) == 6


def test_failures(memory_ds):
    _add_dummy_mnist(memory_ds)

    # primary axis doesn't match
    with pytest.raises(SampleUpdateError):
        memory_ds.images[0:3] = np.zeros((25, 30), dtype="uint8")
    with pytest.raises(SampleUpdateError):
        memory_ds.images[0:3] = np.zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(SampleUpdateError):
        memory_ds.images[0] = np.zeros((2, 25, 30), dtype="uint8")
    with pytest.raises(SampleUpdateError):
        memory_ds.labels[0:3] = [1, 2, 3, 4]

    # dimensionality doesn't match

    memory_ds.images[0:5] = np.zeros((5, 30), dtype="uint8")
    with pytest.raises(SampleUpdateError):
        memory_ds.labels[0:5] = np.zeros((5, 2, 3), dtype="uint8")

    # make sure no data changed
    assert len(memory_ds.images) == 10
    assert len(memory_ds.labels) == 10
    np.testing.assert_array_equal(
        memory_ds.images[:5].numpy(), np.zeros((5, 30, 1), dtype="uint8")
    )
    np.testing.assert_array_equal(
        memory_ds.images[5:].numpy(), np.ones((5, 28, 28), dtype="uint8")
    )
    np.testing.assert_array_equal(
        memory_ds.labels.numpy(), np.ones((10, 1), dtype="uint8")
    )
    assert memory_ds.images.shape == (10, None, None)
    assert memory_ds.labels.shape == (10, 1)


def test_warnings(memory_ds):
    tensor = memory_ds.create_tensor(
        "tensor", max_chunk_size=8 * KB, tiling_threshold=4 * KB
    )

    tensor.extend(np.ones((10, 12, 12), dtype="int32"))

    # this update makes (small) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[0:5] = np.zeros((5, 0, 0), dtype="int32")

    # this update makes (large) suboptimal chunks
    with pytest.warns(UserWarning):
        tensor[:] = np.zeros((10, 32, 31), dtype="int32")


@pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "png"},
        {"chunk_compression": "png"},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
    ],
)
def test_inplace_updates(memory_ds, compression):
    ds = memory_ds
    ds.create_tensor("x", **compression)
    ds.x.extend(np.zeros((5, 32, 32, 3), dtype="uint8"))
    ds.x += 1
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)))
    ds.x += ds.x
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)) * 2)
    ds.x *= np.zeros(3, dtype="uint8")
    np.testing.assert_array_equal(ds.x.numpy(), np.zeros((5, 32, 32, 3)))
    ds.x += 6
    ds.x //= 2
    np.testing.assert_array_equal(ds.x.numpy(), np.ones((5, 32, 32, 3)) * 3)
    ds.x[:3] *= 0
    np.testing.assert_array_equal(
        ds.x.numpy(),
        np.concatenate([np.zeros((3, 32, 32, 3)), np.ones((2, 32, 32, 3)) * 3]),
    )

    # Different shape
    ds.x.append(np.zeros((100, 50, 3), dtype="uint8"))
    ds.x[5] += 1
    np.testing.assert_array_equal(ds.x[5].numpy(), np.ones((100, 50, 3)))
    np.testing.assert_array_equal(
        ds.x[:5].numpy(),
        np.concatenate([np.zeros((3, 32, 32, 3)), np.ones((2, 32, 32, 3)) * 3]),
    )
    ds.x[:5] *= 0
    np.testing.assert_array_equal(ds.x[:5].numpy(), np.zeros((5, 32, 32, 3)))
    np.testing.assert_array_equal(ds.x[5].numpy(), np.ones((100, 50, 3)))


@pytest.mark.parametrize("aslist", (True, False))
@pytest.mark.parametrize(
    "idx", [3, slice(None), slice(5, 9), slice(3, 7, 2), [3, 7, 6, 4]]
)
def test_sequence_htype(memory_ds, aslist, idx):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="sequence")
        for _ in range(10):
            ds.x.append([np.ones((3, 7)) for _ in range(5)])
    assert ds.x[0].numpy().shape == (5, 3, 7)
    ds.x[idx] += 1
    expected = np.ones((10, 5, 3, 7))
    expected[idx] += 1
    np.testing.assert_array_equal(np.array(ds.x.numpy(aslist=aslist)), expected)
    assert ds.x.shape == (10, 5, 3, 7)


def test_sequence_htype_with_broadcasting(memory_ds):
    ds = memory_ds
    with ds:
        arr = np.random.randint(0, 10, (2, 7, 9))
        expected = arr.reshape(1, *arr.shape).repeat(15, 0).reshape(5, 3, *arr.shape)
        ds.create_tensor("x", htype="sequence")
        for _ in range(5):
            ds.x.append([arr] * 3)
        assert ds.x.shape == expected.shape

        def _check():
            np.testing.assert_array_equal(ds.x.numpy(), expected)

        ds.x += 3
        expected += 3
        _check()
        ds.x[:3] *= 2
        expected[:3] *= 2
        _check()
        ds.x[0][2][1] *= 7
        expected[0][2][1] *= 7
        _check()
        ds.x[1][:2][0] = np.ones((2, 7, 9), np.int32) * 13
        expected[1][:2][0] = np.ones((2, 7, 9), np.int32) * 13
        _check()
        expected[:, :] = np.ones((2, 7, 9), np.int32) * 17
        ds.x[:, :] = np.ones((2, 7, 9), np.int32) * 17
        _check()
        expected[:, 1:] = np.ones((2, 2, 7, 9), np.int32) - 9
        ds.x[:, 1:] = np.ones((2, 2, 7, 9), np.int32) - 9
        _check()
        expected[:] = np.zeros_like(expected) * 13
        ds.x[:] = expected
        _check()
        ds.x[:] = expected.reshape(1, 1, 1, 1, *expected.shape)
        _check()


@pytest.mark.parametrize("shape", [(13, 17, 3), (1007, 3001, 3)])
@pytest.mark.slow
def test_sequence_htype_with_deeplake_read(local_ds, shape, compressed_image_paths):
    ds = local_ds
    imgs = list(map(deeplake.read, compressed_image_paths["jpeg"][:3]))
    new_imgs = list(map(deeplake.read, compressed_image_paths["jpeg"][3:6]))
    arrs = np.random.randint(0, 256, (5, *shape), dtype=np.uint8)
    with ds:
        ds.create_tensor("x", htype="sequence[image]", sample_compression="png")
        for i in range(5):
            if i % 2:
                ds.x.append(imgs)
            else:
                ds.x.append(arrs)
    ds.x[0][1] = new_imgs[1]
    if len(new_imgs[1].shape) == 2:
        np.testing.assert_array_equal(ds.x[0][1].numpy().squeeze(), new_imgs[1].array)
    else:
        np.testing.assert_array_equal(ds.x[0][1].numpy(), new_imgs[1].array)
    ds.x[1] = new_imgs
    for t, img in zip(ds.x[1], new_imgs):
        if len(img.shape) == 2:
            np.testing.assert_array_equal(t.numpy().squeeze(), img.array)
        else:
            np.testing.assert_array_equal(t.numpy(), img.array)


def test_byte_positions_encoder_update_bug(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("abc")
        for i in range(11):
            ds.abc.append(np.ones((1, 1)))
        ds.abc[10] = np.ones((5, 5))
        ds.abc[0] = np.ones((2, 2))
    assert ds.abc[10].numpy().shape == (5, 5)
    assert ds.abc[0].numpy().shape == (2, 2)
    for i in range(1, 10):
        assert ds.abc[i].numpy().shape == (1, 1)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
        {"sample_compression": "png"},
        {"chunk_compression": "png"},
    ],
)
@pytest.mark.parametrize("htype", ["generic", "sequence"])
def test_update_partial(memory_ds, htype, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype=htype, **args)
        ds.x.append(np.ones((10, 10, 3), dtype=np.uint8))
        ds.x[0][0:2, 0:3, :1] = np.zeros((2, 3, 1), dtype=np.uint8)
    assert ds.x[0].shape[:3] == (10, 10, 3)
    arr = ds.x[0].numpy()
    exp = np.ones((10, 10, 3), dtype=np.uint8)
    exp[0:2, 0:3, 0] *= 0
    np.testing.assert_array_equal(arr.reshape(-1), exp.reshape(-1))
    with ds:
        ds.x[0][1] += 1
        ds.x[0][1] *= 3
    exp[1] += 1
    exp[1] *= 3
    arr = ds.x[0].numpy()
    np.testing.assert_array_equal(arr.reshape(-1), exp.reshape(-1))


@pytest.mark.slow
def test_ds_update_image(local_ds, cat_path, dog_path):
    with local_ds as ds:
        ds.create_tensor("images_sc", htype="image", sample_compression="png")
        ds.create_tensor("images_cc", htype="image", chunk_compression="png")
        ds.create_tensor("images", htype="image", sample_compression=None)

    cat = deeplake.read(cat_path)
    dog = deeplake.read(dog_path)
    samples = ([cat] + [dog] * 2) * 2

    with ds:
        ds.images_sc.extend(samples)
        ds.images_cc.extend(samples)
        ds.images.extend(samples)

    ds[1].update({"images_sc": cat, "images_cc": cat, "images": cat})

    with pytest.raises(SampleUpdateError):
        ds[2].update(
            {"images_sc": cat, "images_cc": cat, "images": deeplake.read("bad_sample")}
        )

    np.testing.assert_array_equal(
        [ds[1].images_sc.numpy(), ds[1].images_cc.numpy(), ds[1].images.numpy()],
        [cat.array] * 3,
    )
    np.testing.assert_array_equal(
        [ds[2].images_sc.numpy(), ds[2].images_cc.numpy(), ds[2].images.numpy()],
        [dog.array] * 3,
    )

    ds[:4].update({"images_sc": [cat] * 4, "images_cc": [cat] * 4, "images": [cat] * 4})

    np.testing.assert_array_equal(ds[:4].images_sc.numpy(), [cat.array] * 4)
    np.testing.assert_array_equal(ds[:4].images_cc.numpy(), [cat.array] * 4)
    np.testing.assert_array_equal(ds[:4].images.numpy(), [cat.array] * 4)

    with pytest.raises(SampleUpdateError):
        ds[:6].update(
            {
                "images_sc": [cat] * 6,
                "images_cc": [cat] * 6,
                "images": [cat] * 5 + [deeplake.read("bad_sample")],
            }
        )

    np.testing.assert_array_equal(ds[:4].images_sc.numpy(), [cat.array] * 4)
    np.testing.assert_array_equal(ds[:4].images_cc.numpy(), [cat.array] * 4)
    np.testing.assert_array_equal(ds[:4].images.numpy(), [cat.array] * 4)
    np.testing.assert_array_equal(ds[4:].images_sc.numpy(), [dog.array] * 2)
    np.testing.assert_array_equal(ds[4:].images_cc.numpy(), [dog.array] * 2)
    np.testing.assert_array_equal(ds[4:].images.numpy(), [dog.array] * 2)

    ds[:6].update({"images_sc": [cat] * 6, "images_cc": [cat] * 6, "images": [cat] * 6})

    np.testing.assert_array_equal(ds[:6].images_sc.numpy(), [cat.array] * 6)
    np.testing.assert_array_equal(ds[:6].images_cc.numpy(), [cat.array] * 6)
    np.testing.assert_array_equal(ds[:6].images.numpy(), [cat.array] * 6)


def test_ds_update_generic(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")

    ds.abc.extend(list(range(10)))
    ds.xyz.extend(list(range(10)))

    ds[0].update({"abc": 1, "xyz": 1})
    ds[2:5].update({"abc": [1] * 3, "xyz": [1] * 3})

    np.testing.assert_array_equal(ds.abc[:5].numpy().flatten(), [1] * 5)
    np.testing.assert_array_equal(ds.xyz[:5].numpy().flatten(), [1] * 5)

    with pytest.raises(SampleUpdateError):
        ds[5:].update({"abc": [1] * 5, "xyz": [1] * 4 + ["hello"]})

    np.testing.assert_array_equal(
        ds.abc.numpy().flatten(), [1] * 5 + list(range(5, 10))
    )
    np.testing.assert_array_equal(
        ds.xyz.numpy().flatten(), [1] * 5 + list(range(5, 10))
    )

    ds[5:].update({"abc": [1] * 5, "xyz": [1] * 5})
    np.testing.assert_array_equal(ds.abc.numpy().flatten(), [1] * 10)
    np.testing.assert_array_equal(ds.xyz.numpy().flatten(), [1] * 10)


@pytest.mark.parametrize(("sc", "cc"), [("lz4", None), (None, "lz4"), (None, None)])
def test_ds_update_text_like(local_ds, sc, cc):
    with local_ds as ds:
        ds.create_tensor(
            "text", htype="text", sample_compression=sc, chunk_compression=cc
        )
        ds.create_tensor(
            "list", htype="list", sample_compression=sc, chunk_compression=cc
        )
        ds.create_tensor(
            "json", htype="json", sample_compression=sc, chunk_compression=cc
        )

        text_samples = (["hello"] + ["world"] * 2) * 2
        t = "hello"
        ds.text.extend(text_samples)

        list_samples = ([[1, 2, 3]] + [[4, 5, 6]] * 2) * 2
        l = [1, 2, 3]
        ds.list.extend(list_samples)

        json_samples = ([{"a": 1}] + [{"b": 2, "c": 3}] * 2) * 2
        j = {"a": 1}
        ds.json.extend(json_samples)

    ds[1].update({"text": t, "list": l, "json": j})
    assert ds[1].text.data()["value"] == t
    assert ds[1].list.data()["value"] == l
    assert ds[1].json.data()["value"] == j

    ds[:3].update({"text": [t] * 3, "list": [l] * 3, "json": [j] * 3})
    assert ds[:3].text.data()["value"] == [t] * 3
    assert ds[:3].list.data()["value"] == [l] * 3
    assert ds[:3].json.data()["value"] == [j] * 3

    with pytest.raises(SampleUpdateError):
        ds[3:].update(
            {
                "text": [t] * 3,
                "list": [l] * 3,
                "json": [j] * 2 + [deeplake.read("bad_sample")],
            }
        )

    assert ds[3:].text.data()["value"] == text_samples[3:]
    assert ds[3:].list.data()["value"] == list_samples[3:]
    assert ds[3:].json.data()["value"] == json_samples[3:]

    ds[3:].update({"text": [t] * 3, "list": [l] * 3, "json": [j] * 3})
    assert ds.text.data()["value"] == [t] * 6
    assert ds.list.data()["value"] == [l] * 6
    assert ds.json.data()["value"] == [j] * 6


@pytest.mark.slow
def test_ds_update_sequence(local_ds, cat_path, dog_path):
    with local_ds as ds:
        ds.create_tensor("seq", htype="sequence")
        ds.create_tensor("seq_image", htype="sequence[image]", sample_compression="png")

        seq_samples = [[1, 2, 3], [4, 5, 6], [4, 5, 6]] * 2
        ds.seq.extend(seq_samples)

        dog = deeplake.read(dog_path)
        cat = deeplake.read(cat_path)
        seq_image_samples = [[cat, cat], [dog, dog], [dog, dog]] * 2
        ds.seq_image.extend(seq_image_samples)

    ds[1].update({"seq": [1, 2, 3], "seq_image": [cat, cat]})
    np.testing.assert_array_equal(ds[1].seq.numpy(), [[1], [2], [3]])
    assert ds[1].seq_image.shape == (2, 900, 900, 3)

    ds[:3].update({"seq": [[1, 2, 3]] * 3, "seq_image": [[cat, cat]] * 3})
    np.testing.assert_array_equal(ds[:3].seq.numpy(), [[[1], [2], [3]]] * 3)
    assert ds[:3].seq_image.shape == (3, 2, 900, 900, 3)

    with pytest.raises(SampleUpdateError):
        ds[3:].update(
            {"seq": [[1, 2, 3]] * 3, "seq_image": [[cat, cat], [cat, cat], [dog]]}
        )

    np.testing.assert_array_equal(
        ds[3:].seq.numpy(), np.array(seq_samples[3:]).reshape(3, 3, 1)
    )
    assert ds[3].seq_image.shape == (2, 900, 900, 3)
    assert ds[4:].seq_image.shape == (2, 2, 323, 480, 3)

    ds[3:].update({"seq": [[1, 2, 3]] * 3, "seq_image": [[cat, cat]] * 3})
    np.testing.assert_array_equal(ds.seq.numpy(), [[[1], [2], [3]]] * 6)
    assert ds.seq_image.shape == (6, 2, 900, 900, 3)


def test_ds_update_link(local_ds, cat_path, dog_path):
    with local_ds as ds:
        ds.create_tensor("images1", htype="link[image]", sample_compression="png")
        ds.create_tensor("images2", htype="link[image]", sample_compression="png")

        dog = deeplake.link(dog_path)
        cat = deeplake.link(cat_path)

        ds.images1.extend([cat] * 6)
        ds.images2.extend([dog] * 6)

        ds[0].update({"images1": dog, "images2": cat})

        assert ds[0].images1.shape == (323, 480, 3)
        assert ds[0].images2.shape == (900, 900, 3)

        ds[:3].update({"images1": [dog] * 3, "images2": [cat] * 3})
        assert ds[:3].images1.shape == (3, 323, 480, 3)
        assert ds[:3].images2.shape == (3, 900, 900, 3)

        with pytest.raises(SampleUpdateError):
            ds[3:].update(
                {
                    "images1": [dog] * 3,
                    "images2": [cat] * 2 + [deeplake.link("bad_sample")],
                }
            )

        assert ds[3:].images1.shape == (3, 900, 900, 3)
        assert ds[3:].images2.shape == (3, 323, 480, 3)

        ds[3:].update({"images1": [dog] * 3, "images2": [cat] * 3})

        assert ds.images1.shape == (6, 323, 480, 3)
        assert ds.images2.shape == (6, 900, 900, 3)


def test_ds_update_polygon(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc", htype="polygon", chunk_compression="lz4")
        ds.create_tensor("xyz", htype="polygon", chunk_compression="lz4")

        abc_samples = np.ones((6, 3, 3, 2))
        xyz_samples = np.ones((6, 2, 2, 2))

        ds.abc.extend(abc_samples)
        ds.xyz.extend(xyz_samples)

    ds[0].update({"abc": np.ones((2, 2, 2)), "xyz": np.ones((3, 3, 2))})
    assert ds[0].abc.shape == (2, 2, 2)
    assert ds[0].xyz.shape == (3, 3, 2)

    ds[:3].update({"abc": [np.ones((2, 2, 2))] * 3, "xyz": [np.ones((3, 3, 2))] * 3})
    assert ds[:3].abc.shape == (3, 2, 2, 2)
    assert ds[:3].xyz.shape == (3, 3, 3, 2)

    with pytest.raises(SampleUpdateError):
        ds[3:].update(
            {
                "abc": [np.ones((2, 2, 2))] * 3,
                "xyz": [np.ones((3, 3, 2))] * 2 + [np.ones((3, 2))],
            }
        )

    assert ds[3:].abc.shape == (3, 3, 3, 2)
    assert ds[3:].xyz.shape == (3, 2, 2, 2)

    ds[3:].update({"abc": [np.ones((2, 2, 2))] * 3, "xyz": [np.ones((3, 3, 2))] * 3})
    assert ds.abc.shape == (6, 2, 2, 2)
    assert ds.xyz.shape == (6, 3, 3, 2)


@pytest.mark.slow
def test_ds_update_tiles(local_ds, cat_path, dog_path):
    with local_ds as ds:
        ds.create_tensor(
            "images1", htype="image", sample_compression="jpg", tiling_threshold=1 * KB
        )
        ds.create_tensor(
            "images2", htype="image", sample_compression="jpg", tiling_threshold=1 * KB
        )

        cat = deeplake.read(cat_path)
        dog = deeplake.read(dog_path)

        ds.images1.extend([cat] * 6)
        ds.images2.extend([dog] * 6)

    ds[0].update({"images1": dog, "images2": cat})
    assert ds[0].images1.shape == (323, 480, 3)
    assert ds[0].images2.shape == (900, 900, 3)

    ds[:3].update({"images1": [dog] * 3, "images2": [cat] * 3})
    assert ds[:3].images1.shape == (3, 323, 480, 3)
    assert ds[:3].images2.shape == (3, 900, 900, 3)

    with pytest.raises(SampleUpdateError):
        ds[3:].update(
            {
                "images1": [dog] * 3,
                "images2": [cat] * 2 + [deeplake.read("bad_sample")],
            }
        )

    assert ds[3:].images1.shape == (3, 900, 900, 3)
    assert ds[3:].images2.shape == (3, 323, 480, 3)

    ds[3:].update({"images1": [dog] * 3, "images2": [cat] * 3})
    assert ds.images1.shape == (6, 323, 480, 3)
    assert ds.images2.shape == (6, 900, 900, 3)


def test_update_bug(local_path):
    with deeplake.empty(local_path, overwrite=True) as ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2, 3, 4, 5])

    with pytest.raises(SampleUpdateError):
        ds.abc[4] = "abcd"

    ds.abc[4] = 1

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3], [4], [1]])

    ds = deeplake.load(local_path)

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3], [4], [1]])


def test_update_vc_bug(local_path):
    with deeplake.empty(local_path, overwrite=True) as ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2, 3, 4, 5])

    ds.commit()

    with pytest.raises(SampleUpdateError):
        ds.abc[4] = "abcd"

    ds.abc[4] = 1

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3], [4], [1]])

    ds = deeplake.load(local_path)

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3], [4], [1]])


def test_setattr_update(local_path):
    with deeplake.empty(local_path, overwrite=True) as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz/t1")
        ds.abc.extend([1, 2, 3, 4, 5])
        ds.xyz.t1.extend([1, 2, 3, 4, 5])

    ds[3].abc = 10

    np.testing.assert_array_equal(ds.abc.numpy(), [[1], [2], [3], [10], [5]])

    with pytest.raises(TypeError):
        ds[3] = 10

    with pytest.raises(TypeError):
        ds[3].xyz = 10
