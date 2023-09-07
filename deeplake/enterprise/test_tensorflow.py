import pickle
import deeplake
import numpy as np
import pytest
from deeplake.util.exceptions import EmptyTensorError, TensorDoesNotExistError

from deeplake.util.remove_cache import get_base_storage
from deeplake.core.index.index import IndexEntry
from deeplake.tests.common import requires_tensorflow, requires_libdeeplake
from deeplake.core.dataset import Dataset
from deeplake.core.storage import MemoryProvider, GCSProvider
from deeplake.constants import KB

from PIL import Image  # type: ignore

from deeplake.integrations.tf.common import default_collate

# ensure tests have multiple chunks without a ton of data
TF_TESTS_MAX_CHUNK_SIZE = 5 * KB


def double(sample):
    return sample * 2


def to_tuple(sample, t1, t2):
    return sample[t1], sample[t2]


def identity_collate(batch):
    return batch


def reorder_collate(batch):
    x = [((x["a"], x["b"]), x["c"]) for x in batch]
    return default_collate(x)


def dict_to_list(sample):
    return [sample["a"], sample["b"], sample["c"]]


def my_transform_collate(batch):
    x = [(c, a, b) for a, b, c in batch]
    return default_collate(x)


def index_transform(sample):
    return sample["index"], sample["xyz"]


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
@pytest.mark.skip("causing lockups")
def test_tensorflow_small(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("image", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.commit()
        ds.create_tensor("image2", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = ds.dataloader()
        return
    dl = ds.dataloader().batch(1).tensorflow(num_workers=2)

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    sub_ds = ds[5:]
    sub_dl = sub_ds.dataloader().tensorflow(num_workers=0)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 6 + i, 6 + i))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_ds2 = ds[8:12]
    sub_dl2 = sub_ds2.dataloader().tensorflow(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl2):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (8 + i) * np.ones((1, 9 + i, 9 + i))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
            )

    sub_ds3 = ds[:5]
    sub_dl3 = sub_ds3.dataloader().tensorflow(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl3):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (i) * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
            )


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
@pytest.mark.skip("causing lockups")
def test_tensorflow_transform(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("image", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.checkout("alt", create=True)
        ds.create_tensor("image2", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = ds.dataloader()
        return

    dl = (
        ds.dataloader()
        .batch(1)
        .transform(to_tuple, t1="image", t2="image2")
        .tensorflow(num_workers=2)
    )

    for _ in range(2):
        for i, batch in enumerate(dl):
            actual_image = batch[0].numpy()
            expected_image = i * np.ones((1, i + 1, i + 1))
            actual_image2 = batch[1].numpy()
            expected_image2 = i * np.ones((1, 12, 12))
            np.testing.assert_array_equal(actual_image, expected_image)
            np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_transform_dict(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("image", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))
        ds.create_tensor("image3", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
        ds.image3.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = ds.dataloader()
        return

    dl = ds.dataloader().transform({"image": double, "image2": None}).tensorflow()

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            assert set(batch.keys()) == {"image", "image2"}
            np.testing.assert_array_equal(
                batch["image"].numpy(), 2 * i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    for _ in range(2):
        for i, (image, image2) in enumerate(dl):
            np.testing.assert_array_equal(
                image.numpy(), 2 * i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(image2.numpy(), i * np.ones((1, 12, 12)))


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_with_compression(local_auth_ds: Dataset):
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with local_auth_ds as ds:
        images = ds.create_tensor(
            "images",
            htype="image",
            sample_compression="png",
            max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE,
        )
        labels = ds.create_tensor(
            "labels", htype="class_label", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE
        )

        assert images.meta.sample_compression == "png"

        images.extend(np.ones((16, 12, 12, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="uint32"))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = ds.dataloader()
        return

    dl = ds.dataloader().tensorflow(num_workers=0)

    for _ in range(2):
        for batch in dl:
            X = batch["images"].numpy()
            T = batch["labels"].numpy()
            assert X.shape == (1, 12, 12, 3)
            assert T.shape == (1, 1)


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_custom_tensor_order(local_auth_ds):
    with local_auth_ds as ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t, max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
            ds[t].extend(np.random.random((3, 4, 5)))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = ds.dataloader()
        return

    with pytest.raises(TensorDoesNotExistError):
        dl = ds.dataloader().tensorflow(tensors=["c", "d", "e"])

    dl = ds.dataloader().tensorflow(tensors=["c", "d", "a"], return_index=False)

    for i, batch in enumerate(dl):
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        assert "b" not in batch
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
        np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
        np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])
        batch = pickle.loads(pickle.dumps(batch))
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
        np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
        np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
@pytest.mark.skip("causing lockups")
def test_readonly_with_two_workers(local_auth_ds):
    local_auth_ds.create_tensor("images", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
    local_auth_ds.create_tensor("labels", max_chunk_size=TF_TESTS_MAX_CHUNK_SIZE)
    local_auth_ds.images.extend(np.ones((10, 12, 12)))
    local_auth_ds.labels.extend(np.ones(10))

    base_storage = get_base_storage(local_auth_ds.storage)
    base_storage.flush()
    base_storage.enable_readonly()
    ds = Dataset(
        storage=local_auth_ds.storage,
        token=local_auth_ds.token,
        read_only=True,
        verbose=False,
    )

    ptds = ds.dataloader().tensorflow(num_workers=2)
    # no need to check input, only care that readonly works
    for _ in ptds:
        pass


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_corrupt_dataset():
    raise NotImplementedError


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_tensorflow_local_cache():
    raise NotImplementedError


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_groups(local_auth_ds, compressed_image_paths):
    img1 = deeplake.read(compressed_image_paths["jpeg"][0])
    img2 = deeplake.read(compressed_image_paths["png"][0])
    with local_auth_ds as ds:
        ds.create_tensor("images/jpegs/cats", htype="image", sample_compression="jpeg")
        ds.create_tensor("images/pngs/flowers", htype="image", sample_compression="png")
        for _ in range(10):
            ds.images.jpegs.cats.append(img1)
            ds.images.pngs.flowers.append(img2)

    another_ds = deeplake.dataset(
        ds.path,
        token=ds.token,
    )
    dl = another_ds.dataloader().tensorflow(return_index=False)
    for i, (cat, flower) in enumerate(dl):
        assert cat[0].shape == another_ds.images.jpegs.cats[i].numpy().shape
        assert flower[0].shape == another_ds.images.pngs.flowers[i].numpy().shape

    dl = another_ds.images.dataloader().tensorflow(return_index=False)
    for sample in dl:
        cat = sample["images/jpegs/cats"]
        flower = sample["images/pngs/flowers"]
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_string_tensors(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("strings", htype="text")
        ds.strings.extend([f"string{idx}" for idx in range(5)])

    ptds = ds.dataloader().tensorflow(collate_fn=identity_collate)
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch[0]["strings"], f"string{idx}")


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_tensorflow_large():
    raise NotImplementedError


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize(
    "index",
    [
        slice(2, 7),
        slice(3, 10, 2),
        slice(None, 10),
        slice(None, None, -1),
        slice(None, None, -2),
        [2, 3, 4],
        [2, 4, 6, 8],
        [2, 2, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 9],
        [4, 3, 2, 1],
    ],
)
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_view(local_auth_ds, index):
    arr_list_1 = [np.random.randn(15, 15, i) for i in range(10)]
    arr_list_2 = [np.random.randn(40, 15, 4, i) for i in range(10)]
    label_list = list(range(10))

    with local_auth_ds as ds:
        ds.create_tensor("img1")
        ds.create_tensor("img2")
        ds.create_tensor("label")
        ds.img1.extend(arr_list_1)
        ds.img2.extend(arr_list_2)
        ds.label.extend(label_list)

    ptds = ds[index].dataloader().tensorflow()
    idxs = list(IndexEntry(index).indices(len(ds)))
    for idx, batch in enumerate(ptds):
        idx = idxs[idx]
        np.testing.assert_array_equal(batch["img1"][0], arr_list_1[idx])
        np.testing.assert_array_equal(batch["img2"][0], arr_list_2[idx])
        np.testing.assert_array_equal(batch["label"][0], idx)


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_collate(local_auth_ds, shuffle):
    with local_auth_ds as ds:
        ds.create_tensor("a")
        ds.create_tensor("b")
        ds.create_tensor("c")
        for _ in range(100):
            ds.a.append(0)
            ds.b.append(1)
            ds.c.append(2)

    ptds = ds.dataloader().batch(4).tensorflow(collate_fn=reorder_collate)
    if shuffle:
        ptds = ptds.shuffle()
    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 2
        np.testing.assert_array_equal(batch[0][0], np.array([0, 0, 0, 0]).reshape(4, 1))
        np.testing.assert_array_equal(batch[0][1], np.array([1, 1, 1, 1]).reshape(4, 1))
        np.testing.assert_array_equal(batch[1], np.array([2, 2, 2, 2]).reshape(4, 1))


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_transform_collate(local_auth_ds, shuffle):
    with local_auth_ds as ds:
        local_auth_ds.create_tensor("a")
        local_auth_ds.create_tensor("b")
        local_auth_ds.create_tensor("c")
        for _ in range(100):
            local_auth_ds.a.append(0 * np.ones((300, 300)))
            local_auth_ds.b.append(1 * np.ones((300, 300)))
            local_auth_ds.c.append(2 * np.ones((300, 300)))

    ptds = (
        local_auth_ds.dataloader()
        .batch(4)
        .tensorflow(
            collate_fn=my_transform_collate,
        )
        .transform(dict_to_list)
    )
    if shuffle:
        ptds = ptds.shuffle()
    for batch in ptds:
        assert len(batch) == 3
        for i in range(2):
            assert len(batch[i]) == 4
        np.testing.assert_array_equal(batch[0], 2 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[1], 0 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[2], 1 * np.ones((4, 300, 300)))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_tensorflow_ddp():
    raise NotImplementedError


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize("compression", [None, "jpeg"])
@pytest.mark.slow
@pytest.mark.flaky
def test_tensorflow_decode(local_auth_ds, compressed_image_paths, compression):
    with local_auth_ds:
        local_auth_ds.create_tensor("image", sample_compression=compression)
        local_auth_ds.image.extend(
            np.array([i * np.ones((10, 10, 3), dtype=np.uint8) for i in range(5)])
        )
        local_auth_ds.image.extend(
            [deeplake.read(compressed_image_paths["jpeg"][0])] * 5
        )
    if isinstance(
        get_base_storage(local_auth_ds.storage), (MemoryProvider, GCSProvider)
    ):
        with pytest.raises(ValueError):
            dl = local_auth_ds.dataloader()
        return

    ptds = local_auth_ds.dataloader().tensorflow(
        collate_fn=identity_collate, decode_method={"image": "tobytes"}
    )

    for i, batch in enumerate(ptds):
        image = batch[0]["image"]
        assert isinstance(image, bytes)
        if i < 5 and not compression:
            np.testing.assert_array_equal(
                np.frombuffer(image, dtype=np.uint8).reshape(10, 10, 3),
                i * np.ones((10, 10, 3), dtype=np.uint8),
            )
        elif i >= 5 and compression:
            with open(compressed_image_paths["jpeg"][0], "rb") as f:
                assert f.read() == image

    if compression:
        ptds = local_auth_ds.dataloader().numpy(decode_method={"image": "pil"})
        for i, batch in enumerate(ptds):
            image = batch[0]["image"]
            assert isinstance(image, Image.Image)
            if i < 5:
                np.testing.assert_array_equal(
                    np.array(image), i * np.ones((10, 10, 3), dtype=np.uint8)
                )
            elif i >= 5:
                with Image.open(compressed_image_paths["jpeg"][0]) as f:
                    np.testing.assert_array_equal(np.array(f), np.array(image))


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.flaky
def test_rename(local_auth_ds):
    tensor_name = "red/green"
    with local_auth_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("blue/green")
        ds.abc.append([1, 2, 3])
        ds.rename_tensor("abc", "xyz")
        ds.rename_group("blue", "red")
        ds[tensor_name].append([1, 2, 3, 4])
    loader = ds.dataloader().tensorflow(return_index=False)
    for sample in loader:
        assert set(sample.keys()) == {"xyz", tensor_name}
        np.testing.assert_array_equal(np.array(sample["xyz"]), np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(
            np.array(sample[tensor_name]), np.array([[1, 2, 3, 4]])
        )


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize(
    "num_workers",
    [
        0,
        pytest.param(2, marks=pytest.mark.skip("causing lockups")),
    ],
)
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes(local_auth_ds, num_workers):
    shuffle = False
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = (
        ds.dataloader().batch(4).tensorflow(num_workers=num_workers, return_index=True)
    )
    if shuffle:
        ptds = ptds.shuffle()

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(batch["index"][i], batch["xyz"][i][0, 0])


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize(
    "num_workers",
    [
        0,
        pytest.param(2, marks=pytest.mark.skip("causing lockups")),
    ],
)
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes_transform(local_auth_ds, num_workers):
    shuffle = False
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform(index_transform)
        .tensorflow(num_workers=num_workers, return_index=True)
    )
    if shuffle:
        ptds = ptds.shuffle()

    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 4
        assert len(batch[1]) == 4

        for i in range(4):
            np.testing.assert_array_equal(batch[0][i], batch[1][i][0, 0])


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes_transform_dict(local_auth_ds, num_workers):
    shuffle = False
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform({"xyz": double, "index": None})
        .tensorflow(num_workers=num_workers, return_index=True)
    )
    if shuffle:
        ptds = ptds.shuffle()

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(2 * batch["index"][i], batch["xyz"][i][0, 0])

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform({"xyz": double})
        .tensorflow(num_workers=num_workers, return_index=True)
    )
    if shuffle:
        ptds = ptds.shuffle()

    for batch in ptds:
        assert batch.keys() == {"xyz"}


@requires_tensorflow
@requires_libdeeplake
@pytest.mark.parametrize(
    "num_workers", [0, pytest.param(2, marks=pytest.mark.skip("causing lockups"))]
)
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes_tensors(local_auth_ds, num_workers):
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    with pytest.raises(ValueError):
        (
            ds.dataloader()
            .batch(4)
            .tensorflow(
                num_workers=num_workers, return_index=True, tensors=["xyz", "index"]
            )
        )

    ptds = (
        ds.dataloader()
        .batch(4)
        .tensorflow(num_workers=num_workers, return_index=True, tensors=["xyz"])
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}


@requires_libdeeplake
@requires_tensorflow
@pytest.mark.flaky
def test_uneven_iteration(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(5)))
        ds.y.extend(list(range(10)))
    ptds = ds.dataloader().tensorflow()
    for i, batch in enumerate(ptds):
        x, y = np.array(batch["x"][0]), np.array(batch["y"][0])
        np.testing.assert_equal(x, i)
        np.testing.assert_equal(y, i)


@requires_libdeeplake
@requires_tensorflow
@pytest.mark.flaky
def test_tensorflow_error_handling(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(5)))

    ptds = ds.dataloader().tensorflow()
    with pytest.raises(EmptyTensorError):
        for _ in ptds:
            continue

    ptds = ds.dataloader().tensorflow(tensors=["x", "y"])
    with pytest.raises(EmptyTensorError):
        for _ in ptds:
            continue

    ptds = ds.dataloader().tensorflow(tensors=["x"])
    for _ in ptds:
        continue
