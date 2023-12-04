import pickle
import deeplake
import numpy as np
import pytest
from functools import partial
from deeplake.util.exceptions import EmptyTensorError, TensorDoesNotExistError

from deeplake.util.remove_cache import get_base_storage
from deeplake.core.index.index import IndexEntry
from deeplake.tests.common import (
    requires_torch,
    requires_libdeeplake,
    convert_data_according_to_torch_version,
)
from deeplake.core.dataset import Dataset
from deeplake.constants import KB

from PIL import Image  # type: ignore

try:
    from torch.utils.data._utils.collate import default_collate
except ImportError:
    pass

from unittest.mock import patch


# ensure tests have multiple chunks without a ton of data
PYTORCH_TESTS_MAX_CHUNK_SIZE = 5 * KB


def double(sample):
    return sample * 2


def identity(batch):
    return batch


def identity_collate(batch):
    return batch


def to_tuple(sample, t1, t2):
    return sample[t1], sample[t2]


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


def dummy_init_fn(arg):
    return f"function called with arg {arg}"


@requires_libdeeplake
def test_setting_woker_init_function(local_auth_ds):
    dl = local_auth_ds.dataloader().pytorch()

    assert dl.worker_init_fn == None
    dl.worker_init_fn = partial(dummy_init_fn, 1024)
    assert dl.worker_init_fn() == "function called with arg 1024"


@requires_torch
@requires_libdeeplake
def test_offset_ds_iteration(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("abc", htype="generic", dtype="uint16")
        ds.abc.extend([i for i in range(10)])

    dl = (
        local_auth_ds.dataloader()
        .offset(4)
        .transform(identity)
        .pytorch(collate_fn=identity)
    )

    idx_table = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    for i, item in enumerate(dl):
        assert idx_table[i] == item[0]["index"].astype(int)


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize(
    "ds",
    [
        pytest.param(
            "hub_cloud_ds",
            marks=[pytest.mark.slow, pytest.mark.skip("Causing lockups")],
        ),
        "local_auth_ds",
    ],
    indirect=True,
)
@pytest.mark.flaky
@pytest.mark.skip("causing lockups")
def test_pytorch_small(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.commit()
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))
    dl = ds.dataloader().batch(1).pytorch(num_workers=2)

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
    sub_dl = sub_ds.dataloader().pytorch(num_workers=0)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 6 + i, 6 + i))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_ds2 = ds[8:12]
    sub_dl2 = sub_ds2.dataloader().pytorch(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl2):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (8 + i) * np.ones((1, 9 + i, 9 + i))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
            )

    sub_ds3 = ds[:5]
    sub_dl3 = sub_ds3.dataloader().pytorch(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl3):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (i) * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
            )


@requires_torch
@requires_libdeeplake
@pytest.mark.flaky(retry_count=3)
@pytest.mark.slow
@pytest.mark.timeout(10)
@pytest.mark.skip("causing lockups")
def test_pytorch_transform(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.checkout("alt", create=True)
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    dl = (
        ds.dataloader()
        .batch(1)
        .transform(to_tuple, t1="image", t2="image2")
        .pytorch(num_workers=2, collate_fn=identity_collate)
    )

    for _ in range(2):
        for i, batch in enumerate(dl):
            actual_image, actual_image2 = batch[0]
            expected_image = i * np.ones((i + 1, i + 1))
            expected_image2 = i * np.ones((12, 12))
            np.testing.assert_array_equal(actual_image, expected_image)
            np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_libdeeplake
@pytest.mark.flaky
def test_inequal_tensors_dataloader_length(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("images")
        ds.create_tensor("label")
        ds.images.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))

    ld = local_auth_ds.dataloader().batch(1).pytorch()
    assert len(ld) == 0
    ld1 = local_auth_ds.dataloader().batch(2).pytorch(tensors=["images"])
    assert len(ld1) == 8


@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
@pytest.mark.slow
def test_pytorch_transform_dict(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))
        ds.create_tensor("image3", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image3.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    dl = ds.dataloader().transform({"image": double, "image2": None}).pytorch()

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


@pytest.mark.slow
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
def test_pytorch_with_compression(local_auth_ds: Dataset):
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with local_auth_ds as ds:
        images = ds.create_tensor(
            "images",
            htype="image",
            sample_compression="png",
            max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE,
        )
        labels = ds.create_tensor(
            "labels", htype="class_label", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE
        )

        assert images.meta.sample_compression == "png"

        images.extend(np.ones((16, 12, 12, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="uint32"))

    dl = ds.dataloader().pytorch(num_workers=0)

    for _ in range(2):
        for batch in dl:
            X = batch["images"].numpy()
            T = batch["labels"].numpy()
            assert X.shape == (1, 12, 12, 3)
            assert T.shape == (1, 1)


@pytest.mark.slow
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
def test_custom_tensor_order(local_auth_ds):
    with local_auth_ds as ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t, max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
            ds[t].extend(np.random.random((3, 4, 5)))

    with pytest.raises(TensorDoesNotExistError):
        ds.dataloader().pytorch(tensors=["c", "d", "e"])

    dl = ds.dataloader().pytorch(tensors=["c", "d", "a"], return_index=False)

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


@pytest.mark.slow
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
@pytest.mark.skip("causing lockups")
@pytest.mark.timeout(10)
def test_readonly_with_two_workers(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("images", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.create_tensor("labels", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.images.extend(np.ones((10, 12, 12)))
        ds.labels.extend(np.ones(10))

    base_storage = get_base_storage(ds.storage)
    base_storage.flush()
    base_storage.enable_readonly()
    ds = Dataset(
        storage=ds.storage,
        token=ds.token,
        read_only=True,
        verbose=False,
    )

    ptds = ds.dataloader().pytorch(num_workers=2)
    # no need to check input, only care that readonly works
    for _ in ptds:
        continue


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_corrupt_dataset():
    raise NotImplementedError


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_pytorch_local_cache():
    raise NotImplementedError


@requires_torch
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
    dl = another_ds.dataloader().pytorch(return_index=False)
    for i, (cat, flower) in enumerate(dl):
        assert cat[0].shape == another_ds.images.jpegs.cats[i].numpy().shape
        assert flower[0].shape == another_ds.images.pngs.flowers[i].numpy().shape

    dl = another_ds.images.dataloader().pytorch(return_index=False)
    for sample in dl:
        cat = sample["images/jpegs/cats"]
        flower = sample["images/pngs/flowers"]
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)


@pytest.mark.slow
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
def test_string_tensors(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("strings", htype="text")
        ds.strings.extend([f"string{idx}" for idx in range(5)])

    ptds = ds.dataloader().pytorch()
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["strings"], f"string{idx}")


@pytest.mark.slow
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
def test_tag_tensors(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("tags", htype="tag")
        ds.tags.extend(
            [
                f"tag{idx}" if idx % 2 == 0 else [f"tag{idx}", f"tag{idx}"]
                for idx in range(5)
            ]
        )

    ptds = ds.pytorch(batch_size=1)
    for idx, batch in enumerate(ptds):
        if idx % 2 == 0:
            np.testing.assert_array_equal(batch["tags"], [[f"tag{idx}"]])
        else:
            np.testing.assert_array_equal(batch["tags"], [[f"tag{idx}", f"tag{idx}"]])

    ptds2 = ds.pytorch(batch_size=None)
    for idx, batch in enumerate(ptds2):
        if idx % 2 == 0:
            np.testing.assert_array_equal(batch["tags"], [f"tag{idx}"])
        else:
            np.testing.assert_array_equal(batch["tags"], [f"tag{idx}", f"tag{idx}"])


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_pytorch_large():
    raise NotImplementedError


@requires_torch
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
def test_pytorch_view(local_auth_ds, index):
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

    ptds = ds[index].dataloader().pytorch()
    idxs = list(IndexEntry(index).indices(len(ds)))
    for idx, batch in enumerate(ptds):
        idx = idxs[idx]
        np.testing.assert_array_equal(batch["img1"][0], arr_list_1[idx])
        np.testing.assert_array_equal(batch["img2"][0], arr_list_2[idx])
        np.testing.assert_array_equal(batch["label"][0], idx)


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.slow
@pytest.mark.flaky
def test_pytorch_collate(local_auth_ds, shuffle):
    with local_auth_ds as ds:
        ds.create_tensor("a")
        ds.create_tensor("b")
        ds.create_tensor("c")
        for _ in range(100):
            ds.a.append(0)
            ds.b.append(1)
            ds.c.append(2)

    ptds = ds.dataloader().batch(4).pytorch(collate_fn=reorder_collate)
    if shuffle:
        ptds = ptds.shuffle()
    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 2
        np.testing.assert_array_equal(batch[0][0], np.array([0, 0, 0, 0]).reshape(4, 1))
        np.testing.assert_array_equal(batch[0][1], np.array([1, 1, 1, 1]).reshape(4, 1))
        np.testing.assert_array_equal(batch[1], np.array([2, 2, 2, 2]).reshape(4, 1))


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.slow
@pytest.mark.flaky
def test_pytorch_transform_collate(local_auth_ds, shuffle):
    with local_auth_ds as ds:
        ds.create_tensor("a")
        ds.create_tensor("b")
        ds.create_tensor("c")
        for _ in range(100):
            ds.a.append(0 * np.ones((300, 300)))
            ds.b.append(1 * np.ones((300, 300)))
            ds.c.append(2 * np.ones((300, 300)))

    ptds = (
        ds.dataloader()
        .batch(4)
        .pytorch(
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
def test_pytorch_ddp():
    raise NotImplementedError


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize("compression", [None, "jpeg"])
@pytest.mark.slow
@pytest.mark.flaky
def test_pytorch_decode(local_auth_ds, compressed_image_paths, compression):
    with local_auth_ds as ds:
        ds.create_tensor("image", sample_compression=compression)
        ds.image.extend(
            np.array([i * np.ones((10, 10, 3), dtype=np.uint8) for i in range(5)])
        )
        ds.image.extend([deeplake.read(compressed_image_paths["jpeg"][0])] * 5)

    ptds = ds.dataloader().pytorch(decode_method={"image": "tobytes"})

    for i, batch in enumerate(ptds):
        image = convert_data_according_to_torch_version(batch["image"])
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
        ptds = ds.dataloader().numpy(decode_method={"image": "pil"})
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


@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
@pytest.mark.slow
def test_rename(local_auth_ds):
    group_name = "red/green"
    with local_auth_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("blue/green")
        ds.abc.append([1, 2, 3])
        ds.rename_tensor("abc", "xyz")
        ds.rename_group("blue", "red")
        ds[group_name].append([1, 2, 3, 4])
    loader = ds.dataloader().pytorch(return_index=False)
    for sample in loader:
        assert set(sample.keys()) == {"xyz", group_name}
        np.testing.assert_array_equal(np.array(sample["xyz"]), np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(
            np.array(sample[group_name]), np.array([[1, 2, 3, 4]])
        )


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize(
    "num_workers",
    [
        0,
        pytest.param(2, marks=pytest.mark.skip(reason="causing lockups")),
    ],
)
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes(local_auth_ds, num_workers):
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = ds.dataloader().batch(4).pytorch(num_workers=num_workers, return_index=True)
    ptds = ptds.shuffle()

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(batch["index"][i], batch["xyz"][i][0, 0])


@requires_torch
@requires_libdeeplake
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_workers",
    [
        0,
        pytest.param(2, marks=pytest.mark.skip("causing lockups")),
    ],
)
@pytest.mark.flaky
def test_indexes_transform(local_auth_ds, num_workers):
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform(index_transform)
        .pytorch(
            num_workers=num_workers, return_index=True, collate_fn=identity_collate
        )
    )

    for batch in ptds:
        assert len(batch) == 4
        assert len(batch[0]) == 2
        assert len(batch[1]) == 2


@requires_torch
@requires_libdeeplake
@pytest.mark.parametrize(
    "num_workers", [0, pytest.param(2, marks=pytest.mark.skip("causing lockups"))]
)
@pytest.mark.slow
@pytest.mark.flaky
def test_indexes_transform_dict(local_auth_ds, num_workers):
    with local_auth_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform({"xyz": double, "index": None})
        .pytorch(num_workers=num_workers, return_index=True)
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(2 * batch["index"][i], batch["xyz"][i][0, 0])

    ptds = (
        ds.dataloader()
        .batch(4)
        .transform({"xyz": double})
        .pytorch(num_workers=num_workers, return_index=True)
    )

    for batch in ptds:
        assert batch.keys() == {"xyz"}


@requires_torch
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
            .pytorch(
                num_workers=num_workers, return_index=True, tensors=["xyz", "index"]
            )
        )

    ptds = (
        ds.dataloader()
        .batch(4)
        .pytorch(num_workers=num_workers, return_index=True, tensors=["xyz"])
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}


@requires_libdeeplake
@requires_torch
@pytest.mark.flaky
@pytest.mark.slow
def test_uneven_iteration(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(5)))
        ds.y.extend(list(range(10)))
    ptds = ds.dataloader().pytorch()
    for i, batch in enumerate(ptds):
        x, y = np.array(batch["x"][0]), np.array(batch["y"][0])
        np.testing.assert_equal(x, i)
        np.testing.assert_equal(y, i)


@requires_libdeeplake
@requires_torch
@pytest.mark.slow
def test_pytorch_error_handling(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(5)))

    ptds = ds.dataloader().pytorch()
    with pytest.raises(EmptyTensorError):
        for _ in ptds:
            continue

    ptds = ds.dataloader().pytorch(tensors=["x", "y"])
    with pytest.raises(EmptyTensorError):
        for _ in ptds:
            continue

    ptds = ds.dataloader().pytorch(tensors=["x"])
    for _ in ptds:
        continue


@requires_libdeeplake
@requires_torch
def test_batch_sampler_attribute(local_auth_ds):
    ld = local_auth_ds.dataloader().pytorch()

    from torch.utils.data import BatchSampler

    assert isinstance(ld.batch_sampler, BatchSampler)
    assert ld.batch_sampler.sampler is not None


@requires_libdeeplake
@requires_torch
@pytest.mark.slow
@pytest.mark.flaky
def test_pil_decode_method(local_auth_ds):
    from indra.pytorch.exceptions import CollateExceptionWrapper  # type: ignore

    with local_auth_ds as ds:
        ds.create_tensor("x", htype="image", sample_compression="jpeg")
        ds.x.extend(np.random.randint(0, 255, (10, 10, 10, 3), np.uint8))

    ptds = ds.dataloader().pytorch(return_index=False)
    for batch in ptds:
        assert len(batch.keys()) == 1
        assert "x" in batch.keys()
        assert batch["x"].shape == (1, 10, 10, 3)

    ptds = ds.dataloader().pytorch(decode_method={"x": "pil"})
    with pytest.raises(CollateExceptionWrapper):
        for _ in ptds:
            continue

    def custom_transform(batch):
        batch["x"] = np.array(batch["x"])
        return batch

    ptds = (
        ds.dataloader()
        .pytorch(decode_method={"x": "pil"}, return_index=False)
        .transform(custom_transform)
    )
    for batch in ptds:
        assert len(batch.keys()) == 1
        assert "x" in batch.keys()
        assert batch["x"].shape == (1, 10, 10, 3)


@patch("deeplake.constants.RETURN_DUMMY_DATA_FOR_DATALOADER", True)
@requires_torch
@requires_libdeeplake
@pytest.mark.flaky
def test_pytorch_dummy_data(local_auth_ds):
    x_data = [
        np.random.randint(0, 255, (100, 100, 3), dtype="uint8"),
        np.random.randint(0, 255, (120, 120, 3), dtype="uint8"),
    ]
    y_data = [np.random.rand(100, 100, 3), np.random.rand(120, 120, 3)]
    z_data = ["hello", "world"]
    with local_auth_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.create_tensor("z")
        ds.x.extend(x_data)
        ds.y.extend(y_data)
        ds.z.extend(z_data)

    ptds = ds.dataloader()
    for i, batch in enumerate(ptds):
        x = x_data[i]
        dummy_x = batch[0]["x"]
        assert dummy_x.shape == x.shape
        assert dummy_x.dtype == x.dtype

        y = y_data[i]
        dummy_y = batch[0]["y"]
        assert dummy_y.shape == y.shape
        assert dummy_y.dtype == y.dtype

        dummy_z = batch[0]["z"]
        assert dummy_z[0] == "a"


@requires_libdeeplake
@requires_torch
@pytest.mark.flaky
@pytest.mark.slow
def test_json_data_loader(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor(
            "json",
            htype="json",
            sample_compression=None,
        )
        d = {"x": 1, "y": 2, "z": 3}
        for _ in range(10):
            ds.json.append(d)

    dl = ds.dataloader().batch(2)

    for batch in dl:
        sample1 = batch[0]["json"]
        sample2 = batch[1]["json"]

        assert sample1 == d
        assert sample2 == d


@requires_libdeeplake
@requires_torch
@pytest.mark.flaky
@pytest.mark.slow
def test_list_data_loader(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor(
            "list",
            htype="list",
            sample_compression=None,
        )
        l = [1, 2, 3]
        for _ in range(10):
            ds.list.append(l)

    dl = ds.dataloader().batch(2)

    for batch in dl:
        sample1 = batch[0]["list"]
        sample2 = batch[1]["list"]
        assert sample1.tolist() == l
        assert sample2.tolist() == l


@requires_libdeeplake
@requires_torch
@pytest.mark.flaky
@pytest.mark.slow
def test_pytorch_data_decode(local_auth_ds, cat_path):
    with local_auth_ds as ds:
        ds.create_tensor("generic")
        for i in range(10):
            ds.generic.append(i)
        ds.create_tensor("text", htype="text")
        for i in range(10):
            ds.text.append(f"hello {i}")
        ds.create_tensor("json", htype="json")
        for i in range(10):
            ds.json.append({"x": i})
        ds.create_tensor("list", htype="list")
        for i in range(10):
            ds.list.append([i, i + 1])
        ds.create_tensor("class_label", htype="class_label")
        animals = [
            "cat",
            "dog",
            "bird",
            "fish",
            "horse",
            "cow",
            "pig",
            "sheep",
            "goat",
            "chicken",
        ]
        ds.class_label.extend(animals)
        ds.create_tensor("image", htype="image", sample_compression="jpeg")
        for i in range(10):
            ds.image.append(deeplake.read(cat_path))

    decode_method = {tensor: "data" for tensor in list(ds.tensors.keys())}
    ptds = (
        ds.dataloader()
        .transform(identity)
        .pytorch(decode_method=decode_method, collate_fn=identity_collate)
    )
    for i, batch in enumerate(ptds):
        sample = batch[0]
        assert sample["text"]["value"] == f"hello {i}"
        assert sample["json"]["value"] == {"x": i}
        assert sample["list"]["value"].tolist() == [i, i + 1]
        assert sample["class_label"]["value"] == [i]
        assert sample["class_label"]["text"] == [animals[i]]
        assert sample["image"]["value"].shape == (900, 900, 3)
        assert sample["generic"]["value"] == i
