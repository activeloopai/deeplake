import hub
import numpy as np
import pickle
import pytest

from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import DatasetUnsupportedPytorch, TensorDoesNotExistError
from hub.tests.common import requires_torch
from hub.core.dataset import Dataset
from hub.core.storage.memory import MemoryProvider
from hub.constants import KB

from hub.tests.dataset_fixtures import enabled_datasets


# ensure tests have multiple chunks without a ton of data
PYTORCH_TESTS_MAX_CHUNK_SIZE = 5 * KB


def double(sample):
    return sample * 2


def to_tuple(sample):
    return sample["image"], sample["image2"]


def pytorch_small_shuffle_helper(start, end, dataloader):
    for _ in range(2):
        all_values = []
        for i, batch in enumerate(dataloader):
            value = batch["image"].numpy()[0][0][0]
            value2 = batch["image2"].numpy()[0][0][0]
            assert value == value2
            all_values.append(value)
            np.testing.assert_array_equal(
                batch["image"].numpy(), value * np.ones(batch["image"].shape)
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), value2 * np.ones((1, 12, 12))
            )

    assert set(all_values) == set(range(start, end))


@requires_torch
@enabled_datasets
def test_pytorch_small(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.commit()
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch()
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    dls = ds.pytorch(num_workers=2, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(0, 16, dls)

    sub_ds = ds[5:]

    sub_dl = sub_ds.pytorch(num_workers=0)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 6 + i, 6 + i))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_dls = sub_ds.pytorch(num_workers=0, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(5, 16, sub_dls)

    sub_ds2 = ds[8:12]

    sub_dl2 = sub_ds2.pytorch(num_workers=0, batch_size=1)

    for _ in range(2):
        for i, batch in enumerate(sub_dl2):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (8 + i) * np.ones((1, 9 + i, 9 + i))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
            )

    sub_dls2 = sub_ds2.pytorch(num_workers=2, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(8, 12, sub_dls2)

    sub_ds3 = ds[:5]

    sub_dl3 = sub_ds3.pytorch(num_workers=0, batch_size=1)

    for _ in range(2):
        for i, batch in enumerate(sub_dl3):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (i) * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
            )

    sub_dls3 = sub_ds3.pytorch(num_workers=0, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(0, 5, sub_dls3)


@requires_torch
@enabled_datasets
def test_pytorch_transform(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.checkout("alt", create=True)
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(num_workers=2, transform=to_tuple, batch_size=1)

    for _ in range(2):
        for i, batch in enumerate(dl):
            actual_image = batch[0].numpy()
            expected_image = i * np.ones((1, i + 1, i + 1))
            actual_image2 = batch[1].numpy()
            expected_image2 = i * np.ones((1, 12, 12))
            np.testing.assert_array_equal(actual_image, expected_image)
            np.testing.assert_array_equal(actual_image2, expected_image2)

    dls = ds.pytorch(num_workers=0, transform=to_tuple, batch_size=1, shuffle=True)

    for _ in range(2):
        all_values = []
        for i, batch in enumerate(dls):
            actual_image = batch[0].numpy()
            actual_image2 = batch[1].numpy()

            value = actual_image[0][0][0]
            value2 = actual_image2[0][0][0]
            assert value == value2
            all_values.append(value)

            expected_image = value * np.ones(actual_image.shape)
            expected_image2 = value * np.ones(actual_image2.shape)
            np.testing.assert_array_equal(actual_image, expected_image)
            np.testing.assert_array_equal(actual_image2, expected_image2)

        assert set(all_values) == set(range(16))


@requires_torch
@enabled_datasets
def test_pytorch_transform_dict(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))
        ds.create_tensor("image3", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(
        num_workers=2, transform={"image": double, "image2": None}, batch_size=1
    )

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


@requires_torch
@enabled_datasets
def test_pytorch_with_compression(ds: Dataset):
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with ds:
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

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(num_workers=0, batch_size=1)
    dls = ds.pytorch(num_workers=0, batch_size=1, shuffle=True)

    for dataloader in [dl, dls]:
        for _ in range(2):
            for batch in dataloader:
                X = batch["images"].numpy()
                T = batch["labels"].numpy()
                assert X.shape == (1, 12, 12, 3)
                assert T.shape == (1, 1)


@requires_torch
@enabled_datasets
def test_custom_tensor_order(ds):
    with ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t, max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
            ds[t].extend(np.random.random((3, 4, 5)))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch()
        return

    with pytest.raises(TensorDoesNotExistError):
        dl = ds.pytorch(num_workers=0, tensors=["c", "d", "e"])

    dl = ds.pytorch(num_workers=0, tensors=["c", "d", "a"])

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

    dls = ds.pytorch(num_workers=0, tensors=["c", "d", "a"])
    for i, batch in enumerate(dls):
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        assert "b" not in batch
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        batch = pickle.loads(pickle.dumps(batch))
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)


@requires_torch
def test_readonly_with_two_workers(local_ds):
    local_ds.create_tensor("images", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.create_tensor("labels", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.images.extend(np.ones((10, 12, 12)))
    local_ds.labels.extend(np.ones(10))

    base_storage = get_base_storage(local_ds.storage)
    base_storage.flush()
    base_storage.enable_readonly()
    ds = Dataset(storage=local_ds.storage, read_only=True, verbose=False)

    ptds = ds.pytorch(num_workers=2)
    # no need to check input, only care that readonly works
    for _ in ptds:
        pass


@requires_torch
def test_corrupt_dataset(local_ds, corrupt_image_paths, compressed_image_paths):
    img_good = hub.read(compressed_image_paths["jpeg"][0])
    img_bad = hub.read(corrupt_image_paths["jpeg"])
    with local_ds:
        local_ds.create_tensor("image", htype="image", sample_compression="jpeg")
        for i in range(3):
            for i in range(10):
                local_ds.image.append(img_good)
            local_ds.image.append(img_bad)
    num_samples = 0
    num_batches = 0
    dl = local_ds.pytorch(num_workers=0, batch_size=2)
    for (batch,) in dl:
        num_batches += 1
        num_samples += len(batch)
    assert num_samples == 30
    assert num_batches == 15


@requires_torch
@enabled_datasets
def test_pytorch_local_cache(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch()
        return

    epochs = 2

    for _ in range(epochs):
        dl = ds.pytorch(num_workers=1, batch_size=1, use_local_cache=True)
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

        dls = ds.pytorch(
            num_workers=2,
            batch_size=1,
            shuffle=True,
            use_local_cache=True,
        )
        pytorch_small_shuffle_helper(0, 16, dls)


@requires_torch
def test_groups(local_ds, compressed_image_paths):
    img1 = hub.read(compressed_image_paths["jpeg"][0])
    img2 = hub.read(compressed_image_paths["png"][0])
    with local_ds:
        local_ds.create_tensor(
            "images/jpegs/cats", htype="image", sample_compression="jpeg"
        )
        local_ds.create_tensor(
            "images/pngs/flowers", htype="image", sample_compression="png"
        )
        for _ in range(10):
            local_ds.images.jpegs.cats.append(img1)
            local_ds.images.pngs.flowers.append(img2)
    dl = local_ds.pytorch()
    for cat, flower in dl:
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)

    with local_ds:
        local_ds.create_tensor(
            "arrays/x",
        )
        local_ds.create_tensor(
            "arrays/y",
        )
        for _ in range(10):
            local_ds.arrays.x.append(np.random.random((2, 3)))
            local_ds.arrays.y.append(np.random.random((4, 5)))

    dl = local_ds.images.pytorch()
    for cat, flower in dl:
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)


@requires_torch
def test_string_tensors(local_ds):
    with local_ds:
        local_ds.create_tensor("strings", htype="text")
        local_ds.strings.extend([f"string{idx}" for idx in range(5)])

    ptds = local_ds.pytorch(batch_size=1)
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["strings"], [f"string{idx}"])

    ptds2 = local_ds.pytorch(batch_size=None)
    for idx, batch in enumerate(ptds2):
        np.testing.assert_array_equal(batch["strings"], f"string{idx}")


@requires_torch
def test_pytorch_large(local_ds):
    arr_list_1 = [np.random.randn(1500, 1500, i) for i in range(5)]
    arr_list_2 = [np.random.randn(400, 1500, 4, i) for i in range(5)]
    label_list = list(range(5))

    with local_ds as ds:
        ds.create_tensor("img1")
        ds.create_tensor("img2")
        ds.create_tensor("label")
        ds.img1.extend(arr_list_1)
        ds.img2.extend(arr_list_2)
        ds.label.extend(label_list)

    ptds = local_ds.pytorch()
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["img1"][0], arr_list_1[idx])
        np.testing.assert_array_equal(batch["img2"][0], arr_list_2[idx])
        np.testing.assert_array_equal(batch["label"][0], idx)


def run_ddp(rank, size, ds, q, backend="gloo"):
    import torch.distributed as dist
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)

    s = 0
    for x in ds.pytorch(num_workers=0):
        s += int(x["image"][0].mean())

    q.put(s)


@requires_torch
@enabled_datasets
def test_pytorch_ddp(ds):
    import multiprocessing as mp

    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(255)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ds.pytorch()
        return

    size = 2
    processes = []
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    for rank in range(size):
        p = ctx.Process(target=run_ddp, args=(rank, size, ds, q), daemon=True)
        p.start()
        processes.append(p)

    s = 0
    for p in processes:
        p.join()
        p.terminate()
        s += q.get()

    q.close()

    assert s == sum(list(range(254)))
