import pytest

import numpy as np

from hub.core.query import DatasetQuery
import hub
from uuid import uuid4
import os


first_row = {"images": [1, 2, 3], "labels": [0]}
second_row = {"images": [6, 7, 5], "labels": [1]}
rows = [first_row, second_row]
class_names = ["dog", "cat", "fish"]


@hub.compute
def hub_compute_filter(sample_in, mod):
    val = sample_in.abc.numpy()[0]
    return val % mod == 0


def _populate_data(ds, n=1):
    with ds:
        ds.create_tensor("images")
        ds.create_tensor("labels", htype="class_label", class_names=class_names)
        for _ in range(n):
            for row in rows:
                ds.images.append(row["images"])
                ds.labels.append(row["labels"])


@pytest.fixture
def sample_ds(local_ds):
    _populate_data(local_ds)
    return local_ds


@pytest.mark.parametrize(
    ["query", "results"],
    [
        ["images.max == 3", [True, False]],
        ["images.min == 5", [False, True]],
        ["images[1] == 2", [True, False]],
        ["labels == 0", [True, False]],
        ["labels > 0 ", [False, True]],
        ["labels in ['cat', 'dog']", [True, True]],
        ["labels < 0 ", [False, False]],
        ["labels.contains(0)", [True, False]],  # weird usecase
    ],
)
def test_query(sample_ds, query, results):
    query = DatasetQuery(sample_ds, query)
    r = query.execute()

    for i in range(len(results)):
        if results[i]:
            assert i in r
        else:
            assert i not in r


def test_different_size_ds_query(local_ds):

    with local_ds as ds:
        ds.create_tensor("images")
        ds.create_tensor("labels")
        ds.create_tensor("test", htype="class_label")

        ds.images.append([0])
        ds.images.append([1])
        ds.images.append([2])

        ds.labels.append([0])
        ds.labels.append([1])

        ds.test.append([0])

    result = ds.filter("labels == 0", progressbar=False)
    assert len(result) == 1

    result = ds.filter("images == 2", progressbar=False)
    assert len(result) == 0

    # invalid queries
    result = ds.filter("images == 'ln'", progressbar=False)
    assert len(result) == 0

    result = ds.filter("labels == 'lb'", progressbar=False)
    assert len(result) == 0

    result = ds.filter("test == 'lb'", progressbar=False)
    assert len(result) == 0


def test_query_scheduler(local_ds):
    with local_ds as ds:
        ds.create_tensor("labels")
        ds.labels.extend(np.arange(10_000))

    f1 = "labels % 2 == 0"
    f2 = lambda s: s.labels.numpy() % 2 == 0

    view1 = ds.filter(f1, num_workers=2, progressbar=True)
    view2 = ds.filter(f2, num_workers=2, progressbar=True)

    np.testing.assert_array_equal(view1.labels.numpy(), view2.labels.numpy())


def test_dataset_view_save():
    with hub.dataset(".tests/ds", overwrite=True) as ds:
        _populate_data(ds)
    view = ds.filter("labels == 'dog'")
    view.store(".tests/ds_view", overwrite=True)
    view2 = hub.dataset(".tests/ds_view")
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())


@pytest.mark.parametrize(
    "ds_generator",
    [
        "local_ds_generator",
        "s3_ds_generator",
        # "gcs_ds_generator",
        "hub_cloud_ds_generator",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "stream,num_workers,read_only,progressbar,query_type",
    [
        (False, 2, False, True, "string"),
        (True, 0, True, False, "function"),
    ],
)
def test_inplace_dataset_view_save(
    ds_generator, stream, num_workers, read_only, progressbar, query_type
):
    ds = ds_generator()
    if read_only and not ds.path.startswith("hub://"):
        return
    _populate_data(ds, n=2)
    ds.read_only = read_only
    f = (
        f"labels == 'dog'#{uuid4().hex}"
        if query_type == "string"
        else lambda s: s.labels == "dog"
    )
    view = ds.filter(
        f, store_result=stream, num_workers=num_workers, progressbar=progressbar
    )
    assert read_only or len(ds._get_query_history()) == int(stream)
    vds_path = view.store()
    assert read_only or len(ds._get_query_history()) == 1
    view2 = hub.dataset(vds_path)
    if ds.path.startswith("hub://"):
        assert vds_path.startswith("hub://")
        if read_only:
            assert vds_path[6:].split("/")[1] == "queries"
        else:
            assert ds.path + "/.queries/" in vds_path
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())


def test_group(local_ds):
    with local_ds as ds:
        ds.create_tensor("labels/t1")
        ds.create_tensor("labels/t2")

        ds.labels.t1.append([0])
        ds.labels.t2.append([1])

    result = local_ds.filter("labels.t1 == 0", progressbar=False)
    assert len(result) == 1

    result = local_ds.filter("labels.t2 == 1", progressbar=False)
    assert len(result) == 1


def test_filter_hub_compute(local_ds):
    with local_ds:
        local_ds.create_tensor("abc")
        for i in range(100):
            local_ds.abc.append(i)

    result = local_ds.filter(hub_compute_filter(mod=2), progressbar=False)
    assert len(result) == 50


def test_multi_category_labels(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="png")
        ds.create_tensor(
            "label", htype="class_label", class_names=["cat", "dog", "tree"]
        )
        r = np.random.randint(50, 100, (32, 32, 3), dtype=np.uint8)
        ds.image.append(r)
        ds.label.append([0, 1])
        ds.image.append(r + 2)
        ds.label.append([1, 2])
        ds.image.append(r * 2)
        ds.label.append([0, 2])
    view1 = ds.filter("label == 0")
    view2 = ds.filter("label == 'cat'")
    view3 = ds.filter("'cat' in label")
    view4 = ds.filter("label.contains('cat')")
    exp_images = np.array([r, r * 2])
    exp_labels = np.array([[0, 1], [0, 2]], dtype=np.uint8)
    for v in (view1, view2, view3, view4):
        np.testing.assert_array_equal(v.image.numpy(), exp_images)
        np.testing.assert_array_equal(v.label.numpy(), exp_labels)


def test_query_shape(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="png")
        shapes = [(16, 32, 3), (32, 16, 3), (32, 32, 3), (16, 16, 3)]
        counts = [5, 4, 3, 2]
        for shape, count in zip(shapes, counts):
            ds.image.extend(np.random.randint(50, 100, (count, *shape), dtype=np.uint8))
    for shape, count in zip(shapes, counts):
        assert len(ds.filter(f"image.shape == {shape}")) == count


def test_query_sample_info(local_ds, compressed_image_paths):
    ds = local_ds
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="jpg")
        path_to_shape = {}
        for path in compressed_image_paths["jpeg"]:
            img = hub.read(path)
            ds.image.append(img)
            path_to_shape[path] = img.shape
    for path in compressed_image_paths["jpeg"]:
        view = ds.filter(f"r'{path}' in image.sample_info['filename']")
        np.testing.assert_array_equal(
            view[0].image.numpy().reshape(-1), np.array(hub.read(path)).reshape(-1)
        )  # reshape to ignore grayscale normalization


@hub.compute
def create_dataset(class_num, sample_out):
    """Add new element with a specific class"""
    sample_out.append({"classes": np.uint32(class_num)})
    return sample_out


def test_query_bug_transformed_dataset(local_ds):
    with local_ds as ds:
        ds.create_tensor(
            "classes",
            htype="class_label",
            class_names=["class_0", "class_1", "class_2"],
        )

    with local_ds as ds:
        # Add 30 elements with randomly generated class
        list_classes = list(np.random.randint(3, size=30, dtype=np.uint32))
        create_dataset().eval(list_classes, ds, num_workers=2)

    ds_view = local_ds.filter("classes == 'class_0'", scheduler="threaded")
    np.testing.assert_array_equal(ds_view.classes.numpy()[:, 0], [0] * len(ds_view))
