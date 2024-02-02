import pytest

import numpy as np

from deeplake.core.query import DatasetQuery
from deeplake.util.exceptions import (
    DatasetViewSavingError,
    InvalidOperationError,
    InvalidViewException,
    DatasetHandlerError,
)
import deeplake
from uuid import uuid4


first_row = {"images": [1, 2, 3], "labels": [0]}
second_row = {"images": [6, 7, 5], "labels": [1]}
rows = [first_row, second_row]
class_names = ["dog", "cat", "fish"]


@deeplake.compute
def deeplake_compute_filter(sample_in, mod):
    val = sample_in.abc.numpy()[0]
    return val % mod == 0


@pytest.mark.slow
def _populate_data_linked(ds, n, compressed_image_paths, labels):
    with ds:
        if "images" not in ds:
            ds.create_tensor("images", htype="link[image]", sample_compression="png")
            if labels:
                ds.create_tensor("labels", htype="class_label", class_names=class_names)
            for i in range(n):
                ds.images.append(deeplake.link(compressed_image_paths["png"][0]))
                if labels:
                    ds.labels.append(rows[i % 2]["labels"])


def _populate_data(ds, n=1, linked=False, paths=None, labels=True):
    if linked:
        return _populate_data_linked(ds, n, paths, labels)
    with ds:
        if "images" not in ds:
            ds.create_tensor("images")
            if labels:
                ds.create_tensor("labels", htype="class_label", class_names=class_names)
        for _ in range(n):
            for row in rows:
                ds.images.append(row["images"])
                if labels:
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "optimize,idx_subscriptable", [(True, True), (False, False), (True, False)]
)
def test_sub_sample_view_save(optimize, idx_subscriptable, compressed_image_paths):
    id = str(uuid4())
    arr = np.random.random((100, 32, 32, 3))
    with deeplake.dataset(".tests/ds", overwrite=True) as ds:
        ds.create_tensor("x")
        ds.x.extend(arr)
    _populate_data(ds, linked=True, n=100, paths=compressed_image_paths, labels=False)
    with pytest.raises(DatasetViewSavingError):
        ds.save_view(optimize=optimize)
    view = ds[10:77:2, 2:17, 19:31, :1]
    arr = arr[10:77:2, 2:17, 19:31, :1]
    if not idx_subscriptable:
        view = view[0]
        arr = arr[0]
    np.testing.assert_array_equal(view.x.numpy(), arr)
    with pytest.raises(DatasetViewSavingError):
        view.save_view(optimize=optimize)
    ds.commit()
    ds.save_view(optimize=optimize, id=id)
    view.save_view(optimize=optimize, id=id)  # test overwrite
    assert len(ds.get_views()) == 1
    view2 = ds.get_views()[0].load()
    np.testing.assert_array_equal(view.x.numpy(), view2.x.numpy())
    view3 = ds.get_view(id).load()
    np.testing.assert_array_equal(view.x.numpy(), view3.x.numpy())


@pytest.mark.parametrize("optimize", [True, False])
def test_dataset_view_save(optimize):
    with deeplake.dataset(".tests/ds", overwrite=True) as ds:
        _populate_data(ds)
    view = ds.filter("labels == 'dog'")
    with pytest.raises(DatasetViewSavingError):
        view.save_view(path=".tests/ds_view", overwrite=True, optimize=optimize)
    ds.commit()
    view.save_view(path=".tests/ds_view", overwrite=True, optimize=optimize)
    view2 = deeplake.dataset(".tests/ds_view")
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())
    _populate_data(ds)
    ds.commit()
    view = ds.filter("labels == 'dog'")
    _populate_data(ds)
    with pytest.raises(InvalidViewException):
        view.save_view(path=".tests/ds_view", overwrite=True, optimize=optimize)
    ds.commit()
    view = ds.filter("labels == 'dog'")
    view.save_view(path=".tests/ds_view", overwrite=True, optimize=optimize)


@pytest.mark.parametrize(
    "ds_generator",
    [
        "local_ds_generator",
        pytest.param("s3_ds_generator", marks=pytest.mark.slow),
        # pytest.param("gcs_ds_generator", marks=pytest.mark.slow),
        pytest.param("azure_ds_generator", marks=pytest.mark.slow),
        pytest.param("hub_cloud_ds_generator", marks=pytest.mark.slow),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "stream,num_workers,read_only,progressbar,query_type,optimize,linked",
    [
        (False, 2, True, True, "string", True, False),
        (True, 0, False, False, "function", False, True),
    ],
)
@pytest.mark.timeout(1200)
def test_inplace_dataset_view_save(
    ds_generator,
    stream,
    num_workers,
    read_only,
    progressbar,
    query_type,
    optimize,
    linked,
    compressed_image_paths,
):
    ds = ds_generator()
    is_hub = ds.path.startswith("hub://")
    if read_only and not is_hub:
        return
    id = str(uuid4())
    to_del = [id]
    _populate_data(ds, n=10, linked=linked, paths=compressed_image_paths)
    ds.commit()
    ds.read_only = read_only
    f = (
        f"labels == 'dog'"
        if query_type == "string"
        else lambda s: int(s.labels.numpy()) == 0
    )
    view = ds.filter(
        f, save_result=stream, num_workers=num_workers, progressbar=progressbar
    )
    indices = list(view.sample_indices)
    assert indices
    assert list(view[:4:2].sample_indices) == indices[:4:2]
    if stream:
        to_del.append(view._vds.info["id"])
    assert len(ds.get_views()) == int(stream)
    vds_path = view.save_view(optimize=optimize, id=id)
    assert len(ds.get_views()) == 1 + int(stream)
    view2 = deeplake.dataset(vds_path, token=ds.token)
    assert indices == list(view2.sample_indices)
    if ds.path.startswith("hub://"):
        assert vds_path.startswith("hub://")
        assert ds.path + "/.queries/" in vds_path
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())
    ds_orig = ds
    if not read_only and is_hub:
        ds = deeplake.load(ds.path, read_only=True, token=ds.token)
    entry = ds.get_view(id)
    assert entry.virtual == (not optimize)
    assert indices == list(entry.load().sample_indices)
    entry.optimize()
    assert not entry.virtual
    view3 = entry.load()
    assert indices == list(view3.sample_indices)
    assert list(view3[:4:2].sample_indices) == indices[:4:2], (
        list(view3.sample_indices),
        indices,
    )
    with pytest.raises(InvalidOperationError):
        for t in view3.tensors:
            view3[t].append(np.zeros((1,)))
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view3[t].numpy())
    for id in to_del:
        ds_orig.delete_view(id)
        with pytest.raises(KeyError):
            ds_orig.get_view(id)


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


def test_filter_deeplake_compute(local_ds):
    with local_ds:
        local_ds.create_tensor("abc")
        for i in range(100):
            local_ds.abc.append(i)

    result = local_ds.filter(deeplake_compute_filter(mod=2), progressbar=False)
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


@pytest.mark.slow
def test_query_sample_info(local_ds, compressed_image_paths):
    ds = local_ds
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="jpg")
        path_to_shape = {}
        for path in compressed_image_paths["jpeg"]:
            img = deeplake.read(path)
            ds.image.append(img)
            path_to_shape[path] = img.shape
    for path in compressed_image_paths["jpeg"]:
        view = ds.filter(f"r'{path}' in image.sample_info['filename']")
        np.testing.assert_array_equal(
            view[0].image.numpy().reshape(-1), np.array(deeplake.read(path)).reshape(-1)
        )  # reshape to ignore grayscale normalization


@deeplake.compute
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


def test_view_sample_indices(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
    assert list(ds[:5].sample_indices) == list(range(5))
    assert list(ds[5:].sample_indices) == list(range(5, 10))


def test_query_view_union(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
    v1 = ds.filter(lambda s: s.x.numpy() % 2)
    v2 = ds.filter(lambda s: not (s.x.numpy() % 2))
    union = ds[sorted(list(set(v1.sample_indices).union(v2.sample_indices)))]
    np.testing.assert_array_equal(union.x.numpy(), ds.x.numpy())


def test_view_saving_with_path(local_ds):
    with local_ds as ds:
        ds.create_tensor("nums")
        ds.nums.extend(list(range(100)))
        ds.commit()
        with pytest.raises(DatasetViewSavingError):
            ds[:10].save_view(path=local_ds.path)
        vds_path = local_ds.path + "/../vds"
        try:
            deeplake.delete(vds_path, force=True)
        except DatasetHandlerError:
            pass
        ds[:10].save_view(path=vds_path)
        with pytest.raises(DatasetViewSavingError):
            ds[:10].save_view(path=vds_path)
        deeplake.delete(vds_path, force=True)


def test_strided_view_bug(local_ds):
    with local_ds as ds:
        ds.create_tensor("nums")
        ds.nums.extend(list(range(200)))
        ds.commit()
    view = ds[:100:2]
    view.save_view()
    view2 = ds.get_views()[0].load()
    np.testing.assert_array_equal(view.nums.numpy(), view2.nums.numpy())


def test_view_mutability(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(list(range(50)))

    full_view = ds[:]
    half_view = ds[:25]

    with pytest.raises(InvalidOperationError):
        half_view.abc.extend(list(range(50)))

    a = ds.commit()

    half_view_2 = ds[:50]

    ds.abc.extend(list(range(50)))

    # full_view, half_view points to last commit, not HEAD
    np.testing.assert_array_equal(full_view.abc.numpy(), ds[:50].abc.numpy())
    np.testing.assert_array_equal(half_view.abc.numpy(), ds[:25].abc.numpy())

    # half_view_2 invalidated due to update
    with pytest.raises(InvalidViewException):
        half_view_2.abc.numpy(), ds[:50].abc.numpy()

    view1 = ds[10:20]

    ds.checkout(a)
    # view1 invalidated because of base ds checkout
    with pytest.raises(InvalidViewException):
        view1.abc

    ds.checkout("main")

    assert full_view.commit_id == a


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [1, 2])
def test_link_materialize(local_ds, num_workers):
    with local_ds as ds:
        ds.create_tensor("abc", htype="link[image]", sample_compression="jpg")
        ds.abc.extend(
            [
                (
                    deeplake.link("https://picsum.photos/20/20")
                    if i % 2
                    else deeplake.link("https://picsum.photos/10/10")
                )
                for i in range(20)
            ]
        )
        ds.commit()

    view = ds[::2]
    view.save_view(id="view_1", optimize=True, num_workers=num_workers)

    loaded = ds.load_view("view_1")

    assert loaded.abc.numpy().shape == (10, 10, 10, 3)
