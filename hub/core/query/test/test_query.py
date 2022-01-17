import pytest

import numpy as np

from hub.core.query import DatasetQuery
import hub


first_row = {"images": [1, 2, 3], "labels": [0]}
second_row = {"images": [6, 7, 5], "labels": [1]}
rows = [first_row, second_row]
class_names = ["dog", "cat", "fish"]


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

        ds.images.append([0])
        ds.images.append([1])
        ds.images.append([2])

        ds.labels.append([0])
        ds.labels.append([1])

    result = ds.filter("labels == 0", progressbar=False)
    assert len(result) == 1

    result = ds.filter("images == 2", progressbar=False)
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
        # "s3_ds_generator",
        # "gcs_ds_generator",
        # "hub_cloud_ds_generator",
    ], indirect=True
)
@pytest.mark.parametrize("stream", [False])
@pytest.mark.parametrize("num_workers", [2])
@pytest.mark.parametrize("read_only", [False])
@pytest.mark.parametrize("progressbar", [True])
def test_inplace_dataset_view_save(ds_generator, stream, num_workers, read_only, progressbar):
    ds = ds_generator()
    if read_only and not ds.path.startswith("hub://"):
        return
    with ds:
        _populate_data(ds, n=2)
    view = ds.filter("labels == 'dog'", store_result=stream, num_workers=num_workers, progressbar=progressbar)
    assert len(ds._get_query_history()) == int(stream)
    vds_path = view.store()
    assert len(ds._get_query_history()) == 1
    view2 = hub.dataset(vds_path)
    if ds.path.startswith("hub://"):
        assert vds_path.startswith("hub://")
        if read_only:
            assert vds_path[6:].split("/")[1] == "queries"
        else:
            assert ds.path + "/.queries/" in vds_path
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())
    if ds.path.startswith("hub://") and read_only:
        # Delete queries ds from testing acc:
        org = ds.path[6:].split("/")[1]
        hub.delete(f"hub://{org}/queries", large_ok=True)
    def filter_result(ds):
        return ds[0].labels.numpy()

    assert (
        ds.filter("labels == 3141", num_workers=2, progressbar=False)[0].labels.numpy()
        == 3141
    )
    assert (
        ds.filter(lambda s: s.labels.numpy() == 3141, num_workers=2, progressbar=False)[0].labels.numpy()
        == 3141
    )


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
