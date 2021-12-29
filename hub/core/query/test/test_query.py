import pytest

import numpy as np

from hub.core.query import DatasetQuery
from hub.core.query.query import EvalGenericTensor, EvalLabelClassTensor
from hub.core.index import Index
from hub.tests.dataset_fixtures import (
    local_ds_generator,
    s3_ds_generator,
    hub_ds_generator,
)
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
def sample_ds(memory_ds):
    _populate_data(memory_ds)
    return memory_ds


def test_tensor_functions(sample_ds):
    for ind, row in enumerate(rows):
        i = EvalGenericTensor(DatasetQuery(sample_ds[ind], ""), sample_ds[ind].images)
        i.at_index(Index(ind))

        l = EvalGenericTensor(DatasetQuery(sample_ds[ind], ""), sample_ds[ind].labels)
        l.at_index(Index(ind))

        assert i.min == min(row["images"])
        assert i.max == max(row["images"])
        assert i.mean == sum(row["images"]) / len(row["images"])
        assert i.shape[0] == len(row["images"])
        assert i.size == len(row["images"])
        assert i[1] == row["images"][1]

        assert l == row["labels"][0]
        assert l != row["labels"][0] + 2
        assert l > row["labels"][0] - 1
        assert l < row["labels"][0] + 1
        assert l >= row["labels"][0]
        assert l <= row["labels"][0]


def test_class_label_tensor_function(sample_ds):
    eval_object = EvalLabelClassTensor(
        DatasetQuery(sample_ds[0], ""), sample_ds[0].labels
    )
    eval_object.at_index(Index(0))
    assert eval_object == "dog"

    eval_object = EvalLabelClassTensor(
        DatasetQuery(sample_ds[1], ""), sample_ds[1].labels
    )
    eval_object.at_index(Index(1))
    assert eval_object == "cat"


def test_tensor_subscript(memory_ds):
    arr = [[[1], [2]], [[2], [3]], [[4], [5]]]

    memory_ds.create_tensor("images")
    memory_ds.images.append(arr)

    i = EvalGenericTensor(DatasetQuery(memory_ds[0], ""), memory_ds[0].images)
    i.at_index(Index(0))

    assert i[2, 1] == arr[2][1]
    assert i[1].min == min(arr[1])[0]


@pytest.mark.parametrize(
    ["query", "results"],
    [
        ["images.max == 3", [True, False]],
        ["images.min == 5", [False, True]],
        ["images[:1].min == 6", [False, True]],
        ["images[1] == 2", [True, False]],
        ["labels == 0", [True, False]],
        ["labels > 0 ", [False, True]],
        ["labels in [cat, dog]", [True, True]],
        ["labels < 0 ", [False, False]],
        ["labels.contains(0)", [True, False]],  # weird usecase
    ],
)
def test_query(sample_ds, query, results):
    query = DatasetQuery(sample_ds, query)

    for i in range(len(results)):
        assert query(sample_ds[i]) == results[i]


def test_query_string_tensor(memory_ds):
    data = ["string1", "string2", ""]

    with memory_ds as ds:
        ds.create_tensor("text", htype="text")
        for v in data:
            ds.text.append(v)

    assert DatasetQuery(memory_ds, 'text == "string1"')(memory_ds[0]) == True
    assert DatasetQuery(memory_ds, 'text == "string1"')(memory_ds[1]) == False
    assert DatasetQuery(memory_ds, "len(text) == 0")(memory_ds[2]) == True


def test_query_json_tensor(memory_ds):
    data = ['{ "a": 1 }', '{ "b": { "a": 1 }}', ""]

    with memory_ds as ds:
        ds.create_tensor("json", htype="json")
        for v in data:
            ds.json.append(v)

    assert DatasetQuery(memory_ds, 'json["a"] == 1')(memory_ds[0]) == True
    assert DatasetQuery(memory_ds, 'json["b"]["a"] == 1')(memory_ds[1]) == True

    with pytest.raises(KeyError):
        assert (
            DatasetQuery(memory_ds, 'json["b"]["a"] == None')(memory_ds[0]) == True
        )  # not sure what should happen here


def test_query_groups(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("images/image1")
        ds.create_tensor("images/image2")

        ds.images.image1.append([1, 2, 3])
        ds.images.image2.append([3, 2, 1])

    assert (
        DatasetQuery(memory_ds, "images.image1.mean == images.image2.mean")(
            memory_ds[0]
        )
        == True
    )


def test_different_size_ds_query(memory_ds):

    with memory_ds as ds:
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

    def filter_result(ds):
        return ds[0].labels.numpy()

    assert (
        filter_result(ds.filter("labels == 3141", num_workers=2, progressbar=False))
        == 3141
    )
    assert (
        filter_result(
            ds.filter(
                lambda s: s.labels.numpy() == 3141, num_workers=2, progressbar=False
            )
        )
        == 3141
    )


def test_dataset_view_save():
    with hub.dataset(".tests/ds", overwrite=True) as ds:
        _populate_data(ds)
    view = ds.filter("labels == 'dog'")
    view.store(".tests/ds_view", overwrite=True)
    view2 = hub.dataset(".tests/ds_view")
    for t in view.tensors:
        np.testing.assert_array_equal(view[t].numpy(), view2[t].numpy())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    "ds_generator",
    [s3_ds_generator, local_ds_generator, hub_ds_generator],
    indirect=True,
)
@pytest.mark.parametrize("read_only", [False, True])
def test_inplace_dataset_view_save(ds_generator, stream, num_workers, read_only):
    ds = ds_generator()
    if read_only and not ds.path.startswith("hub://"):
        return
    with ds:
        _populate_data(ds, n=2)
    view = ds.filter("labels == 'dog'", store_result=stream, num_workers=num_workers)
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
