import pytest

import numpy as np

from hub.core.query import DatasetQuery

first_row = {"images": [1, 2, 3], "labels": [0]}
second_row = {"images": [6, 7, 5], "labels": [1]}
rows = [first_row, second_row]
class_names = ["dog", "cat", "fish"]


@pytest.fixture
def sample_ds(local_ds):
    with local_ds as ds:
        ds.create_tensor("images")
        ds.create_tensor("labels", htype="class_label", class_names=class_names)

        for row in rows:
            ds.images.append(row["images"])
            ds.labels.append(row["labels"])

    return local_ds


@pytest.mark.parametrize(
    ["query", "results"],
    [
        ["images.max == 3", [True, False]],
        ["images.min == 5", [False, True]],
        ["images[:1].min == 6", [False, True]],
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
