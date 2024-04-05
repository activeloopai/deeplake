import deeplake
import numpy as np
from deeplake.tests.common import requires_libdeeplake
from deeplake.util.exceptions import (
    DynamicTensorNumpyError,
    EmptyTokenException,
)

from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
import random
import math
import pytest


@requires_libdeeplake
def test_indexing(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    assert len(deeplake_indra_ds) == len(indra_ds)

    # test slice indices
    assert np.all(deeplake_indra_ds.label.numpy() == indra_ds.label.numpy())
    assert np.all(deeplake_indra_ds.label[5:55].numpy() == indra_ds.label[5:55].numpy())

    assert np.all(deeplake_indra_ds[5:55].label.numpy() == indra_ds.label[5:55].numpy())

    # test int indices
    assert np.all(deeplake_indra_ds.label[0].numpy() == indra_ds.label[0].numpy())

    assert np.all(deeplake_indra_ds[0].label.numpy() == indra_ds.label[0].numpy())

    # test list indices
    assert np.all(
        deeplake_indra_ds.label[[0, 1]].numpy() == indra_ds.label[[0, 1]].numpy()
    )

    assert np.all(
        deeplake_indra_ds[[0, 1]].label.numpy() == indra_ds.label[[0, 1]].numpy()
    )

    # test tuple indices
    assert np.all(
        deeplake_indra_ds[(0, 1),].label.numpy() == indra_ds.label[(0, 1),].numpy()
    )

    assert np.all(
        deeplake_indra_ds[(0, 1),].label.numpy() == indra_ds.label[(0, 1),].numpy()
    )


@requires_libdeeplake
def test_save_view(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    deeplake_ds.commit("First")

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)
    deeplake_indra_ds.save_view()
    assert (
        deeplake_indra_ds.base_storage["queries.json"]
        == deeplake_ds.base_storage["queries.json"]
    )


@requires_libdeeplake
def test_empty_token_exception(local_auth_ds):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    with local_auth_ds:
        local_auth_ds.create_tensor("label", htype="generic", dtype=np.int32)

    loaded = deeplake.load(local_auth_ds.path, token="")

    with pytest.raises(EmptyTokenException):
        dss = dataset_to_libdeeplake(loaded)


@requires_libdeeplake
def test_load_view(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor(
            "image", htype="image", dtype=np.uint8, sample_compression="jpg"
        )
        for i in range(100):
            deeplake_ds.label.append(i % 10)
            deeplake_ds.image.append(np.random.randint(0, 255, (100, 200, 3), np.uint8))

    deeplake_ds.commit("First")

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    with pytest.raises(Exception):
        dataloader = deeplake_indra_ds.pytorch()

    query_str = "select * group by label"
    view = deeplake_ds.query(query_str)
    view_path = view.save_view()
    view_id = view_path.split("/")[-1]
    view = deeplake_ds.load_view(view_id)

    dataloader = view[:3].dataloader().pytorch()
    iss = []
    for i, batch in enumerate(dataloader):
        assert len(batch["label"]) == 10
        iss.append(i)

    assert iss == [0, 1, 2]
    assert np.all(indra_ds.image.numpy() == deeplake_indra_ds.image.numpy())

    view = deeplake_ds[0:50].query(query_str)
    view_path = view.save_view()
    view_id = view_path.split("/")[-1]
    view = deeplake_ds.load_view(view_id)

    dataloader = view[:3].dataloader().pytorch()
    iss = []
    for i, batch in enumerate(dataloader):
        assert len(batch["label"]) == 5
        iss.append(i)

    assert iss == [0, 1, 2]
    assert np.all(indra_ds.image.numpy() == deeplake_indra_ds.image.numpy())


@requires_libdeeplake
def test_query(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor(
            "image", htype="image", dtype=np.uint8, sample_compression="jpg"
        )
        for i in range(100):
            deeplake_ds.label.append(int(i / 10))
            deeplake_ds.image.append(np.random.randint(0, 255, (100, 200, 3), np.uint8))

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    view = deeplake_indra_ds.query("SELECT * GROUP BY label")
    assert len(view) == 10
    assert view.label.shape == view.tensors["label"].shape
    for i in range(len(view)):
        arr = view.label[i].numpy()
        assert len(arr) == 10
        for a in arr:
            assert np.all(a == i)

    view2 = view.query("SELECT * WHERE all(label == 2)")
    assert len(view2) == 1
    arr = view2.label.numpy()
    assert len(arr) == 10
    for a in arr:
        assert np.all(a == 2)


@requires_libdeeplake
def test_metadata(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor(
            "image", htype="image", dtype=np.uint8, sample_compression="jpeg"
        )
        deeplake_ds.create_tensor("none_metadata")
        deeplake_ds.create_tensor(
            "sequence", htype="sequence[class_label]", dtype=np.uint8
        )

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)
    assert deeplake_indra_ds.label.htype == "generic"
    assert deeplake_indra_ds.label.dtype == np.int32
    assert deeplake_indra_ds.label.sample_compression == None
    assert deeplake_indra_ds.image.htype == "image"
    assert deeplake_indra_ds.image.dtype == np.uint8
    assert deeplake_indra_ds.image.sample_compression == "jpeg"
    assert deeplake_indra_ds.sequence.htype == "sequence[class_label]"
    assert deeplake_indra_ds.sequence.dtype == np.uint8
    assert deeplake_indra_ds.sequence.sample_compression == None
    assert deeplake_indra_ds.none_metadata.htype == None
    assert deeplake_indra_ds.none_metadata.dtype == None
    assert deeplake_indra_ds.none_metadata.sample_compression == None


@requires_libdeeplake
def test_accessing_data(local_auth_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    assert np.all(
        np.isclose(deeplake_indra_ds.label.numpy(), deeplake_indra_ds["label"].numpy())
    )


@requires_libdeeplake
def test_sequences_accessing_data(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(200):
            deeplake_ds.label.append(int(i / 101))
        deeplake_ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", dtype=np.uint8
        )
        for i in range(199):
            deeplake_ds.image.append(np.zeros((10, 10, 3), dtype=np.uint8))
        deeplake_ds.image.append(np.zeros((20, 10, 3), np.uint8))

    deeplake_indra_ds = deeplake_ds.query("SELECT * GROUP BY label")
    assert len(deeplake_indra_ds) == 2
    assert deeplake_indra_ds.image.shape == (2, None, None, 10, 3)
    assert deeplake_indra_ds[0].image.shape == (101, 10, 10, 3)
    assert deeplake_indra_ds[0, 0].image.shape == (10, 10, 3)
    assert len(deeplake_indra_ds[0].image.numpy()) == 101
    assert deeplake_indra_ds[1].image.shape == (99, None, 10, 3)
    assert deeplake_indra_ds[1, 0].image.shape == (10, 10, 3)
    assert deeplake_indra_ds[1, 98].image.shape == (20, 10, 3)
    assert len(deeplake_indra_ds[1].image.numpy()) == 99
    assert deeplake_indra_ds[1].image.numpy()[0].shape == (10, 10, 3)
    assert deeplake_indra_ds[1].image.numpy()[98].shape == (20, 10, 3)


@requires_libdeeplake
def test_query_tensors_polygon_htype_consistency(local_auth_ds_generator):
    ds = local_auth_ds_generator()
    with ds:
        ds.create_tensor(
            "polygon",
            dtype=np.float32,
            htype="polygon",
            sample_compression=None,
        )
        ds.create_tensor(
            "labels",
            dtype=np.uint16,
            htype="generic",
            sample_compression=None,
        )
        for i in range(10):
            polygons = []
            for j in range(i):
                p = np.ndarray((3 + j, 2))
                for k in range(3 + j):
                    p[k] = [
                        200 * (j % 3) + 150 * (math.sin(6.28 * k / (3 + j)) + 1) / 2,
                        200 * (j / 3) + 150 * (math.cos(6.28 * k / (3 + j)) + 1) / 2,
                    ]
                polygons.append(p)
            ds.labels.append(i)
            ds.polygon.append(polygons)

    view = ds.query("select *, labels+labels as new_tensor")
    for i in range(len(ds)):
        orig = ds.polygon[i].numpy()
        new = view.polygon[i].numpy()

        for i, j in zip(orig, new):
            assert np.all(i == j)


@requires_libdeeplake
def test_random_split_with_seed(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    from deeplake.core.seed import DeeplakeRandom

    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(i % 100))

    deeplake_indra_ds = deeplake_ds.query("SELECT * GROUP BY label")

    initial_state = np.random.get_state()
    DeeplakeRandom().seed(100)
    split1 = deeplake_indra_ds.random_split([0.2, 0.2, 0.6])
    assert len(split1) == 3
    assert len(split1[0]) == 20

    DeeplakeRandom().seed(101)
    split2 = deeplake_indra_ds.random_split([0.2, 0.2, 0.6])
    assert len(split2) == 3
    assert len(split2[0]) == 20

    DeeplakeRandom().seed(100)
    split3 = deeplake_indra_ds.random_split([0.2, 0.2, 0.6])
    assert len(split3) == 3
    assert len(split3[0]) == 20

    for i in range(len(split1)):
        assert np.all(split1[i].label.numpy() == split3[i].label.numpy())
        assert not np.all(split1[i].label.numpy() == split2[i].label.numpy())

    np.random.set_state(initial_state)


@requires_libdeeplake
def test_random_split(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(i % 100))

    deeplake_indra_ds = deeplake_ds.query("SELECT * GROUP BY label")

    split = deeplake_indra_ds.random_split([0.2, 0.2, 0.6])
    assert len(split) == 3
    assert len(split[0]) == 20

    l = split[0].dataloader().pytorch()
    for b in l:
        pass
    assert len(split[1]) == 20
    l = split[1].dataloader().pytorch()
    for b in l:
        pass
    assert len(split[2]) == 60
    l = split[1].dataloader().pytorch()
    for b in l:
        pass

    split = deeplake_indra_ds.random_split([30, 20, 10, 40])
    assert len(split) == 4
    assert len(split[0]) == 30
    assert len(split[1]) == 20
    assert len(split[2]) == 10
    assert len(split[3]) == 40

    train, val = deeplake_indra_ds[0:50].random_split([0.8, 0.2])
    assert len(train) == 40
    l = train.dataloader().pytorch().shuffle()
    for b in l:
        pass
    assert len(val) == 10
    l = val.dataloader().pytorch().shuffle()
    for b in l:
        pass


@requires_libdeeplake
def test_virtual_tensors(local_auth_ds_generator):
    deeplake_ds = local_auth_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor("embeddings", htype="generic", dtype=np.float32)
        deeplake_ds.create_tensor("text", htype="text")
        deeplake_ds.create_tensor("json", htype="json")
        for i in range(100):
            count = i % 5
            deeplake_ds.label.append([int(i % 100)] * count)
            deeplake_ds.embeddings.append(
                [1.0 / float(i + 1), 0.0, -1.0 / float(i + 1)]
            )
            deeplake_ds.text.append(f"Hello {i}")
            deeplake_ds.json.append('{"key": "val"}')

    deeplake_indra_ds = deeplake_ds.query("SELECT shape(label)[0] as num_labels")
    assert np.all(
        deeplake_indra_ds.num_labels.data()["value"]
        == deeplake_indra_ds.num_labels.numpy()
    )
    assert list(deeplake_indra_ds.tensors.keys()) == ["num_labels"]
    assert len(deeplake_indra_ds) == 100
    assert deeplake_indra_ds.num_labels[0].numpy() == [0]
    assert deeplake_indra_ds.num_labels[1].numpy() == [1]
    assert deeplake_indra_ds.num_labels[2].numpy() == [2]
    assert deeplake_indra_ds.num_labels[3].numpy() == [3]
    assert deeplake_indra_ds.num_labels[4].numpy() == [4]
    assert np.sum(deeplake_indra_ds.num_labels.numpy()) == 200
    deeplake_indra_ds = deeplake_ds.query("SELECT *, shape(label)[0] as num_labels")
    assert list(deeplake_indra_ds.tensors.keys()) == [
        "label",
        "embeddings",
        "text",
        "json",
        "num_labels",
    ]
    assert deeplake_indra_ds.text[0].data() == deeplake_ds.text[0].data()
    assert deeplake_indra_ds.json[0].data() == {"value": '{"key": "val"}'}
    assert deeplake_ds.json[0].data() == {"value": '{"key": "val"}'}

    deeplake_indra_ds = deeplake_ds.query(
        "SELECT l2_norm(embeddings - ARRAY[0, 0, 0]) as score order by l2_norm(embeddings - ARRAY[0, 0, 0]) asc"
    )
    assert list(deeplake_indra_ds.tensors.keys()) == ["score"]
    assert len(deeplake_indra_ds) == 100
    for i in range(100, 1):
        assert deeplake_indra_ds.score[100 - i].numpy() == [
            np.sqrt(2.0 / (i + 1) / (i + 1))
        ]

    assert list(deeplake_indra_ds.sample_indices) == list(range(100))
    deeplake_indra_ds = deeplake_ds.query(
        "SELECT *, l2_norm(embeddings - ARRAY[0, 0, 0]) as score order by l2_norm(embeddings - ARRAY[0, 0, 0]) asc"
    )
    assert list(deeplake_indra_ds.sample_indices) == list(reversed(range(100)))
    assert list(deeplake_indra_ds.embeddings.sample_indices) == list(
        reversed(range(100))
    )
    assert list(deeplake_indra_ds.score.sample_indices) == list(range(100))

    deeplake_indra_ds = deeplake_ds.query(
        "SELECT l2_norm(embeddings - ARRAY[0, 0, 0]) as score order by l2_norm(embeddings - ARRAY[0, 0, 0]) asc"
    )
    assert list(deeplake_indra_ds.sample_indices) == list(range(100))
    assert list(deeplake_indra_ds.score.sample_indices) == list(range(100))
