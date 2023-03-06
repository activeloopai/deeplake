import deeplake as dp
import numpy as np
from deeplake.tests.common import requires_libdeeplake
from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
import random


@requires_libdeeplake
def test_indexing(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    assert len(deeplake_indra_ds) == len(indra_ds)
    assert deeplake_indra_ds.__getstate__() == deeplake_ds.__getstate__()

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
def test_save_view(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
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
def test_load_view(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor(
            "image", htype="image", dtype=np.uint8, sample_compression="jpg"
        )
        for i in range(100):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))
            deeplake_ds.image.append(np.random.randint(0, 255, (100, 200, 3), np.uint8))

    deeplake_ds.commit("First")

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    query_str = "select * group by label"
    view = deeplake_ds.query(query_str)
    view_path = view.save_view()
    view_id = view_path.split("/")[-1]
    view = deeplake_ds.load_view(view_id)

    dataloader = view[:3].pytorch()
    iss = []
    for i, batch in enumerate(dataloader):
        iss.append(i)

    assert np.all(indra_ds.image.numpy() == deeplake_indra_ds.image.numpy())


@requires_libdeeplake
def test_query(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
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
def test_copy():
    pass


@requires_libdeeplake
def test_optimize_views():
    pass


@requires_libdeeplake
def test_parallel_computing():
    pass


@requires_libdeeplake
def test_accessing_data(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    indra_ds = dataset_to_libdeeplake(deeplake_ds)
    deeplake_indra_ds = DeepLakeQueryDataset(deeplake_ds=deeplake_ds, indra_ds=indra_ds)

    assert np.all(
        np.isclose(deeplake_indra_ds.label.numpy(), deeplake_indra_ds["label"].numpy())
    )
