import deeplake as dp
import numpy as np
from deeplake.tests.common import requires_libdeeplake
from deeplake.core.dataset import DeepLakeQueryDataset, DeepLakeQueryTensor
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
    np.isclose(
        deeplake_indra_ds.images.numpy(),
        np.dstack(indra_ds.tensors[0][:]).transpose(2, 0, 1),
    )
    np.isclose(
        deeplake_indra_ds.images[5:55].numpy(),
        np.dstack(indra_ds.tensors[0][5:55]).transpose(2, 0, 1),
    )

    np.isclose(
        deeplake_indra_ds[5:55].images.numpy(),
        np.dstack(indra_ds.tensors[0][5:55]).transpose(2, 0, 1),
    )

    # test int indices
    np.isclose(
        deeplake_indra_ds.images[0].numpy(),
        np.dstack(indra_ds.tensors[0][0]).transpose(2, 0, 1),
    )

    np.isclose(
        deeplake_indra_ds[0].images.numpy(),
        np.dstack(indra_ds.tensors[0][0]).transpose(2, 0, 1),
    )

    # test list indices
    np.isclose(
        deeplake_indra_ds.images[[0, 1]].numpy(),
        np.dstack(indra_ds.tensors[0][[0, 1]]).transpose(2, 0, 1),
    )

    np.isclose(
        deeplake_indra_ds[[0, 1]].images.numpy(),
        np.dstack(indra_ds.tensors[0][[0, 1]]).transpose(2, 0, 1),
    )

    # test tuple indices

    np.isclose(
        deeplake_indra_ds[(0, 1)].images.numpy(),
        np.dstack(indra_ds.tensors[0][(0, 1)]).transpose(2, 0, 1),
    )

    np.isclose(
        deeplake_indra_ds[(0, 1)].images.numpy(),
        np.dstack(indra_ds.tensors[0][(0, 1)]).transpose(2, 0, 1),
    )


@requires_libdeeplake
def test_save_view(local_ds_generator):
    from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

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
        for i in range(1000):
            deeplake_ds.label.append(int(100 * random.uniform(0.0, 1.0)))

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

    assert np.isclose(indra_ds.tensors[0][:], deeplake_indra_ds.image.numpy())


@requires_libdeeplake
def test_query():
    # can we run queries sequentially?
    pass


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

    assert np.isclose(
        deeplake_indra_ds.images.numpy(), deeplake_indra_ds["images"].numpy()
    )

    # test with hirarcy
    # some code goes here

    deeplake_indra_ds_data = deeplake_indra_ds.labels[0].data()
    deeplake_indra_ds_value = deeplake_indra_ds_data["value"]
    assert isinstance(deeplake_indra_ds_value, dict)
    assert len(deeplake_indra_ds_value) == 100
    assert isinstance(deeplake_indra_ds_value, list)
