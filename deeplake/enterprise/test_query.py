import pytest
from math import floor

import deeplake
from click.testing import CliRunner
from deeplake.cli.auth import login, logout
from deeplake.constants import QUERY_MESSAGE_MAX_SIZE
from deeplake.tests.common import requires_libdeeplake
from deeplake.util.exceptions import EmptyTokenException
import numpy as np


@pytest.mark.slow
@requires_libdeeplake
def test_query(hub_cloud_ds):
    with hub_cloud_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = hub_cloud_ds.query("SELECT * WHERE CONTAINS(label, 2)")
    assert len(dsv) == 20
    for i in range(20):
        assert dsv.label[i].numpy() == 2

    dsv2 = hub_cloud_ds.query("SELECT * WHERE label < 3")
    assert len(dsv2) == 60
    dsv3 = dsv2.query("SELECT * WHERE label > 1")
    assert len(dsv3) == 20


@requires_libdeeplake
def test_query_on_local_datasets(local_ds, hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials
    runner = CliRunner()
    runner.invoke(logout)

    path = local_ds.path
    ds = deeplake.empty(path, overwrite=True)
    ds.create_tensor("label")
    for i in range(100):
        ds.label.append(floor(i / 20))

    with pytest.raises(EmptyTokenException):
        dsv = ds.query("SELECT * WHERE CONTAINS(label, 2)")

    runner.invoke(login, f"-u {username} -p {password}")
    ds = deeplake.empty(path, overwrite=True)
    ds.create_tensor("label")
    for i in range(100):
        ds.label.append(floor(i / 20))
    dsv = ds.query("SELECT * WHERE CONTAINS(label, 2)")
    assert len(dsv) == 20


@requires_libdeeplake
def test_default_query_message(hub_cloud_ds_generator):
    with hub_cloud_ds_generator() as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))
        ds.commit()

    query_string = "SELECT * WHERE CONTAINS(label, 2)"
    dsv = ds.query(query_string)
    dsv.save_view(id="test_1")

    ds = hub_cloud_ds_generator()
    message = ds.get_view("test_1").message
    assert message == query_string

    query_string = "SELECT * WHERE " + " OR ".join(
        [f"CONTAINS(label, {i})" for i in range(100)]
    )
    dsv = ds.query(query_string)
    dsv.save_view(id="test_2")

    ds = hub_cloud_ds_generator()
    message = ds.get_view("test_2").message
    assert message == query_string[: QUERY_MESSAGE_MAX_SIZE - 3] + "..."


@requires_libdeeplake
def test_sample(local_auth_ds):
    with local_auth_ds as ds:
        ds.create_tensor("label")
        for i in range(100):
            ds.label.append(floor(i / 20))

    dsv = local_auth_ds.sample_by(
        "max_weight(label == 2: 10, label == 1: 1)", replace=False, size=10
    )
    assert len(dsv) == 10
    for i in range(10):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = local_auth_ds.sample_by(
        "max_weight(label == 2: 10, label == 1: 1)", replace=True
    )
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 2 or dsv.label[i].numpy() == 1

    dsv = local_auth_ds.sample_by("label")
    assert len(dsv) == 100

    weights = list()
    for i in range(100):
        weights.append(1 if floor(i / 20) == 0 else 0)

    dsv = local_auth_ds.sample_by(weights)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 0

    weights = np.ndarray((100), np.int32)
    for i in range(100):
        weights[i] = 1 if floor(i / 10) == 0 else 0

    dsv = local_auth_ds.sample_by(weights)
    assert len(dsv) == 100
    for i in range(100):
        assert dsv.label[i].numpy() == 0
