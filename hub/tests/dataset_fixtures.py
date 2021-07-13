import os
import pytest
from hub import Dataset


all_enabled_datasets = pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", "s3_ds"],  # TODO: add hub cloud
    indirect=True,
)


@pytest.fixture
def memory_ds(memory_path):
    return Dataset(memory_path)


@pytest.fixture
def local_ds(local_path):
    return Dataset(local_path)


@pytest.fixture
def local_ds_generator(local_path):
    def generate_local_ds():
        return Dataset(local_path)

    return generate_local_ds


@pytest.fixture
def s3_ds(s3_path):
    return Dataset(s3_path)


@pytest.fixture
def hub_cloud_ds(hub_cloud_path, hub_testing_token):
    return Dataset(hub_cloud_path, token=hub_testing_token)


@pytest.fixture
def ds(request):
    """Used with parametrize to use all enabled dataset fixtures."""
    return request.getfixturevalue(request.param)
