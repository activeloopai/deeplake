import pytest
from uuid import uuid1

from hub.core.storage import S3Provider, MappedProvider, LocalProvider


SESSION_UUID = "TEST"  # uuid1()


def _skip_if_false(request, opt, message):
    if not request.config.getoption(opt):
        pytest.skip(message)


def pytest_addoption(parser):
    parser.addoption(
        "--memory", action="store_true", help="run tests with memory storage provider"
    )
    parser.addoption(
        "--local", action="store_true", help="run tests with local storage provider"
    )
    parser.addoption(
        "--s3", action="store_true", help="run tests with s3 storage provider"
    )


@pytest.fixture
def memory_storage(request):
    _skip_if_false(request, "--memory", "Skipping memory storage")
    return MappedProvider()


@pytest.fixture
def local_storage(request):
    _skip_if_false(request, "--local", "Skipping local storage")
    return LocalProvider("pytest_local_provider/%s" % request.node.name)


@pytest.fixture
def s3_storage(request):
    _skip_if_false(request, "--s3", "Skipping s3 storage")
    return S3Provider("snark-test/hub-2.0/%s/%s" % (request.node.name, SESSION_UUID))
