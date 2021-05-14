import pytest
import os
from uuid import uuid1
from hub.core.storage import MemoryProvider, LocalProvider, S3Provider


@pytest.fixture
def storage(request):
    store = request.param
    if store == "memory":
        return MemoryProvider("hub2/tests/")
    elif store == "local":
        return LocalProvider("./hub2/tests/")
    elif store == "s3":
        return S3Provider("snark-hub/hub2/tests/")


parametrize_all_storage_providers = pytest.mark.parametrize(
    "storage", ["memory", "local", "s3"], indirect=True
)


def current_test_name(with_uuid=False):
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    if with_uuid:
        return os.path.join(test_file, str(uuid1()))
    return test_file
