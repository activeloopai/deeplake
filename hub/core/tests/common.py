import pytest
import os
from uuid import uuid1

parametrize_all_storage_providers = pytest.mark.parametrize(
    "storage", ["memory", "local", "s3"], indirect=True
)


def current_test_name(with_uuid=False):
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    if with_uuid:
        return os.path.join(test_file, str(uuid1()))
    return test_file
