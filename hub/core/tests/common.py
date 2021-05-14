import pytest
import os
from uuid import uuid1


parametrize_all_storage_providers = pytest.mark.parametrize(
    "storage", ["memory", "local", "s3"], indirect=True
)

parametrize_all_cache = pytest.mark.parametrize(
    "cache", ["memory_local", "memory_s3", "local_s3", "memory_local_s3"], indirect=True
)



def current_test_name(with_uuid=False):
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    if with_uuid:
        return os.path.join(test_file, str(uuid1()))
    return test_file
