import os
from uuid import uuid1

import pytest

ALL_PROVIDERS = ["memory", "local", "s3"]
ALL_CACHES = ["memory,local", "memory,s3", "local,s3", "memory,local,s3"]


parametrize_all_storages = pytest.mark.parametrize(
    "storage",
    ALL_PROVIDERS,
    indirect=True,
)

parametrize_all_caches = pytest.mark.parametrize(
    "storage",  # caches are used the same as `storage`
    ALL_CACHES,
    indirect=True,
)

parametrize_all_storages_and_caches = pytest.mark.parametrize(
    "storage",
    ALL_PROVIDERS + ALL_CACHES,
    indirect=True,
)


def current_test_name(with_uuid: bool = False) -> str:
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    if with_uuid:
        return os.path.join(test_file, str(uuid1()))
    return test_file
