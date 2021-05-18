import os
from uuid import uuid1

SESSION_ID = str(uuid1())


def current_test_name(with_id: bool, is_id_prefix: bool) -> str:
    full_name = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
    test_file = full_name.split("::")[0].split("/")[-1].split(".py")[0]
    test_name = full_name.split("::")[1]
    output = os.path.join(test_file, test_name)
    if with_id:
        if is_id_prefix:
            return os.path.join(SESSION_ID, output)
        return os.path.join(output, SESSION_ID)
    return output
