import pytest

from deeplake.util.tensor_db import parse_runtime_parameters


def test_runtime_parameter_parsing():
    path = "hub://organization/dataset"
    runtime = None

    assert parse_runtime_parameters(path, runtime) == {"tensor_db": False}

    runtime = {"tensor_db": True}
    assert parse_runtime_parameters(path, runtime) == {"tensor_db": True}

    runtime = {"tensor_db": False}
    assert parse_runtime_parameters(path, runtime) == {"tensor_db": False}

    runtime = {"some_unsupported_parameter": True}
    with pytest.raises(ValueError):
        parse_runtime_parameters(path, runtime) == {"tensor_db": False}

    runtime = {"tensor_db": True, "some_unsupported_parameter": True}
    with pytest.raises(ValueError):
        parse_runtime_parameters(path, runtime) == {"tensor_db": True}

    path = "s3://bucket/organization/dataset"
    runtime = {"tensor_db": True}
    with pytest.raises(ValueError):
        parse_runtime_parameters(path, runtime)
