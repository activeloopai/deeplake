import deeplake
import pytest
from deeplake.util.exceptions import DatasetHandlerError
from deeplake.util.access_method import parse_access_method


def test_access_method_parsing():
    assert parse_access_method("download") == ("download", 0, "threaded")
    assert parse_access_method("download:processed") == ("download", 0, "processed")
    assert parse_access_method("download:5") == ("download", 5, "threaded")
    assert parse_access_method("download:processed:5") == ("download", 5, "processed")
    assert parse_access_method("download:5:processed") == ("download", 5, "processed")
    with pytest.raises(ValueError):
        parse_access_method("download:5:processed:5")
    with pytest.raises(ValueError):
        parse_access_method("download:processed:processed")
    with pytest.raises(ValueError):
        parse_access_method("download:processed:5:processed")


def test_access_method(s3_ds_generator):
    with pytest.raises(DatasetHandlerError):
        deeplake.dataset(
            "./some_non_existent_path", access_method="local", download=True
        )

    with pytest.raises(DatasetHandlerError):
        deeplake.dataset("./some_non_existent_path", access_method="local")

    with pytest.raises(DatasetHandlerError):
        deeplake.dataset("./some_non_existent_path", download=True)

    ds = s3_ds_generator()
    with ds:
        ds.create_tensor("x")
        for i in range(10):
            ds.x.append(i)

    ds = s3_ds_generator(access_method="local")
    assert not ds.path.startswith("s3://")
    for i in range(10):
        assert ds.x[i].numpy() == i

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="invalid")

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="local", overwrite=True)

    with pytest.raises(ValueError):
        s3_ds_generator(access_method="local", download=True, overwrite=True)

    with pytest.raises(ValueError):
        s3_ds_generator(download=True, overwrite=True)

    ds = s3_ds_generator(access_method="local")
    assert not ds.path.startswith("s3://")
    for i in range(10):
        assert ds.x[i].numpy() == i

    remote = s3_ds_generator()
    for i in range(10):
        remote.x.append(10 + i)

    ds = s3_ds_generator(download=True)
    assert not ds.path.startswith("s3://")
    for i in range(20):
        ds.x[i].numpy() == i

    ds.delete()
