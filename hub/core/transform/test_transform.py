from hub.api.dataset import Dataset
from hub import transform  # type: ignore
import numpy as np
from hub.core.tests.common import parametrize_all_dataset_storages


def fn1(i, mul=1, copy=1):
    d = {}
    d["image"] = np.ones((337, 200)) * i * mul
    d["label"] = np.ones((1,)) * i * mul
    return [d for _ in range(copy)]


def fn2(sample, mul=1, copy=1):
    d = {}
    d["image"] = sample["image"] * mul
    d["label"] = sample["label"] * mul
    return [d for _ in range(copy)]


def fn3(i, mul=1, copy=1):
    d = {}
    d["image"] = np.ones((1310, 2087)) * i * mul
    d["label"] = np.ones((13,)) * i * mul
    return [d for _ in range(copy)]


@parametrize_all_dataset_storages
def test_single_transform_hub_dataset(ds):
    with Dataset("./test/transform_hub_in") as data_in:
        data_in.create_tensor("image")
        data_in.create_tensor("label")
        for i in range(100):
            data_in.image.append(i * np.ones((100, 100)))
            data_in.label.append(i * np.ones((1,)))
    data_in = Dataset("./test/transform_hub_in")
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    transform(data_in, [fn2], ds_out)
    data_in.delete()
    assert len(ds_out) == 100
    for index in range(100):
        np.testing.assert_array_equal(
            ds_out[index].image.numpy(), index * np.ones((100, 100))
        )
        np.testing.assert_array_equal(
            ds_out[index].label.numpy(), index * np.ones((1,))
        )


@parametrize_all_dataset_storages
def test_chain_transform_list_small(ds):
    ls = [i for i in range(100)]
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    transform(
        ls,
        [fn1, fn2],
        ds_out,
        workers=1,
        pipeline_kwargs=[{"mul": 5, "copy": 2}, {"mul": 3, "copy": 3}],
    )
    assert len(ds_out) == 600
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )


@parametrize_all_dataset_storages
def test_chain_transform_list_big(ds):
    ls = [i for i in range(2)]
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    transform(
        ls,
        [fn3, fn2],
        ds_out,
        workers=3,
        pipeline_kwargs=[{"mul": 5, "copy": 2}, {"mul": 3, "copy": 2}],
    )
    assert len(ds_out) == 8
    for i in range(2):
        for index in range(4 * i, 4 * i + 4):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((1310, 2087))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((13,))
            )
