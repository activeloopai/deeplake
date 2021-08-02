import hub
import pytest
import numpy as np
from click.testing import CliRunner
from hub.core.storage.memory import MemoryProvider
from hub.util.remove_cache import remove_memory_cache
from hub.tests.dataset_fixtures import enabled_datasets
from hub.util.exceptions import InvalidOutputDatasetError

all_compressions = pytest.mark.parametrize("sample_compression", [None, "png", "jpeg"])


@hub.compute
def fn1(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((337, 200)) * sample_in * mul)
        samples_out.label.append(np.ones((1,)) * sample_in * mul)


@hub.compute
def fn2(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy() * mul)
        samples_out.label.append(sample_in.label.numpy() * mul)


@hub.compute
def fn3(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((1310, 2087)) * sample_in * mul)
        samples_out.label.append(np.ones((13,)) * sample_in * mul)


@hub.compute
def read_image(sample_in, samples_out):
    samples_out.image.append(hub.read(sample_in))


@hub.compute
def crop_image(sample_in, samples_out, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy()[:100, :100, :])


@enabled_datasets
def test_single_transform_hub_dataset(ds):
    with CliRunner().isolated_filesystem():
        with hub.dataset("./test/transform_hub_in_generic") as data_in:
            data_in.create_tensor("image")
            data_in.create_tensor("label")
            for i in range(1, 100):
                data_in.image.append(i * np.ones((i, i)))
                data_in.label.append(i * np.ones((1,)))
        data_in = hub.dataset("./test/transform_hub_in_generic")
        ds_out = ds
        ds_out.create_tensor("image")
        ds_out.create_tensor("label")

        fn2(copy=1, mul=2).eval(data_in, ds_out, num_workers=5)
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)


@enabled_datasets
def test_single_transform_hub_dataset_htypes(ds):
    with CliRunner().isolated_filesystem():
        with hub.dataset("./test/transform_hub_in_htypes") as data_in:
            data_in.create_tensor("image", htype="image", sample_compression="png")
            data_in.create_tensor("label", htype="class_label")
            for i in range(1, 100):
                data_in.image.append(i * np.ones((i, i), dtype="uint8"))
                data_in.label.append(i * np.ones((1,), dtype="uint32"))
        data_in = hub.dataset("./test/transform_hub_in_htypes")
        ds_out = ds
        ds_out.create_tensor("image")
        ds_out.create_tensor("label")
        fn2(copy=1, mul=2).eval(data_in, ds_out, num_workers=5)
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)


@enabled_datasets
def test_chain_transform_list_small(ds):
    ls = [i for i in range(100)]
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = hub.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(ls, ds_out, num_workers=5)
    assert len(ds_out) == 600
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )


@enabled_datasets
@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_chain_transform_list_big(ds):
    ls = [i for i in range(2)]
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = hub.compose([fn3(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(ls, ds_out, num_workers=3)
    assert len(ds_out) == 8
    for i in range(2):
        for index in range(4 * i, 4 * i + 4):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((1310, 2087))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((13,))
            )


@enabled_datasets
def test_chain_transform_list_small_processed(ds):
    ls = list(range(100))
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    if isinstance(remove_memory_cache(ds.storage), MemoryProvider):
        with pytest.raises(InvalidOutputDatasetError):
            fn2().eval(ls, ds_out, num_workers=3, scheduler="processed")
        return

    pipeline = hub.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(ls, ds_out, num_workers=3, scheduler="processed")
    assert len(ds_out) == 600
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )


@all_compressions
@enabled_datasets
def test_transform_hub_read(ds, cat_path, sample_compression):
    data_in = [cat_path] * 10
    ds_out = ds
    ds_out.create_tensor("image", htype="image", sample_compression=sample_compression)

    read_image().eval(data_in, ds_out, num_workers=8)
    assert len(ds_out) == 10
    for i in range(10):
        assert ds_out.image[i].numpy().shape == (900, 900, 3)
        np.testing.assert_array_equal(ds_out.image[i].numpy(), ds_out.image[0].numpy())


@all_compressions
@enabled_datasets
def test_transform_hub_read_pipeline(ds, cat_path, sample_compression):
    data_in = [cat_path] * 10
    ds_out = ds
    ds_out.create_tensor("image", htype="image", sample_compression=sample_compression)
    pipeline = hub.compose([read_image(), crop_image(copy=2)])
    pipeline.eval(data_in, ds_out, num_workers=8)
    assert len(ds_out) == 20
    for i in range(20):
        assert ds_out.image[i].numpy().shape == (100, 100, 3)
        np.testing.assert_array_equal(ds_out.image[i].numpy(), ds_out.image[0].numpy())


@enabled_datasets
def test_hub_like(ds):
    with CliRunner().isolated_filesystem():
        data_in = ds
        data_in.create_tensor("image", htype="image", sample_compression="png")
        data_in.create_tensor("label", htype="class_label")
        for i in range(1, 100):
            data_in.image.append(i * np.ones((i, i), dtype="uint8"))
            data_in.label.append(i * np.ones((1,), dtype="uint32"))
        ds_out = hub.like("./test/transform_hub_like", ds)
        fn2(copy=1, mul=2).eval(data_in, ds_out, num_workers=5)
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)
