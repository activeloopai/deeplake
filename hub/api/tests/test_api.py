from hub.constants import UNCOMPRESSED
import numpy as np
import pytest
import uuid
import hub
import os
from hub.api.dataset import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages
from hub.tests.common import assert_array_lists_equal
from hub.util.exceptions import TensorDtypeMismatchError, TensorInvalidSampleShapeError
from hub.client.client import HubBackendClient
from hub.client.utils import has_hub_testing_creds
from click.testing import CliRunner


# need this for 32-bit and 64-bit systems to have correct tests
MAX_INT_DTYPE = np.int_.__name__
MAX_FLOAT_DTYPE = np.float_.__name__


def test_persist_local(local_storage):
    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 224, 224, 3)))

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    assert ds_new.image.shape == (4, 224, 224, 3)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == hub.__version__

    ds.delete()


def test_persist_with_local(local_storage):
    with Dataset(local_storage.root, local_cache_size=512) as ds:
        ds.create_tensor("image")
        ds.image.extend(np.ones((4, 224, 224, 3)))

        ds_new = Dataset(local_storage.root)
        assert len(ds_new) == 0  # shouldn't be flushed yet

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    engine = ds_new.image.chunk_engine
    assert engine.chunk_id_encoder.num_samples == ds_new.image.meta.length
    assert engine.chunk_id_encoder.num_chunks == 1

    assert ds_new.image.shape == (4, 224, 224, 3)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == hub.__version__

    ds.delete()


def test_persist_local_clear_cache(local_storage):
    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 224, 224, 3)))
    ds.clear_cache()
    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    assert ds_new.image.shape == (4, 224, 224, 3)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))
    ds.delete()


@parametrize_all_dataset_storages
def test_populate_dataset(ds):
    assert ds.meta.tensors == []
    ds.create_tensor("image")
    assert len(ds) == 0
    assert len(ds.image) == 0

    ds.image.extend(np.ones((4, 28, 28)))
    assert len(ds) == 4
    assert len(ds.image) == 4

    for _ in range(10):
        ds.image.append(np.ones((28, 28)))
    assert len(ds.image) == 14

    ds.image.extend([np.ones((28, 28)), np.ones((28, 28))])
    assert len(ds.image) == 16

    assert ds.meta.tensors == [
        "image",
    ]
    assert ds.meta.version == hub.__version__


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_larger_data_memory(memory_storage):
    ds = Dataset(memory_storage.root)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4096, 4096)))
    assert len(ds) == 4
    assert ds.image.shape == (4, 4096, 4096)
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((4, 4096, 4096)))


def test_stringify(memory_ds):
    ds = memory_ds
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4)))
    assert (
        str(ds)
        == "Dataset(path='mem://hub_pytest/test_api/test_stringify', tensors=['image'])"
    )
    assert (
        str(ds[1:2])
        == "Dataset(path='mem://hub_pytest/test_api/test_stringify', index=Index([slice(1, 2, None)]), tensors=['image'])"
    )
    assert str(ds.image) == "Tensor(key='image')"
    assert str(ds[1:2].image) == "Tensor(key='image', index=Index([slice(1, 2, None)]))"


def test_stringify_with_path(local_ds):
    ds = local_ds
    assert local_ds.path
    assert str(ds) == f"Dataset(path='{local_ds.path}', tensors=[])"


@parametrize_all_dataset_storages
def test_compute_fixed_tensor(ds):
    ds.create_tensor("image")
    ds.image.extend(np.ones((32, 28, 28)))
    assert len(ds) == 32
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((32, 28, 28)))


@parametrize_all_dataset_storages
def test_compute_dynamic_tensor(ds):
    ds.create_tensor("image")

    a1 = np.ones((32, 28, 28))
    a2 = np.ones((10, 36, 11))
    a3 = np.ones((29, 10))

    image = ds.image

    image.extend(a1)
    image.extend(a2)
    image.append(a3)

    expected_list = [*a1, *a2, a3]
    actual_list = image.numpy(aslist=True)

    assert type(actual_list) == list
    assert_array_lists_equal(expected_list, actual_list)

    # test negative indexing
    np.testing.assert_array_equal(expected_list[-1], image[-1].numpy())
    np.testing.assert_array_equal(expected_list[-2], image[-2].numpy())
    assert_array_lists_equal(expected_list[-2:], image[-2:].numpy(aslist=True))
    assert_array_lists_equal(expected_list[::-3], image[::-3].numpy(aslist=True))

    assert image.shape == (43, None, None)
    assert image.shape_interval.lower == (43, 28, 10)
    assert image.shape_interval.upper == (43, 36, 28)
    assert image.is_dynamic


@parametrize_all_dataset_storages
def test_empty_samples(ds: Dataset):
    tensor = ds.create_tensor("with_empty")

    a1 = np.arange(25 * 4 * 2).reshape(25, 4, 2)
    a2 = np.arange(5 * 10 * 50 * 2).reshape(5, 10, 50, 2)
    a3 = np.arange(0).reshape(0, 0, 2)
    a4 = np.arange(0).reshape(9, 0, 10, 2)

    tensor.append(a1)
    tensor.extend(a2)
    tensor.append(a3)
    tensor.extend(a4)

    actual_list = tensor.numpy(aslist=True)
    expected_list = [a1, *a2, a3, *a4]

    assert tensor.meta.sample_compression == UNCOMPRESSED
    assert tensor.meta.chunk_compression == UNCOMPRESSED

    assert len(tensor) == 16
    assert tensor.shape_interval.lower == (16, 0, 0, 2)
    assert tensor.shape_interval.upper == (16, 25, 50, 2)

    assert_array_lists_equal(actual_list, expected_list)

    # test indexing individual empty samples with numpy while looping, this may seem redundant but this was failing before
    for actual_sample, expected in zip(ds, expected_list):
        actual = actual_sample.with_empty.numpy()
        np.testing.assert_array_equal(actual, expected)


@parametrize_all_dataset_storages
def test_scalar_samples(ds: Dataset):
    tensor = ds.create_tensor("scalars")

    assert tensor.meta.dtype == None

    # first sample sets dtype
    tensor.append(5)
    assert tensor.meta.dtype == MAX_INT_DTYPE

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(5.1)

    tensor.append(10)
    tensor.append(-99)
    tensor.append(np.array(4))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(np.int16(4))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(np.float32(4))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(np.uint8(3))

    tensor.extend([10, 1, 4])
    tensor.extend([1])
    tensor.extend(np.array([1, 2, 3], dtype=MAX_INT_DTYPE))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.extend(np.array([4, 5, 33], dtype="int16"))

    assert len(tensor) == 11

    expected = np.array([5, 10, -99, 4, 10, 1, 4, 1, 1, 2, 3])
    np.testing.assert_array_equal(tensor.numpy(), expected)

    assert tensor.numpy(aslist=True) == expected.tolist()

    assert tensor.shape == (11,)

    # len(shape) for a scalar is `()`. len(shape) for [1] is `(1,)`
    with pytest.raises(TensorInvalidSampleShapeError):
        tensor.append([1])

    # len(shape) for a scalar is `()`. len(shape) for [1, 2] is `(2,)`
    with pytest.raises(TensorInvalidSampleShapeError):
        tensor.append([1, 2])


@parametrize_all_dataset_storages
def test_sequence_samples(ds: Dataset):
    tensor = ds.create_tensor("arrays")

    tensor.append([1, 2, 3])
    tensor.extend([[4, 5, 6]])

    assert len(tensor) == 2

    expected = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(tensor.numpy(), expected)

    assert type(tensor.numpy(aslist=True)) == list
    assert_array_lists_equal(tensor.numpy(aslist=True), expected)


@parametrize_all_dataset_storages
def test_iterate_dataset(ds):
    labels = [1, 9, 7, 4]
    ds.create_tensor("image")
    ds.create_tensor("label")

    ds.image.extend(np.ones((4, 28, 28)))
    ds.label.extend(np.asarray(labels).reshape((4, 1)))

    for idx, sub_ds in enumerate(ds):
        img = sub_ds.image.numpy()
        label = sub_ds.label.numpy()
        np.testing.assert_array_equal(img, np.ones((28, 28)))
        assert label.shape == (1,)
        assert label == labels[idx]


def _check_tensor(tensor, data):
    np.testing.assert_array_equal(tensor.numpy(), data)


def test_compute_slices(memory_ds):
    ds = memory_ds
    shape = (64, 16, 16, 16)
    data = np.arange(np.prod(shape)).reshape(shape)
    ds.create_tensor("data")
    ds.data.extend(data)

    _check_tensor(ds.data[:], data[:])
    _check_tensor(ds.data[10:20], data[10:20])
    _check_tensor(ds.data[5], data[5])
    _check_tensor(ds.data[0][:], data[0][:])
    _check_tensor(ds.data[-1][:], data[-1][:])
    _check_tensor(ds.data[3, 3], data[3, 3])
    _check_tensor(ds.data[30:40, :, 8:11, 4], data[30:40, :, 8:11, 4])
    _check_tensor(ds.data[16, 4, 5, 1:3], data[16, 4, 5, 1:3])
    _check_tensor(ds[[0, 1, 2, 5, 6, 10, 60]].data, data[[0, 1, 2, 5, 6, 10, 60]])
    _check_tensor(ds.data[[0, 1, -2, 5, -6, 10, 60]], data[[0, 1, -2, 5, -6, 10, 60]])
    _check_tensor(ds.data[0][[0, 1, 2, 5, 6, 10, 15]], data[0][[0, 1, 2, 5, 6, 10, 15]])
    _check_tensor(ds.data[[3, 2, 1, 0]][0], data[[3, 2, 1, 0]][0])
    _check_tensor(ds[[3, 2, 1, 0]][0].data, data[[3, 2, 1, 0]][0])
    _check_tensor(ds[[3, 2, 1, 0]].data[0], data[3])
    _check_tensor(ds[(0, 1, 6, 10, 15), :].data, data[(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[(0, 1, 6, 10, 15), :], data[(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[0][(0, 1, 6, 10, 15), :], data[0][(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[0, (0, 1, 5)], data[0, (0, 1, 5)])
    _check_tensor(ds.data[:, :][0], data[:, :][0])
    _check_tensor(ds.data[:, :][0:2], data[:, :][0:2])
    _check_tensor(ds.data[0, :][0:2], data[0, :][0:2])
    _check_tensor(ds.data[:, 0][0:2], data[:, 0][0:2])
    _check_tensor(ds.data[:, 0][0:2], data[:, 0][0:2])
    _check_tensor(ds.data[:, :][0][(0, 1, 2), 0], data[:, :][0][(0, 1, 2), 0])
    _check_tensor(ds.data[0][(0, 1, 2), 0][1], data[0][(0, 1, 2), 0][1])
    _check_tensor(ds.data[:, :][0][(0, 1, 2), 0][1], data[:, :][0][(0, 1, 2), 0][1])
    _check_tensor(ds.data[::-1], data[::-1])
    _check_tensor(ds.data[::-3], data[::-3])
    _check_tensor(ds.data[::-3][4], data[::-3][4])
    _check_tensor(ds.data[-2:], data[-2:])
    _check_tensor(ds.data[-6:][3], data[-6:][3])
    _check_tensor(ds.data[:-6:][3], data[:-6:][3])


def test_length_slices(memory_ds):
    ds = memory_ds
    data = np.array([1, 2, 3, 9, 8, 7, 100, 99, 98, 99, 101])
    ds.create_tensor("data")
    ds.data.extend(data)

    assert len(ds) == 11
    assert len(ds[0]) == 1
    assert len(ds[0:1]) == 1
    assert len(ds[0:0]) == 0
    assert len(ds[1:10]) == 9
    assert len(ds[1:7:2]) == 3
    assert len(ds[1:8:2]) == 4
    assert len(ds[1:9:2]) == 4
    assert len(ds[1:10:2]) == 5
    assert len(ds[[0, 1, 5, 9]]) == 4

    assert len(ds.data) == 11
    assert len(ds.data[0]) == 1
    assert len(ds.data[0:1]) == 1
    assert len(ds.data[0:0]) == 0
    assert len(ds.data[1:10]) == 9
    assert len(ds.data[1:7:2]) == 3
    assert len(ds.data[1:8:2]) == 4
    assert len(ds.data[1:9:2]) == 4
    assert len(ds.data[1:10:2]) == 5
    assert len(ds.data[[0, 1, 5, 9]]) == 4

    assert ds.data.shape == (11,)
    assert ds[0:5].data.shape == (5,)
    assert ds.data[1:6].shape == (5,)


def test_shape_property(memory_ds):
    fixed = memory_ds.create_tensor("fixed_tensor")
    dynamic = memory_ds.create_tensor("dynamic_tensor")

    # dynamic shape property
    dynamic.extend(np.ones((32, 28, 20, 2)))
    dynamic.extend(np.ones((16, 33, 20, 5)))
    assert dynamic.shape == (48, None, 20, None)
    assert dynamic.shape_interval.lower == (48, 28, 20, 2)
    assert dynamic.shape_interval.upper == (48, 33, 20, 5)
    assert dynamic.is_dynamic

    # fixed shape property
    fixed.extend(np.ones((9, 28, 28)))
    fixed.extend(np.ones((13, 28, 28)))
    assert fixed.shape == (22, 28, 28)
    assert fixed.shape_interval.lower == (22, 28, 28)
    assert fixed.shape_interval.upper == (22, 28, 28)
    assert not fixed.is_dynamic


def test_htype(memory_ds: Dataset):
    image = memory_ds.create_tensor("image", htype="image")
    bbox = memory_ds.create_tensor("bbox", htype="bbox")
    label = memory_ds.create_tensor("label", htype="class_label")
    video = memory_ds.create_tensor("video", htype="video")
    bin_mask = memory_ds.create_tensor("bin_mask", htype="binary_mask")
    segment_mask = memory_ds.create_tensor("segment_mask", htype="segment_mask")

    image.append(np.ones((28, 28, 3), dtype=np.uint8))
    bbox.append(np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float32))
    # label.append(5)
    label.append(np.array(5, dtype=np.uint32))
    video.append(np.ones((10, 28, 28, 3), dtype=np.uint8))
    bin_mask.append(np.zeros((28, 28), dtype=np.bool8))
    segment_mask.append(np.ones((28, 28), dtype=np.int32))


def test_dtype(memory_ds: Dataset):
    tensor = memory_ds.create_tensor("tensor")
    dtyped_tensor = memory_ds.create_tensor("dtyped_tensor", dtype="uint8")
    np_dtyped_tensor = memory_ds.create_tensor(
        "np_dtyped_tensor", dtype=MAX_FLOAT_DTYPE
    )
    py_dtyped_tensor = memory_ds.create_tensor("py_dtyped_tensor", dtype=float)

    # .meta.dtype should always be str or None
    assert type(tensor.meta.dtype) == type(None)
    assert type(dtyped_tensor.meta.dtype) == str
    assert type(np_dtyped_tensor.meta.dtype) == str
    assert type(py_dtyped_tensor.meta.dtype) == str

    # .dtype should always be np.dtype or None
    assert type(tensor.dtype) == type(
        None
    ), "An htype with a generic `dtype` should start as None... If this check doesn't exist, float64/float32 may be it's initial type."
    assert dtyped_tensor.dtype == np.uint8
    assert np_dtyped_tensor.dtype == MAX_FLOAT_DTYPE
    assert py_dtyped_tensor.dtype == MAX_FLOAT_DTYPE

    tensor.append(np.ones((10, 10), dtype="float32"))
    dtyped_tensor.append(np.ones((10, 10), dtype="uint8"))
    np_dtyped_tensor.append(np.ones((10, 10), dtype=MAX_FLOAT_DTYPE))
    py_dtyped_tensor.append(np.ones((10, 10), dtype=MAX_FLOAT_DTYPE))

    assert tensor.dtype == np.float32
    assert dtyped_tensor.dtype == np.uint8
    assert np_dtyped_tensor.dtype == MAX_FLOAT_DTYPE
    assert py_dtyped_tensor.dtype == MAX_FLOAT_DTYPE


@pytest.mark.xfail(raises=TensorDtypeMismatchError, strict=True)
def test_dtype_mismatch(memory_ds: Dataset):
    tensor = memory_ds.create_tensor("tensor", dtype="float16")
    tensor.append(np.ones(100, dtype="uint8"))


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_fails_on_wrong_tensor_syntax(memory_ds):
    memory_ds.some_tensor = np.ones((28, 28))


@pytest.mark.skipif(not has_hub_testing_creds(), reason="requires hub credentials")
def test_hub_cloud_dataset():
    username = "testingacc"
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    id = str(uuid.uuid1())

    uri = f"hub://{username}/hub2ds2_{id}"

    client = HubBackendClient()
    token = client.request_auth_token(username, password)

    with Dataset(uri, token=token) as ds:
        ds.create_tensor("image")
        ds.create_tensor("label", htype="class_label")

        for i in range(1000):
            ds.image.append(i * np.ones((100, 100)))
            ds.label.append(np.uint32(i))

    ds = Dataset(uri, token=token)
    for i in range(1000):
        np.testing.assert_array_equal(ds.image[i].numpy(), i * np.ones((100, 100)))
        np.testing.assert_array_equal(ds.label[i].numpy(), np.uint32(i))

    ds.delete()


@parametrize_all_dataset_storages
def test_hub_dataset_suffix_bug(ds):
    # creating dataset with similar name but some suffix removed from end
    ds2 = Dataset(ds.path[:-1])
    ds2.delete()

    
def test_empty_dataset():
    with CliRunner().isolated_filesystem():
        ds = Dataset("test")
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.create_tensor("z")
        ds = Dataset("test")
        assert list(ds.tensors) == ["x", "y", "z"]
