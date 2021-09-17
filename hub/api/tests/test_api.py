import os
import numpy as np
import pytest
import hub
from hub.core.dataset import Dataset
from hub.core.tensor import Tensor
from hub.tests.common import assert_array_lists_equal
from hub.util.exceptions import (
    TensorDtypeMismatchError,
    TensorAlreadyExistsError,
    TensorGroupAlreadyExistsError,
    TensorInvalidSampleShapeError,
    DatasetHandlerError,
    UnsupportedCompressionError,
    InvalidTensorNameError,
)
from hub.constants import MB

from click.testing import CliRunner
from hub.tests.dataset_fixtures import (
    enabled_datasets,
    enabled_persistent_dataset_generators,
    enabled_non_gcs_datasets,
)


# need this for 32-bit and 64-bit systems to have correct tests
MAX_INT_DTYPE = np.int_.__name__
MAX_FLOAT_DTYPE = np.float_.__name__


# not using the predefined parametrizes because `hub_cloud_ds_generator` is not enabled by default
@pytest.mark.parametrize(
    "ds_generator",
    [
        "local_ds_generator",
        "s3_ds_generator",
        "hub_cloud_ds_generator",
    ],
    indirect=True,
)
def test_persist(ds_generator):
    ds = ds_generator()

    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 224, 224, 3)))

    ds_new = ds_generator()
    assert len(ds_new) == 4

    assert ds_new.image.shape == (4, 224, 224, 3)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == hub.__version__


@enabled_persistent_dataset_generators
def test_persist_with(ds_generator):
    with ds_generator() as ds:
        ds.create_tensor("image")
        ds.image.extend(np.ones((4, 224, 224, 3)))

        ds_new = ds_generator()
        assert len(ds_new) == 0  # shouldn't be flushed yet

    ds_new = ds_generator()
    assert len(ds_new) == 4

    engine = ds_new.image.chunk_engine
    assert engine.chunk_id_encoder.num_samples == ds_new.image.meta.length
    assert engine.chunk_id_encoder.num_chunks == 1

    assert ds_new.image.shape == (4, 224, 224, 3)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == hub.__version__


@enabled_persistent_dataset_generators
def test_persist_clear_cache(ds_generator):
    ds = ds_generator()
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 224, 224, 3)))
    ds.clear_cache()
    ds_new = ds_generator()
    assert len(ds_new) == 4

    assert ds_new.image.shape == (4, 224, 224, 3)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))


@enabled_datasets
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
def test_larger_data_memory(memory_ds):
    memory_ds.create_tensor("image")
    memory_ds.image.extend(np.ones((4, 4096, 4096)))
    assert len(memory_ds) == 4
    assert memory_ds.image.shape == (4, 4096, 4096)
    np.testing.assert_array_equal(memory_ds.image.numpy(), np.ones((4, 4096, 4096)))


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


@enabled_non_gcs_datasets
def test_compute_fixed_tensor(ds):
    ds.create_tensor("image")
    ds.image.extend(np.ones((32, 28, 28)))
    assert len(ds) == 32
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((32, 28, 28)))


@enabled_non_gcs_datasets
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


@enabled_datasets
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

    assert tensor.meta.sample_compression is None

    assert len(tensor) == 16
    assert tensor.shape_interval.lower == (16, 0, 0, 2)
    assert tensor.shape_interval.upper == (16, 25, 50, 2)

    assert_array_lists_equal(actual_list, expected_list)

    # test indexing individual empty samples with numpy while looping, this may seem redundant but this was failing before
    for actual_sample, expected in zip(ds, expected_list):
        actual = actual_sample.with_empty.numpy()
        np.testing.assert_array_equal(actual, expected)


@enabled_non_gcs_datasets
def test_safe_downcasting(ds: Dataset):
    int_tensor = ds.create_tensor("int", dtype="uint8")
    int_tensor.append(0)
    int_tensor.append(1)
    int_tensor.extend([2, 3, 4])
    int_tensor.extend([5, 6, np.uint8(7)])
    with pytest.raises(TensorDtypeMismatchError):
        int_tensor.append(-8)
    int_tensor.append(np.array([1]))
    assert len(int_tensor) == 9
    with pytest.raises(TensorDtypeMismatchError):
        int_tensor.append(np.array([1.0]))

    float_tensor = ds.create_tensor("float", dtype="float32")
    float_tensor.append(0)
    float_tensor.append(1)
    float_tensor.extend([2, 3.0, 4.0])
    float_tensor.extend([5.0, 6.0, np.float32(7.0)])
    with pytest.raises(TensorDtypeMismatchError):
        float_tensor.append(float(np.finfo(np.float32).max + 1))
    float_tensor.append(np.array([1]))
    float_tensor.append(np.array([1.0]))
    assert len(float_tensor) == 10


@enabled_datasets
def test_scalar_samples(ds: Dataset):
    tensor = ds.create_tensor("scalars")

    assert tensor.meta.dtype is None

    # first sample sets dtype
    tensor.append(5)
    assert tensor.meta.dtype == MAX_INT_DTYPE

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(5.1)

    tensor.append(10)
    tensor.append(-99)
    tensor.append(np.array(4))

    tensor.append(np.int16(4))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(np.float32(4))

    tensor.append(np.uint8(3))

    tensor.extend([10, 1, 4])
    tensor.extend([1])
    tensor.extend(np.array([1, 2, 3], dtype=MAX_INT_DTYPE))

    tensor.extend(np.array([4, 5, 33], dtype="int16"))

    assert len(tensor) == 16

    assert tensor.shape == (16, 1)

    tensor.append([1])
    tensor.append([1, 2, 3])
    tensor.extend([[1], [2], [3, 4]])
    tensor.append(np.empty(0, dtype=int))

    with pytest.raises(TensorInvalidSampleShapeError):
        tensor.append([[[1]]])

    expected = [
        [5],
        [10],
        [-99],
        [4],
        [4],
        [3],
        [10],
        [1],
        [4],
        [1],
        [1],
        [2],
        [3],
        [4],
        [5],
        [33],
        [1],
        [1, 2, 3],
        [1],
        [2],
        [3, 4],
        [],
    ]

    assert_array_lists_equal(expected, tensor.numpy(aslist=True))

    assert tensor.shape == (22, None)
    assert tensor.shape_interval.lower == (22, 0)
    assert tensor.shape_interval.upper == (22, 3)

    assert len(tensor) == 22


@enabled_datasets
def test_sequence_samples(ds: Dataset):
    tensor = ds.create_tensor("arrays")

    tensor.append([1, 2, 3])
    tensor.extend([[4, 5, 6]])
    ds.clear_cache()

    assert len(tensor) == 2

    expected = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(tensor.numpy(), expected)

    assert type(tensor.numpy(aslist=True)) == list
    assert_array_lists_equal(tensor.numpy(aslist=True), expected)


@enabled_datasets
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

    assert ds.data.shape == (11, 1)
    assert ds[0:5].data.shape == (5, 1)
    assert ds.data[1:6].shape == (5, 1)


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
    image = memory_ds.create_tensor("image", htype="image", sample_compression="png")
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

    # test auto upcasting
    np_dtyped_tensor.append(np.ones((10, 10), dtype="float32"))
    py_dtyped_tensor.append(np.ones((10, 10), dtype="float32"))

    with pytest.raises(TensorDtypeMismatchError):
        tensor.append(np.ones((10, 10), dtype="float64"))

    with pytest.raises(TensorDtypeMismatchError):
        dtyped_tensor.append(np.ones((10, 10), dtype="uint64") * 256)

    assert tensor.dtype == np.float32
    assert dtyped_tensor.dtype == np.uint8
    assert np_dtyped_tensor.dtype == MAX_FLOAT_DTYPE
    assert py_dtyped_tensor.dtype == MAX_FLOAT_DTYPE

    assert len(tensor) == 1
    assert len(dtyped_tensor) == 1


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_fails_on_wrong_tensor_syntax(memory_ds):
    memory_ds.some_tensor = np.ones((28, 28))


def test_array_interface(memory_ds: Dataset):
    tensor = memory_ds.create_tensor("tensor")
    x = np.arange(10).reshape(5, 2)
    tensor.append(x)
    arr1 = np.array(tensor)
    arr2 = np.array(tensor)
    np.testing.assert_array_equal(x, arr1[0])
    np.testing.assert_array_equal(x, arr2[0])
    tensor.append(x)
    np.testing.assert_array_equal(tensor.numpy(), np.concatenate([arr1, arr2]))


def test_hub_dataset_suffix_bug(hub_cloud_ds, hub_cloud_dev_token):
    # creating dataset with similar name but some suffix removed from end
    ds = hub.dataset(hub_cloud_ds.path[:-1], token=hub_cloud_dev_token)

    # need to delete because it's a different path (won't be auto cleaned up)
    ds.delete()


def test_index_range(memory_ds):
    with pytest.raises(ValueError):
        memory_ds[0]

    memory_ds.create_tensor("label")

    with pytest.raises(ValueError):
        memory_ds.label[0]

    memory_ds.label.extend([5, 6, 7])
    assert len(memory_ds) == 3
    assert len(memory_ds.label) == 3

    for valid_idx in [0, 1, 2, -0, -1, -2, -3]:
        memory_ds[valid_idx]
        memory_ds.label[valid_idx]

    for invalid_idx in [3, 4, -4, -5]:
        with pytest.raises(ValueError):
            memory_ds[invalid_idx]
        with pytest.raises(ValueError):
            memory_ds.label[invalid_idx]

    memory_ds[[0, 1, 2]]
    with pytest.raises(ValueError):
        memory_ds[[0, 1, 2, 3, 4, 5]]


def test_empty_dataset():
    with CliRunner().isolated_filesystem():
        ds = hub.dataset("test")
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.create_tensor("z")
        ds = hub.dataset("test")
        assert list(ds.tensors) == ["x", "y", "z"]


def test_like(local_path):
    src_path = os.path.join(local_path, "src")
    dest_path = os.path.join(local_path, "dest")

    src_ds = hub.dataset(src_path)
    src_ds.info.update(key=0)

    src_ds.create_tensor("a", htype="image", sample_compression="png")
    src_ds.create_tensor("b", htype="class_label")
    src_ds.create_tensor("c")
    src_ds.create_tensor("d", dtype=bool)

    src_ds.d.info.update(key=1)

    dest_ds = hub.like(dest_path, src_ds)

    assert tuple(dest_ds.tensors.keys()) == ("a", "b", "c", "d")

    assert dest_ds.a.meta.htype == "image"
    assert dest_ds.a.meta.sample_compression == "png"
    assert dest_ds.b.meta.htype == "class_label"
    assert dest_ds.c.meta.htype == "generic"
    assert dest_ds.d.dtype == bool

    assert dest_ds.info.key == 0
    assert dest_ds.d.info.key == 1

    assert len(dest_ds) == 0


def test_tensor_creation_fail_recovery():
    with CliRunner().isolated_filesystem():
        ds = hub.dataset("test")
        with ds:
            ds.create_tensor("x")
            ds.create_tensor("y")
            with pytest.raises(UnsupportedCompressionError):
                ds.create_tensor("z", sample_compression="something_random")
        ds = hub.dataset("test")
        assert list(ds.tensors) == ["x", "y"]
        ds.create_tensor("z")
        assert list(ds.tensors) == ["x", "y", "z"]


def test_dataset_delete():
    with CliRunner().isolated_filesystem():
        os.mkdir("test")
        with open("test/test.txt", "w") as f:
            f.write("some data")

        with pytest.raises(DatasetHandlerError):
            # Can't delete raw data without force
            hub.dataset.delete("test/")

        hub.dataset.delete("test/", force=True)
        assert not os.path.isfile("test/test.txt")

        hub.empty("test/").create_tensor("tmp")
        assert os.path.isfile("test/dataset_meta.json")

        hub.dataset.delete("test/")
        assert not os.path.isfile("test/dataset_meta.json")

        old_size = hub.constants.DELETE_SAFETY_SIZE
        hub.constants.DELETE_SAFETY_SIZE = 1 * MB

        ds = hub.empty("test/")
        ds.create_tensor("data")
        ds.data.extend(np.zeros((100, 2000)))

        try:
            hub.dataset.delete("test/")
        finally:
            assert os.path.isfile("test/dataset_meta.json")

        hub.dataset.delete("test/", large_ok=True)
        assert not os.path.isfile("test/dataset_meta.json")

        hub.constants.DELETE_SAFETY_SIZE = old_size


def test_invalid_tensor_name(memory_ds):
    with pytest.raises(InvalidTensorNameError):
        memory_ds.create_tensor("version_state")
    with pytest.raises(InvalidTensorNameError):
        memory_ds.create_tensor("info")


def test_compressions_list():
    assert hub.compressions == [
        "bmp",
        "dib",
        "gif",
        "ico",
        "jpeg",
        "jpeg2000",
        "lz4",
        "pcx",
        "png",
        "ppm",
        "sgi",
        "tga",
        "tiff",
        "webp",
        "wmf",
        "xbm",
        None,
    ]


def test_htypes_list():
    assert hub.htypes == [
        "generic",
        "image",
        "class_label",
        "bbox",
        "video",
        "binary_mask",
        "segment_mask",
    ]


def test_groups(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("x")
    with pytest.raises(TensorAlreadyExistsError):
        ds.create_tensor("x/y")
    ds.create_tensor("y/x")
    with pytest.raises(TensorGroupAlreadyExistsError):
        ds.create_tensor("y")
    assert isinstance(ds.y, Dataset)
    assert isinstance(ds.x, Tensor)
    assert isinstance(ds.y.x, Tensor)

    assert "x" in ds._ungrouped_tensors

    ds.create_tensor("/z")
    assert "z" in ds.tensors
    assert "" not in ds.groups
    assert "" not in ds.tensors
    assert isinstance(ds.z, Tensor)

    assert list(ds.groups) == ["y"]
    assert set(ds.tensors) == set(["x", "z", "y/x"])
    assert list(ds.y.tensors) == ["x"]
    z = ds.y.create_group("z")
    assert "z" in ds.y.groups

    c = z.create_tensor("a/b/c")
    d = z.a.b.create_group("d")

    c.append(np.zeros((3, 2)))

    e = ds.create_tensor("/y/z//a/b////d/e/")
    e.append(np.ones((4, 3)))

    ds = local_ds_generator()
    c = ds.y.z.a.b.c
    assert ds.y.z.a.b.parent.group_index == ds.y.z.a.group_index
    np.testing.assert_array_equal(c[0].numpy(), np.zeros((3, 2)))
    assert "d" in ds.y.z.a.b.groups
    e = ds.y.z.a.b.d.e
    np.testing.assert_array_equal(e[0].numpy(), np.ones((4, 3)))

    ds.create_group("g")
    ds.g.create_tensor("g")
