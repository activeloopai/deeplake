import os
import sys
import time

import numpy as np
import pathlib
import pytest
import deeplake
from deeplake.core.dataset import Dataset
from deeplake.core.tensor import Tensor
from deeplake.tests.common import (
    assert_array_lists_equal,
    is_opt_true,
    get_dummy_data_path,
    requires_libdeeplake,
    requires_torch,
)
from deeplake.tests.storage_fixtures import enabled_remote_storages
from deeplake.core.storage import GCSProvider
from deeplake.util.exceptions import (
    GroupInfoNotSupportedError,
    IncompatibleHtypeError,
    InvalidOperationError,
    SampleAppendError,
    TensorDoesNotExistError,
    TensorAlreadyExistsError,
    TensorGroupDoesNotExistError,
    TensorGroupAlreadyExistsError,
    DatasetHandlerError,
    TransformError,
    UnsupportedCompressionError,
    InvalidTensorNameError,
    InvalidTensorGroupNameError,
    RenameError,
    PathNotEmptyException,
    BadRequestException,
    EmptyTensorError,
    InvalidTokenException,
    TokenPermissionError,
    UserNotLoggedInException,
    SampleAppendingError,
    DatasetTooLargeToDelete,
    InvalidDatasetNameException,
    UnsupportedParameterException,
    DynamicTensorNumpyError,
    SampleExtendError,
)
from deeplake.util.path import convert_string_to_pathlib_if_needed, verify_dataset_name
from deeplake.util.testing import assert_array_equal
from deeplake.util.pretty_print import summary_tensor, summary_dataset
from deeplake.util.shape_interval import ShapeInterval
from deeplake.constants import GDRIVE_OPT, MB, KB

from click.testing import CliRunner
from rich import print as rich_print
from io import BytesIO

import pickle

# need this for 32-bit and 64-bit systems to have correct tests
MAX_INT_DTYPE = np.int_.__name__
MAX_FLOAT_DTYPE = np.float_.__name__


# not using the predefined parametrizes because `hub_cloud_ds_generator` is not enabled by default
@pytest.mark.slow
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
    assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == deeplake.__version__

    ds_new.create_tensor("label")
    ds_new.label.extend([1, 2, 3, 4])

    ds2 = ds_generator()

    assert len(ds2) == 4
    assert_array_equal(ds2.label.numpy(), np.array([[1], [2], [3], [4]]))


def test_persist_keys(local_ds_generator):
    ds = local_ds_generator()

    ds.create_tensor("image")

    ds_new = local_ds_generator()
    assert set(ds_new.storage.keys()) == {
        "dataset_meta.json",
        "image/commit_diff",
        "image/tensor_meta.json",
        "_image_id/tensor_meta.json",
        "_image_id/commit_diff",
        "_image_shape/tensor_meta.json",
        "_image_shape/commit_diff",
        "version_control_info.json",
    }


def test_persist_with(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("image")
        ds.image.extend(np.ones((4, 224, 224, 3)))

        ds_new = local_ds_generator()
        assert len(ds_new) == 0  # shouldn't be flushed yet

    ds_new = local_ds_generator()
    assert len(ds_new) == 4

    engine = ds_new.image.chunk_engine
    assert engine.chunk_id_encoder.num_samples == ds_new.image.meta.length
    assert engine.chunk_id_encoder.num_chunks == 1

    assert ds_new.image.shape == (4, 224, 224, 3)

    assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))

    assert ds_new.meta.version == deeplake.__version__


def test_persist_clear_cache(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 224, 224, 3)))
    ds.clear_cache()
    ds_new = local_ds_generator()
    assert len(ds_new) == 4

    assert ds_new.image.shape == (4, 224, 224, 3)

    assert_array_equal(ds_new.image.numpy(), np.ones((4, 224, 224, 3)))


def test_populate_dataset(local_ds):
    assert local_ds.meta.tensors == []
    local_ds.create_tensor("image")
    assert len(local_ds) == 0
    assert len(local_ds.image) == 0

    local_ds.image.extend(np.ones((4, 28, 28)))
    assert len(local_ds) == 4
    assert len(local_ds.image) == 4

    for _ in range(10):
        local_ds.image.append(np.ones((28, 28)))
    assert len(local_ds.image) == 14

    local_ds.image.extend([np.ones((28, 28)), np.ones((28, 28))])
    assert len(local_ds.image) == 16
    assert len(local_ds.image.numpy()) == 16
    assert len(local_ds.image[0:5].numpy()) == 5
    assert len(local_ds.image[:-1].numpy()) == 15
    assert len(local_ds.image[-5:].numpy()) == 5

    assert local_ds.meta.tensors == ["image", "_image_shape", "_image_id"]
    assert local_ds.meta.version == deeplake.__version__


def test_larger_data_memory(memory_ds):
    memory_ds.create_tensor("image", max_chunk_size=2 * MB)
    x = np.ones((4, 1024, 1024))
    memory_ds.image.extend(x)
    assert len(memory_ds) == 4
    assert memory_ds.image.shape == x.shape
    assert_array_equal(memory_ds.image.numpy(), x)
    idxs = [
        0,
        1,
        3,
        -1,
        slice(0, 3),
        slice(2, 4),
        slice(2, None),
        (0, slice(5, None), slice(None, 714)),
        (2, 100, 1007),
        (slice(1, 3), [20, 1000, 2, 400], [-2, 3, 577, 1023]),
    ]
    for idx in idxs:
        assert_array_equal(memory_ds.image[idx].numpy(), x[idx])


def test_stringify(memory_ds, capsys):
    ds = memory_ds
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4)))

    ds.summary()
    assert (
        capsys.readouterr().out
        == "Dataset(path='mem://hub_pytest/test_api/test_stringify', tensors=['image'])\n\n tensor    htype    shape    dtype  compression\n -------  -------  -------  -------  ------- \n  image   generic  (4, 4)   float64   None   \n"
    )

    ds[1:2].summary()
    assert (
        capsys.readouterr().out
        == "Dataset(path='mem://hub_pytest/test_api/test_stringify', index=Index([slice(1, 2, None)]), tensors=['image'])\n\n tensor    htype    shape    dtype  compression\n -------  -------  -------  -------  ------- \n  image   generic  (1, 4)   float64   None   \n"
    )

    ds.image.summary()
    assert (
        capsys.readouterr().out
        == "Tensor(key='image')\n\n  htype    shape    dtype  compression\n -------  -------  -------  ------- \n generic  (4, 4)   float64   None   \n"
    )
    ds[1:2].image.summary()
    assert (
        capsys.readouterr().out
        == "Tensor(key='image', index=Index([slice(1, 2, None)]))\n\n  htype    shape    dtype  compression\n -------  -------  -------  ------- \n generic  (1, 4)   float64   None   \n"
    )


def test_summary(memory_ds):
    ds = memory_ds
    ds.create_tensor("abc")
    ds.abc.extend(np.ones((4, 4)))
    ds.create_tensor("images", htype="image", dtype="int32", sample_compression="jpeg")

    assert (
        summary_dataset(ds)
        == "\n tensor    htype    shape    dtype  compression\n -------  -------  -------  -------  ------- \n   abc    generic  (4, 4)   float64   None   \n images    image    (0,)     int32    jpeg   "
    )
    assert (
        summary_tensor(ds.abc)
        == "\n  htype    shape    dtype  compression\n -------  -------  -------  ------- \n generic  (4, 4)   float64   None   "
    )
    assert (
        summary_tensor(ds.images)
        == "\n  htype    shape    dtype  compression\n -------  -------  -------  ------- \n  image    (0,)     int32    jpeg   "
    )


def test_view_summary(memory_ds, capsys):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(np.ones((2500, 10, 5, 3)))
        ds.abc.extend(np.ones((5000, 5, 5, 3)))
        ds.abc.extend(np.ones((5000, 5, 10, 3)))

        with pytest.raises(ValueError):
            ds[2500:].summary()

        ds[2500:].summary(force=True)
        captured = capsys.readouterr().out
        assert (
            captured
            == f"{str(ds[2500:])}\n\n tensor    htype          shape          dtype  compression\n -------  -------        -------        -------  ------- \n   abc    generic  (10000, 5, 5:10, 3)  float64   None   \n"
        )

        assert (
            summary_dataset(ds[:7500])
            == "\n tensor    htype         shape          dtype  compression\n -------  -------       -------        -------  ------- \n   abc    generic  (7500, 5:10, 5, 3)  float64   None   "
        )


def test_log(memory_ds, capsys):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend([1, 2, 3, 4])

        header = "---------------\nDeep Lake Version Log\n---------------\n\n"
        current_branch = "Current Branch: main\n"
        uncommitted_changes = "** There are uncommitted changes on this branch.\n"

        log = header + current_branch + uncommitted_changes
        ds.log()
        captured = capsys.readouterr().out
        assert captured.strip() == log.strip()

        ds.commit("init")
        ds.checkout("alt", create=True)
        commit1 = "\n" + str(ds.version_state["commit_node"].parent) + "\n"
        current_branch = "Current Branch: alt\n"
        log = header + current_branch + commit1
        ds.log()
        captured = capsys.readouterr().out
        assert captured.strip() == log.strip()

        ds.abc.extend([5, 6, 7, 8])
        log = header + current_branch + uncommitted_changes + commit1
        ds.log()
        captured = capsys.readouterr().out
        assert captured.strip() == log.strip()

        ds.commit("update")
        commit2 = "\n" + str(ds.version_state["commit_node"].parent) + "\n"
        log = header + current_branch + commit2 + commit1
        ds.log()
        captured = capsys.readouterr().out
        assert captured.strip() == log.strip()


def test_stringify_with_path(local_ds, capsys):
    ds = local_ds
    assert local_ds.path
    ds.summary()
    assert (
        capsys.readouterr().out
        == f"Dataset(path='{local_ds.path}', tensors=[])\n\n tensor    htype    shape    dtype  compression\n -------  -------  -------  -------  ------- \n"
    )


def test_fixed_tensor(local_ds):
    local_ds.create_tensor("image")
    local_ds.image.extend(np.ones((32, 28, 28)))
    assert len(local_ds) == 32
    assert_array_equal(local_ds.image.numpy(), np.ones((32, 28, 28)))


def test_dynamic_tensor(local_ds):
    local_ds.create_tensor("image")

    a1 = np.ones((32, 28, 28))
    a2 = np.ones((10, 36, 11))
    a3 = np.ones((29, 10))

    image = local_ds.image

    image.extend(a1)
    image.extend(a2)
    image.append(a3)

    expected_list = [*a1, *a2, a3]
    actual_list = image.numpy(aslist=True)

    assert type(actual_list) == list
    assert_array_lists_equal(expected_list, actual_list)

    # test negative indexing
    assert_array_equal(expected_list[-1], image[-1].numpy())
    assert_array_equal(expected_list[-2], image[-2].numpy())
    assert_array_lists_equal(expected_list[-2:], image[-2:].numpy(aslist=True))
    assert_array_lists_equal(expected_list[::-3], image[::-3].numpy(aslist=True))

    assert image.shape == (43, None, None)
    assert image.shape_interval.lower == (43, 28, 10)
    assert image.shape_interval.upper == (43, 36, 28)
    assert image.is_dynamic


def test_empty_samples(local_ds: Dataset):
    tensor = local_ds.create_tensor("with_empty")
    with pytest.raises(EmptyTensorError):
        ds_pytorch = local_ds.pytorch()

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
    for actual_sample, expected in zip(local_ds, expected_list):
        actual = actual_sample.with_empty.numpy()
        assert_array_equal(actual, expected)


def test_indexed_tensor(local_ds: Dataset):
    tensor = local_ds.create_tensor("abc")
    tensor.append([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert_array_equal(tensor[0, 1].numpy(), np.array([4, 5, 6]))
    assert_array_equal(tensor[0, 0:2].numpy(), np.array([[1, 2, 3], [4, 5, 6]]))
    assert_array_equal(tensor[0, 0::2].numpy(), np.array([[1, 2, 3], [7, 8, 9]]))


def test_safe_downcasting(local_ds):
    int_tensor = local_ds.create_tensor("int", dtype="uint8")
    int_tensor.append(0)
    int_tensor.append(1)
    int_tensor.extend([2, 3, 4])
    int_tensor.extend([5, 6, np.uint8(7)])
    int_tensor.append(np.zeros((0,), dtype="uint64"))
    with pytest.raises(SampleAppendError):
        int_tensor.append(-8)
    int_tensor.append(np.array([1]))
    assert len(int_tensor) == 10
    with pytest.raises(SampleAppendError):
        int_tensor.append(np.array([1.0]))

    float_tensor = local_ds.create_tensor("float", dtype="float32")
    float_tensor.append(0)
    float_tensor.append(1)
    float_tensor.extend([2, 3.0, 4.0])
    float_tensor.extend([5.0, 6.0, np.float32(7.0)])
    with pytest.raises(SampleAppendError):
        float_tensor.append(float(np.finfo(np.float32).max + 1))
    float_tensor.append(np.array([1]))
    float_tensor.append(np.array([1.0]))
    assert len(float_tensor) == 10


def test_scalar_samples(local_ds):
    tensor = local_ds.create_tensor("scalars")

    assert tensor.meta.dtype is None

    # first sample sets dtype
    tensor.append(5)
    assert tensor.meta.dtype == MAX_INT_DTYPE

    with pytest.raises(SampleAppendError):
        tensor.append(5.1)

    tensor.append(10)
    tensor.append(-99)
    tensor.append(np.array(4))

    tensor.append(np.int16(4))

    with pytest.raises(SampleAppendError):
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

    with pytest.raises(SampleAppendError):
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


def test_sequence_samples(local_ds):
    tensor = local_ds.create_tensor("arrays")

    tensor.append([1, 2, 3])
    tensor.extend([[4, 5, 6]])
    local_ds.clear_cache()

    assert len(tensor) == 2
    expected_list = [[1, 2, 3], [4, 5, 6]]
    expected = np.array(expected_list)
    assert_array_equal(tensor.numpy(), expected)

    assert type(tensor.numpy(aslist=True)) == list
    assert_array_lists_equal(tensor.numpy(aslist=True), expected_list)


def test_iterate_dataset(local_ds):
    labels = [1, 9, 7, 4]
    local_ds.create_tensor("image")
    local_ds.create_tensor("label")

    local_ds.image.extend(np.ones((4, 28, 28)))
    local_ds.label.extend(np.asarray(labels).reshape((4, 1)))

    for idx, sub_ds in enumerate(local_ds):
        img = sub_ds.image.numpy()
        label = sub_ds.label.numpy()
        assert_array_equal(img, np.ones((28, 28)))
        assert label.shape == (1,)
        assert label == labels[idx]


def _check_tensor(tensor, data):
    assert_array_equal(tensor.numpy(), data)


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
    data_2 = np.array([1, 2, 3, 9, 8, 7, 100, 99, 98, 99, 101, 12, 15, 18])
    ds.create_tensor("data")
    ds.create_tensor("data_2")

    ds.data.extend(data)
    ds.data_2.extend(data_2)

    assert len(ds) == 11
    assert ds.min_len == len(ds)
    assert ds.max_len == 14
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
    label = memory_ds.create_tensor(
        "label", htype="class_label", class_names=["a", "b", "c", "d", "e", "f"]
    )
    video = memory_ds.create_tensor("video", htype="video", sample_compression="mkv")
    bin_mask = memory_ds.create_tensor("bin_mask", htype="binary_mask")
    segment_mask = memory_ds.create_tensor(
        "segment_mask", htype="segment_mask", class_names=["a", "b", "c"]
    )
    keypoints_coco = memory_ds.create_tensor(
        "keypoints_coco",
        htype="keypoints_coco",
        keypoints=["arm", "leg", "torso"],
        connections=[[0, 2], [1, 2]],
    )
    point = memory_ds.create_tensor("point", htype="point")
    point_cloud = memory_ds.create_tensor(
        "point_cloud", htype="point_cloud", sample_compression="las"
    )
    intrinsics = memory_ds.create_tensor("intrinsics", htype="intrinsics")
    embedding = memory_ds.create_tensor("embedding", htype="embedding")

    image.append(np.ones((28, 28, 3), dtype=np.uint8))
    bbox.append(np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float32))
    # label.append(5)
    label.append(np.array(5, dtype=np.uint32))
    with pytest.raises(SampleAppendError):
        video.append(np.ones((10, 28, 28, 3), dtype=np.uint8))
    bin_mask.append(np.zeros((28, 28), dtype=bool))
    segment_mask.append(np.ones((28, 28), dtype=np.uint32))
    keypoints_coco.append(np.ones((51, 2), dtype=np.int32))
    point.append(np.ones((11, 2), dtype=np.int32))

    point_cloud.append(
        deeplake.read(
            os.path.join(get_dummy_data_path("point_cloud"), "point_cloud.las")
        )
    )
    point_cloud_dummy_data_path = pathlib.Path(get_dummy_data_path("point_cloud"))
    point_cloud.append(deeplake.read(point_cloud_dummy_data_path / "point_cloud.las"))
    # Along the first direcection three matrices are concatenated, the first matrix is P,
    # the second one is Tr and the third one is R
    intrinsics.append(np.zeros((3, 4, 4), dtype=np.float32))
    embedding.append(np.random.rand((100)).astype(np.float32))


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

    with pytest.raises(SampleAppendError):
        tensor.append(np.ones((10, 10), dtype="float64"))

    with pytest.raises(SampleAppendError):
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
    assert_array_equal(x, arr1[0])
    assert_array_equal(x, arr2[0])
    tensor.append(x)
    assert_array_equal(tensor.numpy(), np.concatenate([arr1, arr2]))


@pytest.mark.slow
def test_hub_dataset_suffix_bug(hub_cloud_ds, hub_cloud_dev_token):
    # creating dataset with similar name but some suffix removed from end
    ds = deeplake.dataset(hub_cloud_ds.path[:-1], token=hub_cloud_dev_token)

    # need to delete because it's a different path (won't be auto cleaned up)
    ds.delete()


def test_index_range(memory_ds):
    with pytest.raises(IndexError):
        memory_ds[0]

    memory_ds.create_tensor("label")

    with pytest.raises(IndexError):
        memory_ds.label[0]

    memory_ds.label.extend([5, 6, 7])
    assert len(memory_ds) == 3
    assert len(memory_ds.label) == 3

    for valid_idx in [0, 1, 2, -0, -1, -2, -3]:
        memory_ds[valid_idx]
        memory_ds.label[valid_idx]

    for invalid_idx in [3, 4, -4, -5]:
        with pytest.raises(IndexError):
            memory_ds[invalid_idx]
        with pytest.raises(IndexError):
            memory_ds.label[invalid_idx]

    memory_ds[[0, 1, 2]]
    with pytest.raises(IndexError):
        memory_ds[[0, 1, 2, 3, 4, 5]]


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_empty_dataset(convert_to_pathlib):
    with CliRunner().isolated_filesystem():
        test_path = "test"
        test_path = convert_string_to_pathlib_if_needed(test_path, convert_to_pathlib)
        ds = deeplake.dataset(test_path)
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.create_tensor("z")
        ds = deeplake.dataset(test_path)
        assert list(ds.tensors) == ["x", "y", "z"]


@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_like(local_path, convert_to_pathlib):
    src_path = os.path.join(local_path, "src")
    src_path = convert_string_to_pathlib_if_needed(src_path, convert_to_pathlib)
    dest_path = os.path.join(local_path, "dest")
    dest_path = convert_string_to_pathlib_if_needed(dest_path, convert_to_pathlib)

    src_ds = deeplake.dataset(src_path)
    src_ds.info.update(key=0)

    src_ds.create_tensor("a", htype="image", sample_compression="png")
    src_ds.create_tensor("b", htype="class_label")
    src_ds.create_tensor("c")
    src_ds.create_tensor("d", dtype=bool)

    src_ds.d.info.update(key=1)

    assert src_ds.info.key == 0
    assert src_ds.d.info.key == 1

    dest_ds = deeplake.like(dest_path, src_ds)

    assert tuple(dest_ds.tensors.keys()) == ("a", "b", "c", "d")

    assert dest_ds.a.meta.htype == "image"
    assert dest_ds.a.meta.sample_compression == "png"
    assert dest_ds.b.meta.htype == "class_label"
    assert dest_ds.c.meta.htype is None
    assert dest_ds.d.dtype == bool

    assert dest_ds.info.key == 0
    assert dest_ds.d.info.key == 1

    assert len(dest_ds) == 0


def test_inplace_like(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc", htype="class_label")
        ds.abc.extend([1, 0, 1, 0, 0, 1])
        ds.abc.info.update(class_names=["a", "b"])

    # reload to ensure we are not using cached data
    ds = pickle.loads(pickle.dumps(ds))

    with deeplake.like(local_ds.path, local_ds.path, overwrite=True) as ds:
        assert ds.abc.meta.htype == "class_label"
        assert ds.abc.info.class_names == ["a", "b"]

        ds.abc.extend([1, 0, 1, 0, 0, 1])

    ds = pickle.loads(pickle.dumps(ds))

    with deeplake.like(ds, ds, overwrite=True) as ds:
        assert ds.abc.meta.htype == "class_label"
        assert ds.abc.info.class_names == ["a", "b"]


def test_tensor_creation_fail_recovery():
    with CliRunner().isolated_filesystem():
        ds = deeplake.dataset("test")
        with ds:
            ds.create_tensor("x")
            ds.create_tensor("y")
            with pytest.raises(UnsupportedCompressionError):
                ds.create_tensor("z", sample_compression="something_random")
        ds = deeplake.dataset("test")
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
            deeplake.delete("test/")

        deeplake.delete("test/", force=True)
        assert not os.path.isfile("test/test.txt")

        deeplake.empty("test/").create_tensor("tmp")
        assert os.path.isfile("test/dataset_meta.json")

        deeplake.delete("test/")
        assert not os.path.isfile("test/dataset_meta.json")

        pathlib_path = pathlib.Path("test/")
        deeplake.empty(pathlib_path).create_tensor("tmp")
        assert os.path.isfile("test/dataset_meta.json")

        deeplake.delete(pathlib_path)
        assert not os.path.isfile("test/dataset_meta.json")

        old_size = deeplake.constants.DELETE_SAFETY_SIZE
        deeplake.constants.DELETE_SAFETY_SIZE = 1 * MB

        ds = deeplake.empty("test/")
        ds.create_tensor("data")
        ds.data.extend(np.zeros((100, 2000)))

        with pytest.raises(DatasetTooLargeToDelete):
            deeplake.delete("test/")
        assert os.path.isfile("test/dataset_meta.json")

        deeplake.delete("test/", large_ok=True)
        assert not os.path.isfile("test/dataset_meta.json")

        deeplake.constants.DELETE_SAFETY_SIZE = old_size


@pytest.mark.slow
def test_invalid_token():
    with pytest.raises(InvalidTokenException):
        ds = deeplake.load(
            "hub://activeloop-test/sohas-weapons-train", token="invalid token"
        )

    with pytest.raises(InvalidTokenException):
        ds = deeplake.empty(
            "hub://activeloop-test/sohas-weapons-train", token="invalid token"
        )

    with pytest.raises(InvalidTokenException):
        ds = deeplake.dataset(
            "hub://activeloop-test/sohas-weapons-train", token="invalid token"
        )


@pytest.mark.slow
def test_dataset_rename(hub_cloud_ds_generator, hub_cloud_path, hub_cloud_dev_token):
    ds = hub_cloud_ds_generator()
    ds.create_tensor("abc")
    ds.abc.append([1, 2, 3, 4])
    original_path = ds.path

    new_name = f"renamed-{time.time()}"
    new_path = original_path.rsplit("/", 1)[0] + "/" + new_name

    with pytest.raises(RenameError):
        ds.rename("hub://wrong_org/test")

    ds.rename(new_name)

    assert ds.path.endswith("/" + new_name)
    assert_array_equal(ds.abc.numpy(), np.array([[1, 2, 3, 4]]))

    ds = deeplake.load(new_path, token=hub_cloud_dev_token)
    assert_array_equal(ds.abc.numpy(), np.array([[1, 2, 3, 4]]))

    deeplake.delete(new_path, token=hub_cloud_dev_token)


@pytest.mark.parametrize(
    "path,hub_token",
    [
        ["local_path", "hub_cloud_dev_token"],
        pytest.param("hub_cloud_path", "hub_cloud_dev_token", marks=pytest.mark.slow),
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_workers", [2])
@pytest.mark.parametrize("progressbar", [True])
def test_dataset_deepcopy(path, hub_token, num_workers, progressbar):
    src_path = "_".join((path, "src1"))
    dest_path = "_".join((path, "dest1"))

    src_ds = deeplake.empty(src_path, overwrite=True, token=hub_token)

    with src_ds:
        src_ds.info.update(key=0)

        src_ds.create_tensor("a", htype="image", sample_compression="png")
        src_ds.create_tensor("b", htype="class_label")
        src_ds.create_tensor("c")
        src_ds.create_tensor("d", dtype=bool)

        src_ds.d.info.update(key=1)

        src_ds["a"].append(np.ones((28, 28), dtype="uint8"))
        src_ds["b"].append(0)

    dest_ds = deeplake.deepcopy(
        src_path,
        dest_path,
        token=hub_token,
        num_workers=num_workers,
        progressbar=progressbar,
    )

    assert list(dest_ds.tensors) == ["a", "b", "c", "d"]
    assert dest_ds.a.meta.htype == "image"
    assert dest_ds.a.meta.sample_compression == "png"
    assert dest_ds.b.meta.htype == "class_label"
    assert dest_ds.c.meta.htype == None
    assert dest_ds.d.dtype == bool

    assert dest_ds.info.key == 0
    assert dest_ds.d.info.key == 1

    for tensor in dest_ds.meta.tensors:
        assert_array_equal(src_ds[tensor].numpy(), dest_ds[tensor].numpy())

    deeplake.delete(src_path, token=hub_token)
    deeplake.delete(dest_path, token=hub_token)


@pytest.mark.parametrize(
    "path,hub_token",
    [
        ["local_path", "hub_cloud_dev_token"],
        pytest.param("hub_cloud_path", "hub_cloud_dev_token", marks=pytest.mark.slow),
    ],
    indirect=True,
)
def test_deepcopy(path, hub_token):
    src_path = "_".join((path, "src"))
    dest_path = "_".join((path, "dest"))

    src_ds = deeplake.empty(src_path, overwrite=True, token=hub_token)
    dest_ds = deeplake.empty(dest_path, overwrite=True, token=hub_token)

    with src_ds:
        src_ds.info.update(key=0)

        src_ds.create_tensor("a", htype="image", sample_compression="png")
        src_ds.create_tensor("b", htype="class_label")
        src_ds.create_tensor("c")
        src_ds.create_tensor("d", dtype=bool)
        src_ds.create_group("g")

        src_ds.d.info.update(key=1)

        src_ds["a"].append(np.ones((28, 28), dtype="uint8"))
        src_ds["b"].append(0)

    deeplake.deepcopy(
        src_ds,
        dest_path,
        overwrite=True,
        token=hub_token,
        num_workers=0,
    )

    deeplake.deepcopy(
        src_path,
        dest_path,
        overwrite=True,
        token=hub_token,
        num_workers=1,
    )

    with pytest.raises(TypeError):
        deeplake.deepcopy(
            src_ds.a,
            dest_path,
            overwrite=True,
            token=hub_token,
            num_workers=0,
        )
    with pytest.raises(TypeError):
        deeplake.deepcopy(
            src_ds.g,
            dest_path,
            overwrite=True,
            token=hub_token,
            num_workers=0,
        )
    with pytest.raises(TypeError):
        deeplake.deepcopy(
            src_ds[0],
            dest_path,
            overwrite=True,
            token=hub_token,
            num_workers=0,
        )


@pytest.mark.parametrize(
    "path,hub_token",
    [
        ["local_path", "hub_cloud_dev_token"],
        pytest.param("hub_cloud_path", "hub_cloud_dev_token", marks=pytest.mark.slow),
    ],
    indirect=True,
)
def test_deepcopy_errors(path, hub_token):
    src_path = "_".join((path, "src"))
    dest_path = "_".join((path, "dest"))

    src_ds = deeplake.empty(src_path, overwrite=True, token=hub_token)
    dest_ds = deeplake.empty(dest_path, overwrite=True, token=hub_token)

    with src_ds:
        src_ds.info.update(key=0)

        src_ds.create_tensor("a", htype="image", sample_compression="png")
        src_ds.create_tensor("b", htype="class_label")
        src_ds.create_tensor("c")
        src_ds.create_tensor("d", dtype=bool)

        src_ds.d.info.update(key=1)

        src_ds["a"].append(np.ones((28, 28), dtype="uint8"))
        src_ds["b"].append(0)

    with pytest.raises(DatasetHandlerError):
        deeplake.deepcopy(src_path, dest_path, token=hub_token)

    with pytest.raises(UnsupportedParameterException):
        deeplake.deepcopy(
            src_path,
            dest_path,
            overwrite=True,
            src_token=hub_token,
        )

    with pytest.raises(UnsupportedParameterException):
        deeplake.deepcopy(
            src_path,
            dest_path,
            overwrite=True,
            dest_token=hub_token,
        )


def test_invalid_tensor_name(memory_ds):
    with pytest.raises(InvalidTensorNameError):
        memory_ds.create_tensor("group/version_state")
    with pytest.raises(InvalidTensorNameError):
        memory_ds.create_tensor("info")


def test_compressions_list():
    assert deeplake.compressions == [
        # "apng",
        "avi",
        "bmp",
        "dcm",
        "dib",
        "eps",
        "flac",
        "fli",
        "gif",
        "ico",
        "im",
        "jpeg",
        "jpeg2000",
        "las",
        "lz4",
        "mkv",
        "mp3",
        "mp4",
        "mpo",
        "msp",
        "nii",
        "nii.gz",
        "pcx",
        "ply",
        "png",
        "ppm",
        "sgi",
        "tga",
        "tiff",
        "wav",
        "webp",
        "wmf",
        "xbm",
        None,
    ]


def test_htypes_list():
    assert deeplake.htypes == [
        "audio",
        "bbox",
        "bbox.3d",
        "binary_mask",
        "class_label",
        "dicom",
        "embedding",
        "generic",
        "image",
        "image.gray",
        "image.rgb",
        "instance_label",
        "intrinsics",
        "json",
        "keypoints_coco",
        "list",
        "mesh",
        "nifti",
        "point",
        "point_cloud",
        "polygon",
        "segment_mask",
        "tag",
        "text",
        "video",
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
    assert_array_equal(c[0].numpy(), np.zeros((3, 2)))
    assert "d" in ds.y.z.a.b.groups
    e = ds.y.z.a.b.d.e
    assert_array_equal(e[0].numpy(), np.ones((4, 3)))

    ds.create_group("g")
    ds.g.create_tensor("g")

    with ds:
        ds.create_group("h")
        ds.h.create_group("i")
        ds.h.i.create_tensor("j")
        assert not ds.storage.autoflush
    assert "j" in ds.h.i.tensors
    assert ds.storage.autoflush


def test_tensor_delete(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("x", max_chunk_size=2 * MB)
    ds.x.extend(np.ones((3, 253, 501, 5)))
    ds.delete_tensor("x")
    assert set(ds.storage.keys()) == {"dataset_meta.json", "version_control_info.json"}
    assert ds.tensors == {}

    ds.create_tensor("x/y")
    ds.delete_tensor("x/y")
    ds.create_tensor("x/y")
    ds["x"].delete_tensor("y")
    ds.delete_group("x")
    assert set(ds.storage.keys()) == {"dataset_meta.json", "version_control_info.json"}
    assert ds.tensors == {}

    ds.create_tensor("x/y/z")
    ds.delete_group("x")
    ds.create_tensor("x/y/z")
    ds["x"].delete_group("y")
    ds.create_tensor("x/y/z")
    ds["x/y"].delete_tensor("z")
    ds.delete_group("x")
    assert set(ds.storage.keys()) == {"dataset_meta.json", "version_control_info.json"}
    assert ds.tensors == {}
    assert ds.meta.hidden_tensors == []


def test_group_delete_bug(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc/first")
        ds.delete_group("abc")

    ds = local_ds_generator()
    assert ds.tensors == {}
    assert ds.groups == {}


def test_tensor_rename(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("x/y/z")
    ds["x/y/z"].append([1, 2, 3])
    ds.rename_tensor("x/y/z", "x/y/y")

    assert_array_equal(ds["x/y/y"][0].numpy(), np.array([1, 2, 3]))

    with pytest.raises(TensorDoesNotExistError):
        ds["x/y/z"].numpy()

    ds.create_tensor("x/y/z")
    ds["x/y/z"].append([4, 5, 6])
    assert_array_equal(ds["x/y/z"][0].numpy(), np.array([4, 5, 6]))

    with pytest.raises(RenameError):
        ds.rename_tensor("x/y/y", "x/a")

    with pytest.raises(RenameError):
        ds["x"].rename_tensor("y/y", "y")

    ds.create_tensor("x/y/a/b")
    with pytest.raises(TensorGroupAlreadyExistsError):
        ds["x/y"].rename_tensor("y", "a")

    ds.create_tensor("abc/xyz")
    with pytest.raises(InvalidTensorNameError):
        ds.rename_tensor("abc/xyz", "abc/append")

    ds.create_tensor("abc/efg")
    with pytest.raises(TensorAlreadyExistsError):
        ds.rename_tensor("abc/xyz", "abc/efg")

    ds["x"].rename_tensor("y/y", "y/b")

    assert_array_equal(ds["x/y/b"][0].numpy(), np.array([1, 2, 3]))

    ds = local_ds_generator()
    assert_array_equal(ds["x/y/b"][0].numpy(), np.array([1, 2, 3]))

    ds.delete_tensor("x/y/b")


def test_group_rename(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("g1/g2/g3/g4/t1")
        ds.create_group("g1/g2/g6")
        ds.create_tensor("g1/g2/t")
        ds["g1/g2/g3/g4/t1"].append([1, 2, 3])
        ds["g1/g2"].rename_group("g3/g4", "g3/g5")
        assert_array_equal(ds["g1/g2/g3/g5/t1"].numpy(), np.array([[1, 2, 3]]))
        with pytest.raises(TensorGroupDoesNotExistError):
            ds["g1"].rename_group("g2/g4", "g2/g5")
        with pytest.raises(TensorGroupAlreadyExistsError):
            ds["g1"].rename_group("g2/g3", "g2/g6")
        with pytest.raises(TensorAlreadyExistsError):
            ds["g1"].rename_group("g2/g3", "g2/t")
        with pytest.raises(InvalidTensorGroupNameError):
            ds["g1"].rename_group("g2/g3", "g2/append")
        with pytest.raises(RenameError):
            ds["g1"].rename_group("g2/g3", "g/g4")
        ds["g1"].rename_group("g2", "g6")
        assert_array_equal(ds["g1/g6/g3/g5/t1"].numpy(), np.array([[1, 2, 3]]))

    with local_ds_generator() as ds:
        assert_array_equal(ds["g1/g6/g3/g5/t1"].numpy(), np.array([[1, 2, 3]]))


def test_vc_bug(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("abc")
    ds.abc.append(1)
    a = ds.commit("first")
    ds.checkout(a)
    ds.create_tensor("a/b/c/d")
    assert list(ds.tensors) == ["abc", "a/b/c/d"]


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.slow
def test_tobytes(memory_ds, compressed_image_paths, audio_paths):
    ds = memory_ds
    ds.create_tensor("image", sample_compression="jpeg")
    ds.create_tensor("audio", sample_compression="mp3")
    with ds:
        for _ in range(3):
            ds.image.append(deeplake.read(compressed_image_paths["jpeg"][0]))
            ds.audio.append(deeplake.read(audio_paths["mp3"]))
    with open(compressed_image_paths["jpeg"][0], "rb") as f:
        image_bytes = f.read()
    with open(audio_paths["mp3"], "rb") as f:
        audio_bytes = f.read()
    for i in range(3):
        assert ds.image[i].tobytes() == image_bytes
        assert ds.audio[i].tobytes() == audio_bytes


@pytest.mark.slow
def test_tobytes_link(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("images", htype="link[image]", sample_compression="jpg")
        ds.images.append(deeplake.link("https://picsum.photos/id/237/200/300"))
        sample = deeplake.read("https://picsum.photos/id/237/200/300")
        assert ds.images[0].tobytes() == sample.buffer


def test_tobytes_sequence(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="sequence")
        ds.abc.extend([[1, 2, 3], [4, 5, 6, 7]])
        assert ds.abc[0].tobytes() == np.array([1, 2, 3]).tobytes()
        assert ds.abc[1].tobytes() == np.array([4, 5, 6, 7]).tobytes()

        with pytest.raises(ValueError):
            ds.abc[:2].tobytes()

        with pytest.raises(ValueError):
            ds.abc[0, 1].tobytes()


def test_tensor_clear(local_ds_generator):
    ds = local_ds_generator()
    a = ds.create_tensor("a")
    a.extend([1, 2, 3, 4])
    a.clear()
    assert len(ds) == 0
    assert len(a) == 0

    image = ds.create_tensor("image", htype="image", sample_compression="png")
    image.extend(np.ones((4, 224, 224, 3), dtype="uint8"))
    image.extend(np.ones((4, 224, 224, 3), dtype="uint8"))
    image.clear()
    assert len(ds) == 0
    assert len(image) == 0
    assert image.htype == "image"
    assert image.meta.sample_compression == "png"
    image.extend(np.ones((4, 224, 224, 3), dtype="uint8"))
    a.append([1, 2, 3])

    ds = local_ds_generator()
    assert len(ds) == 1
    assert len(image) == 4
    assert image.htype == "image"
    assert image.meta.sample_compression == "png"


def test_tensor_clear_seq(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc", htype="sequence")
        ds.abc.extend([[1, 2, 3, 4]])
        ds.abc.extend([[1, 2, 3, 4, 5]])
        ds.abc.clear()
        assert ds.abc.shape == (0, 0)


def test_no_view(memory_ds):
    memory_ds.create_tensor("a")
    memory_ds.a.extend([0, 1, 2, 3])
    memory_ds.create_tensor("b")
    memory_ds.b.extend([4, 5, 6])
    memory_ds.create_tensor("c/d")
    memory_ds["c/d"].append([7, 8, 9])

    with pytest.raises(InvalidOperationError):
        memory_ds[:2].create_tensor("c")

    with pytest.raises(InvalidOperationError):
        memory_ds[:3].create_tensor_like("c", memory_ds.a)

    with pytest.raises(InvalidOperationError):
        memory_ds[:2].delete_tensor("a")

    with pytest.raises(InvalidOperationError):
        memory_ds[:2].delete_tensor("c/d")

    with pytest.raises(InvalidOperationError):
        memory_ds[:2].delete_group("c")

    with pytest.raises(InvalidOperationError):
        memory_ds[:2].delete()

    with pytest.raises(InvalidOperationError):
        memory_ds.a[:2].append(0)

    with pytest.raises(InvalidOperationError):
        memory_ds.b[:3].extend([3, 4])

    with pytest.raises(InvalidOperationError):
        memory_ds[1:3].read_only = False

    with pytest.raises(InvalidOperationError):
        memory_ds[0].read_only = True

    memory_ds.read_only = True


@pytest.mark.parametrize(
    "x_args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
@pytest.mark.parametrize(
    "y_args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
@pytest.mark.parametrize(
    "x_size", [5, pytest.param((32 * 1000), marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("htype", ["generic", "sequence"])
def test_ds_append(memory_ds, x_args, y_args, x_size, htype):
    ds = memory_ds
    ds.create_tensor("x", **x_args, max_chunk_size=2**20, htype=htype)
    ds.create_tensor("y", dtype="uint8", htype=htype, **y_args)
    with pytest.raises(SampleAppendError):
        ds.append({"x": np.ones(2), "y": np.zeros(1)})
    ds.append({"x": np.ones(2), "y": [1, 2, 3]})
    ds.create_tensor("z", htype=htype)
    with pytest.raises(KeyError):
        ds.append({"x": np.ones(2), "y": [4, 5, 6, 7]})
    ds.append({"x": np.ones(3), "y": [8, 9, 10]}, skip_ok=True)
    ds.append({"x": np.ones(4), "y": [2, 3, 4]}, skip_ok=True)
    with pytest.raises(ValueError):
        ds.append({"x": np.ones(2), "y": [4, 5], "z": np.ones(4)})
    with pytest.raises(SampleAppendError):
        ds.append({"x": np.ones(x_size), "y": np.zeros(2)}, skip_ok=True)
    assert len(ds.x) == 3
    assert len(ds.y) == 3
    assert len(ds.z) == 0
    assert ds.x.chunk_engine.commit_diff.num_samples_added == 3
    assert ds.y.chunk_engine.commit_diff.num_samples_added == 3
    assert ds.z.chunk_engine.commit_diff.num_samples_added == 0
    assert len(ds) == 0
    for _ in range(3):
        ds.append({"z": np.zeros(2)}, skip_ok=True)
    assert len(ds.z) == 3
    ds.append({"x": np.ones(3), "y": [1, 2, 3]}, append_empty=True)
    assert len(ds.x) == 4
    assert len(ds.y) == 4
    assert len(ds.z) == 4
    assert ds.x.chunk_engine.commit_diff.num_samples_added == 4
    assert ds.y.chunk_engine.commit_diff.num_samples_added == 4
    assert ds.z.chunk_engine.commit_diff.num_samples_added == 4
    assert len(ds) == 4


def test_ds_append_with_ds_view():
    ds1 = deeplake.dataset("mem://x")
    ds2 = deeplake.dataset("mem://y")
    ds1.create_tensor("x")
    ds2.create_tensor("x")
    ds1.create_tensor("y")
    ds2.create_tensor("y")
    ds1.append({"x": [0, 1], "y": [1, 2]})
    ds2.append(ds1[0])
    assert_array_equal(ds1.x, np.array([[0, 1]]))
    assert_array_equal(ds1.x, ds2.x)
    assert_array_equal(ds1.y, np.array([[1, 2]]))
    assert_array_equal(ds1.y, ds2.y)


def test_ds_extend():
    ds1 = deeplake.dataset("mem://x")
    ds2 = deeplake.dataset("mem://y")
    ds1.create_tensor("x")
    ds2.create_tensor("x")
    ds1.create_tensor("y")
    ds2.create_tensor("y")
    ds1.extend({"x": [0, 1, 2, 3], "y": [4, 5, 6, 7]})
    ds2.extend(ds1)
    assert_array_equal(ds1.x, np.arange(4).reshape(-1, 1))
    assert_array_equal(ds1.x, ds2.x)
    assert_array_equal(ds1.y, np.arange(4, 8).reshape(-1, 1))
    assert_array_equal(ds1.y, ds2.y)


@pytest.mark.slow
@pytest.mark.parametrize(
    "src_args", [{}, {"sample_compression": "png"}, {"chunk_compression": "png"}]
)
@pytest.mark.parametrize(
    "dest_args", [{}, {"sample_compression": "png"}, {"chunk_compression": "png"}]
)
@pytest.mark.parametrize("size", [(30, 40, 3), (1261, 759, 3)])
def test_append_with_tensor(src_args, dest_args, size):
    ds1 = deeplake.dataset("mem://ds1")
    ds2 = deeplake.dataset("mem://ds2")
    ds1.create_tensor("x", **src_args, max_chunk_size=2 * MB, tiling_threshold=2 * MB)
    x = np.random.randint(0, 256, size, dtype=np.uint8)
    ds1.x.append(x)
    ds2.create_tensor("y", max_chunk_size=3 * MB, tiling_threshold=2 * MB, **dest_args)
    ds2.y.append(ds1.x[0])
    assert_array_equal(ds1.x.numpy(), ds2.y.numpy())

    with pytest.raises(SampleAppendingError):
        ds1.append(np.zeros((416, 416, 3)))

    with pytest.raises(SampleAppendingError):
        ds1.append(set())

    with pytest.raises(SampleAppendingError):
        ds1.append([1, 2, 3])


def test_extend_with_tensor():
    ds1 = deeplake.dataset("mem://ds1")
    ds2 = deeplake.dataset("mem://ds2")
    with ds1:
        ds1.create_tensor("x")
        ds1.x.extend([1, 2, 3, 4])
    with ds2:
        ds2.create_tensor("x")
        ds2.x.extend(ds1.x)
    assert_array_equal(ds1.x, ds2.x)


def test_empty_extend(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x")
        ds.x.append(1)
        ds.create_tensor("y")
        ds.y.extend(np.zeros((len(ds), 3)))
    assert len(ds) == 0


def test_extend_with_progressbar():
    ds1 = deeplake.dataset("mem://ds1")
    with ds1:
        ds1.create_tensor("x")
        ds1.x.extend([1, 2, 3, 4], progressbar=True)
    assert_array_equal(ds1.x, np.array([[1], [2], [3], [4]]))


def test_auto_htype(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("a")
        ds.create_tensor("b")
        ds.create_tensor("c")
        ds.create_tensor("d")
        ds.create_tensor("e")
        ds.create_tensor("f")
        ds.a.append("hello")
        ds.b.append({"a": [1, 2]})
        ds.c.append([1, 2, 3])
        ds.d.append(np.array([{"x": ["a", 1, 2.5]}]))
        ds.e.append(["a", 1, {"x": "y"}, "b"])
        ds.f.append(ds.e[0])
    assert ds.a.htype == "text"
    assert ds.b.htype == "json"
    assert ds.c.htype == "generic"
    assert ds.d.htype == "json"
    assert ds.e.htype == "json"
    assert ds.f.htype == "json"


def test_datatype_conversions(memory_ds):
    ds = memory_ds
    ds.create_tensor("uuid", dtype=np.int32)

    with pytest.raises(SampleAppendError) as e:
        ds.uuid.append("asdasd")


@pytest.mark.slow
@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_sample_shape(memory_ds, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("w", **args)
        ds.create_tensor("x", **args)
        ds.create_tensor("y", **args)
        ds.create_tensor("z", **args)
        ds.w.extend(np.zeros((5, 4, 3, 2)))
        ds.x.extend(np.ones((5, 4000, 5000)))
        ds.y.extend([np.zeros((2, 3)), np.ones((3, 2))])
        ds.z.extend([np.ones((5, 4000, 3000)), np.ones((5, 3000, 4000))])
    assert ds.w[0].shape == (4, 3, 2)
    assert ds.x[0].shape == (4000, 5000)
    assert ds.y[0].shape == (2, 3)
    assert ds.y[1].shape == (3, 2)
    assert ds.z[0].shape == (5, 4000, 3000)
    assert ds.z[1].shape == (5, 3000, 4000)
    assert ds.w[0][0, :2].shape == (2, 2)
    assert ds.z[1][:2, 10:].shape == (2, 2990, 4000)


@enabled_remote_storages
def test_hub_remote_read_images(storage, memory_ds, color_image_paths, gdrive_creds):
    image_path = color_image_paths["jpeg"]
    with open(image_path, "rb") as f:
        byts = f.read()

    memory_ds.create_tensor("images", htype="image", sample_compression="jpg")

    image = deeplake.read("https://picsum.photos/200/300")
    memory_ds.images.append(image)
    assert memory_ds.images[0].shape == (300, 200, 3)

    storage["sample/samplejpg.jpg"] = byts
    image = deeplake.read(
        f"{storage.root}/sample/samplejpg.jpg",
        creds=gdrive_creds if storage.root.startswith("gdrive://") else None,
    )
    memory_ds.images.append(image)
    assert memory_ds.images[1].shape == (323, 480, 3)

    storage["samplejpg.jpg"] = byts
    image = deeplake.read(
        f"{storage.root}/samplejpg.jpg",
        creds=gdrive_creds if storage.root.startswith("gdrive://") else None,
    )
    memory_ds.images.append(image)
    assert memory_ds.images[2].shape == (323, 480, 3)


def test_hub_remote_read_gdrive_root(request, memory_ds, gdrive_creds):
    if not is_opt_true(request, GDRIVE_OPT):
        pytest.skip(f"{GDRIVE_OPT} flag not set")
    memory_ds.create_tensor("images", htype="image", sample_compression="jpg")
    memory_ds.images.append(deeplake.read("gdrive://cat.jpeg", creds=gdrive_creds))
    assert memory_ds.images[0].shape == (900, 900, 3)


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@enabled_remote_storages
def test_hub_remote_read_videos(storage, memory_ds):
    memory_ds.create_tensor("videos", htype="video", sample_compression="mp4")

    video = deeplake.read(
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    )
    memory_ds.videos.append(video)
    assert memory_ds.videos[0].shape == (361, 720, 1280, 3)

    if isinstance(storage, GCSProvider):
        video = deeplake.read(
            "gcs://gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
        )
        memory_ds.videos.append(video)
        assert memory_ds.videos[1].shape == (361, 720, 1280, 3)


@pytest.mark.slow
@pytest.mark.parametrize("aslist", (True, False))
@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "png"}, {"chunk_compression": "png"}]
)
@pytest.mark.parametrize(
    "idx", [3, slice(None), slice(5, 9), slice(3, 7, 2), [3, 7, 6, 4]]
)
def test_sequence_htype(memory_ds, aslist, args, idx):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="sequence", **args)
        for _ in range(10):
            ds.x.append([np.ones((2, 7, 3), dtype=np.uint8) for _ in range(5)])
    assert_array_equal(
        np.array(ds.x[idx].numpy(aslist=aslist)), np.ones((10, 5, 2, 7, 3))[idx]
    )
    assert ds.x.shape == (10, 5, 2, 7, 3)
    ds.checkout("branch", create=True)
    with ds:
        for _ in range(5):
            ds.x.append([np.ones((2, 7, 3), dtype=np.uint8) for _ in range(5)])
    assert_array_equal(
        np.array(ds.x[idx].numpy(aslist=aslist)), np.ones((15, 5, 2, 7, 3))[idx]
    )
    assert ds.x.shape == (15, 5, 2, 7, 3)
    ds.checkout("main")
    assert_array_equal(
        np.array(ds.x[idx].numpy(aslist=aslist)), np.ones((10, 5, 2, 7, 3))[idx]
    )
    assert ds.x.shape == (10, 5, 2, 7, 3)


@pytest.mark.parametrize("shape", [(13, 17, 3), (1007, 3001, 3)])
@pytest.mark.slow
def test_sequence_htype_with_hub_read(local_ds, shape, compressed_image_paths):
    ds = local_ds
    imgs = list(map(deeplake.read, compressed_image_paths["jpeg"][:3]))
    arrs = np.random.randint(0, 256, (5, *shape), dtype=np.uint8)
    with ds:
        ds.create_tensor("x", htype="sequence[image]", sample_compression="png")
        for i in range(5):
            if i % 2:
                ds.x.append(imgs)
            else:
                ds.x.append(arrs)
    for i in range(5):
        if i % 2:
            for j in range(3):
                assert_array_equal(ds.x[i][j].numpy(), imgs[j].array)
        else:
            for j in range(5):
                assert_array_equal(ds.x[i][j].numpy(), arrs[j])


def test_sequence_shapes(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="sequence")
        ds.abc.extend([[1, 2, 3], [4, 5, 6, 7]])

        assert ds.abc[0].shape == (3, 1)

        assert ds.abc.shape_interval.lower == (2, 3, 1)
        assert ds.abc.shape_interval.upper == (2, 4, 1)

        ds.create_tensor("xyz", htype="sequence")
        ds.xyz.append([[[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]]])

        assert ds.xyz.shape == (1, 2, 2, None)
        assert ds.xyz[0][0].shape == (2, 2)
        assert ds.xyz[0][1].shape == (2, 3)

        assert ds.xyz.shape_interval.lower == (1, 2, 2, 2)
        assert ds.xyz.shape_interval.upper == (1, 2, 2, 3)

        ds.create_tensor("red", htype="sequence")
        ds.red.extend([[4, 5, 6, 7], [1, 2, 3]])

        assert ds.red[0].shape == (4, 1)

        assert ds.red.shape_interval.lower == (2, 3, 1)
        assert ds.abc.shape_interval.upper == (2, 4, 1)


def test_shape_bug(memory_ds):
    ds = memory_ds
    ds.create_tensor("x")
    ds.x.extend(np.ones((5, 9, 2)))
    assert ds.x[1:4, 3:7].shape == (3, 4, 2)

    ds.x.extend(np.ones((5, 9, 3)))

    assert ds.x[1:2].shape == (1, 9, 2)
    assert ds.x[3:8].shape == (5, 9, None)
    assert ds.x[1:4, 2:4, :1].shape == (3, 2, 1)
    assert ds.x[3:7, 2:4, :1].shape == (4, 2, 1)
    assert ds.x[3:7, 2:4, :3].shape == (4, 2, None)


def test_hidden_tensors(local_ds_generator):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("x", hidden=True)
        ds.x.append(1)
        assert ds.tensors == {}
        ds.create_tensor("y")
        assert list(ds.tensors.keys()) == ["y"]
        ds.y.extend([1, 2])
        assert len(ds) == 2  # length of hidden tensor is not considered
        ds._hide_tensor("y")
    ds = local_ds_generator()
    assert ds.tensors == {}
    assert len(ds) == 0
    with ds:
        ds.create_tensor("w")
        ds.create_tensor("z")
        ds.append({"w": 2, "z": 3})  # hidden tensors not required

    # Test access
    assert_array_equal(ds.x, np.array([[1]]))
    assert_array_equal(ds.y, np.array([[1], [2]]))

    assert not ds.w.meta.hidden
    assert not ds.z.meta.hidden
    assert ds.x.meta.hidden
    assert ds.y.meta.hidden


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("progressbar", [True, False])
@pytest.mark.parametrize(
    "index", [slice(None), slice(5, None, None), slice(None, 8, 2), 7]
)
@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_dataset_copy(
    memory_ds, local_ds, num_workers, progressbar, index, convert_to_pathlib
):
    memory_ds.path = convert_string_to_pathlib_if_needed(
        memory_ds.path, convert_to_pathlib
    )
    local_ds.path = convert_string_to_pathlib_if_needed(
        local_ds.path, convert_to_pathlib
    )

    ds = memory_ds
    with ds:
        ds.create_tensor("images/image1")
        ds.create_tensor("images/image2")
        ds.create_tensor("label")
        ds.create_tensor("nocopy")
        for _ in range(10):
            ds.images.image1.append(np.random.randint(0, 256, (10, 10, 3)))
            ds.images.image2.append(np.random.randint(0, 256, (10, 10, 3)))
            ds.label.append(np.random.randint(0, 10, (1, 10)))
            ds.nocopy.append([0])

    deeplake.copy(
        ds[index],
        local_ds.path,
        tensors=["images", "label"],
        overwrite=True,
        num_workers=num_workers,
        progressbar=progressbar,
    )
    local_ds = deeplake.load(local_ds.path)
    assert set(local_ds.tensors) == set(["images/image1", "images/image2", "label"])
    for t in local_ds.tensors:
        assert_array_equal(ds[t][index].numpy(), local_ds[t].numpy())


@pytest.mark.parametrize(
    ("ds_generator", "path", "hub_token"),
    [
        ("local_ds_generator", "local_path", "hub_cloud_dev_token"),
        pytest.param(
            "s3_ds_generator",
            "s3_path",
            "hub_cloud_dev_token",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "gcs_ds_generator",
            "gcs_path",
            "hub_cloud_dev_token",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "hub_cloud_ds_generator",
            "hub_cloud_path",
            "hub_cloud_dev_token",
            marks=pytest.mark.slow,
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("convert_to_pathlib", [True, False])
def test_hub_exists(ds_generator, path, hub_token, convert_to_pathlib):
    path = convert_string_to_pathlib_if_needed(path, convert_to_pathlib)
    ds = ds_generator()
    assert deeplake.exists(path, token=hub_token) == True
    assert deeplake.exists(f"{path}_does_not_exist", token=hub_token) == False


@pytest.mark.slow
def test_partial_read_then_write(s3_ds_generator):
    ds = s3_ds_generator()
    with ds:
        ds.create_tensor("xyz")
        for i in range(10):
            ds.xyz.append(i * np.ones((1000, 1000)))

    ds = s3_ds_generator()
    assert_array_equal(ds.xyz[0].numpy(), 0 * np.ones((1000, 1000)))

    with ds:
        ds.xyz[1] = 20 * np.ones((1000, 1000))


def test_exist_ok(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("abc")
        ds.create_tensor("abc", exist_ok=True)
        ds.create_group("grp")
        with pytest.raises(TensorGroupAlreadyExistsError):
            ds.create_group("grp")
        ds.create_group("grp", exist_ok=True)


def test_exist_ok_htype(local_ds):
    with local_ds as ds:
        ds.create_tensor("test1", htype="text")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("test1")
        ds.create_tensor("test1", htype="text", exist_ok=True)

        ds.create_tensor("test2", htype="bbox")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("test2")
        ds.create_tensor("test2", htype="bbox", exist_ok=True)

        ds.create_tensor("test3", htype="polygon")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("test3")
        ds.create_tensor("test3", htype="polygon", exist_ok=True)

        ds.create_tensor("test4", htype="image", sample_compression="jpg")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("test4")
        ds.create_tensor(
            "test4", htype="image", sample_compression="jpg", exist_ok=True
        )

        ds.create_tensor("test5", htype="class_label")
        with pytest.raises(TensorAlreadyExistsError):
            ds.create_tensor("test5")
        ds.create_tensor("test5", htype="class_label", exist_ok=True)


def verify_label_data(ds):
    text_labels = [
        ["airplane"],
        ["boat"],
        ["airplane"],
        ["car"],
        ["airplane"],
        ["airplane"],
        ["car"],
    ]
    nested_text_labels = [
        [["airplane", "boat", "car"], ["boat", "car", "person"]],
        [["person", "car"], ["airplane", "bus"]],
        [["airplane", "boat"], ["car", "person"]],
    ]
    arr = np.array([0, 1, 0, 2, 0, 0, 2]).reshape((7, 1))
    nested_arr = [
        np.array([[0, 1, 2], [1, 2, 3]]),
        np.array([[3, 2], [0, 4]]),
        np.array([[0, 1], [2, 3]]),
    ]

    random_arr = np.array([[0, 1], [1, 2], [0, 1]])
    random_text_labels = [["l1", "l2"], ["l2", "l3"], ["l1", "l2"]]

    seq_arr = [
        np.array([0, 1, 2]).reshape(3, 1),
        [],
        [],
        np.array([1, 3]).reshape(2, 1),
        np.array([0, 2]).reshape(2, 1),
    ]
    seq_text_labels = [
        [["l1"], ["l2"], ["l3"]],
        [],
        [],
        [["l2"], ["l4"]],
        [["l1"], ["l3"]],
    ]

    # abc
    assert ds.abc.info.class_names == ["airplane", "boat", "car"]
    np_data = ds.abc.numpy()
    data = ds.abc.data()
    assert set(data.keys()) == {"value", "text"}
    assert_array_equal(np_data, arr)
    assert_array_equal(data["value"], np_data)
    assert data["text"] == text_labels

    # xyz
    assert ds.xyz.info.class_names == []
    np_data = ds.xyz.numpy()
    data = ds.xyz.data()
    assert set(data.keys()) == {"value"}
    assert_array_equal(np_data, arr)
    assert_array_equal(data["value"], np_data)

    # nested
    assert ds.nested.info.class_names == ["airplane", "boat", "car", "person", "bus"]
    np_data = ds.nested.numpy(aslist=True)
    data = ds.nested.data(aslist=True)
    assert set(data.keys()) == {"value", "text"}
    for i in range(2):
        assert_array_equal(np_data[i], nested_arr[i])
        assert_array_equal(data["value"][i], np_data[i])
    assert data["text"] == nested_text_labels

    # random
    assert ds.random.info.class_names == ["l1", "l2", "l3", "l4"]
    np_data = ds.random.numpy()
    data = ds.random.data()
    assert set(data.keys()) == {"value", "text"}
    assert_array_equal(np_data, random_arr)
    assert_array_equal(data["value"], np_data)
    assert data["text"] == random_text_labels

    # seq
    # assert ds.random.info.class_names == ["l1", "l2", "l3", "l4"]
    # np_data = ds.seq.numpy(aslist=True)
    # data = ds.seq.data(aslist=True)
    # assert set(data.keys()) == {"numeric", "text"}
    # for i in range(4):
    #     assert_array_equal(np_data[i], seq_arr[i])
    #     assert_array_equal(data["numeric"][i], np_data[i])
    # assert data["text"] == seq_text_labels


def test_text_label(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc", htype="class_label")
        ds.abc.append("airplane")
        ds.abc.append("boat")
        ds.abc.append("airplane")
        ds.abc.extend(["car", "airplane", 0, 2])

        ds.create_tensor("xyz", htype="class_label")
        ds.xyz.append(0)
        ds.xyz.append(1)
        ds.xyz.append(0)
        ds.xyz.extend([2, 0, 0, 2])

        ds.create_tensor("nested", htype="class_label")
        ds.nested.append([[0, 1, 2], [1, 2, 3]])
        ds.nested.info.class_names = ["airplane", "boat", "car"]
        ds.nested.extend([[["person", 2], ["airplane", "bus"]], [[0, 1], ["car", 3]]])

        temp = deeplake.constants._ENABLE_RANDOM_ASSIGNMENT
        deeplake.constants._ENABLE_RANDOM_ASSIGNMENT = True

        ds.create_tensor("random", htype="class_label")
        ds.random[0] = ["l1", "l2"]
        ds.random[1] = ["l2", "l3"]
        ds.random[2] = ["l2", "l4"]
        ds.random[2] = ["l1", "l2"]

        ds.create_tensor("seq", htype="sequence[class_label]")
        ds.seq.append(["l1", "l2", "l3"])
        with pytest.raises(NotImplementedError):
            ds.seq[3] = ["l2", "l4"]
            ds.seq.append(["l3", "l1"])
            ds.seq[4] = ["l1", "l3"]

        deeplake.constants._ENABLE_RANDOM_ASSIGNMENT = temp

        verify_label_data(ds)

    ds = local_ds_generator()
    verify_label_data(ds)


@pytest.mark.parametrize("scheduler", ["threaded", "processed"])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_text_labels_transform(local_ds_generator, scheduler, num_workers):
    with local_ds_generator() as ds:
        ds.create_tensor("labels", htype="class_label")
        ds.create_tensor("multiple_labels", htype="class_label")
        ds.create_tensor("seq_labels", htype="sequence[class_label]")

    labels = ["car", "ship", "train"]
    multiple_labels = [["car", "train"], ["ship", "car"], ["train", "ship"]]
    seq_labels = [["ship", "train", "car"], ["ship", "train"], ["car", "train"]]
    data = list(zip(labels, multiple_labels, seq_labels))

    @deeplake.compute
    def upload(data, ds):
        ds.labels.append(data[0])
        ds.multiple_labels.append(data[1])
        ds.seq_labels.append(data[2])
        return ds

    def convert_to_idx(data, label_idx_map):
        if isinstance(data, str):
            return label_idx_map[data]
        return [convert_to_idx(label, label_idx_map) for label in data]

    upload().eval(data, ds, scheduler=scheduler, num_workers=num_workers)

    assert all(not tensor.startswith("__temp") for tensor in ds._tensors())

    for tensor in ("labels", "multiple_labels", "seq_labels"):
        class_names = ds[tensor].info.class_names
        label_idx_map = {class_names[i]: i for i in range(len(class_names))}
        if tensor == "labels":
            arr = ds[tensor].numpy()
            assert class_names == ["car", "ship", "train"]
            expected = np.array(convert_to_idx(labels, label_idx_map)).reshape((3, 1))
        elif tensor == "multiple_labels":
            arr = ds[tensor].numpy()
            assert class_names == ["car", "train", "ship"]
            expected = np.array(convert_to_idx(multiple_labels, label_idx_map))
        else:
            arr = ds[tensor].numpy(aslist=True)
            assert class_names == ["ship", "train", "car"]
            expected = convert_to_idx(seq_labels, label_idx_map)
            expected = [np.array(seq).reshape(-1, 1).tolist() for seq in expected]
        assert len(arr) == len(expected)
        for a, e in zip(arr, expected):
            assert_array_equal(a, e)


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 2])
def test_transform_upload_fail(local_ds_generator, num_workers):
    @deeplake.compute
    def upload(data, ds):
        ds.append({"images": deeplake.link("lalala"), "labels": data})

    with local_ds_generator() as ds:
        ds.create_tensor("images", htype="link[image]", sample_compression="jpg")
        ds.create_tensor("labels", htype="class_label")

    with pytest.raises(TransformError):
        upload().eval([0, 1, 2, 3], ds)

    @deeplake.compute
    def upload(data, ds):
        ds.append(
            {"images": deeplake.link("https://picsum.photos/20/20"), "labels": data}
        )

    with local_ds_generator() as ds:
        assert list(ds.tensors) == ["images", "labels"]
        upload().eval([0, 1, 2, 3], ds)
        assert_array_equal(ds.labels.numpy().flatten(), np.array([0, 1, 2, 3]))
        assert list(ds.tensors) == ["images", "labels"]


def test_ignore_temp_tensors(local_path):
    with deeplake.dataset(local_path, overwrite=True) as ds:
        ds.create_tensor(
            "__temptensor",
            htype="class_label",
            hidden=True,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            create_id_tensor=False,
        )
        ds.__temptensor.append(123)

    with deeplake.load(local_path) as ds:
        assert list(ds.tensors) == []
        assert ds.meta.hidden_tensors == []
        assert set(ds.storage.keys()) == {
            "dataset_meta.json",
            "version_control_info.json",
        }

    with deeplake.dataset(local_path, overwrite=True) as ds:
        ds.create_tensor(
            "__temptensor",
            htype="class_label",
            hidden=True,
            create_sample_info_tensor=False,
            create_shape_tensor=False,
            create_id_tensor=False,
        )
        ds.__temptensor.append(123)

    with deeplake.load(local_path, read_only=True) as ds:
        assert list(ds.tensors) == []
        assert list(ds._tensors()) == ["__temptensor"]
        assert ds.meta.hidden_tensors == ["__temptensor"]
        assert ds.__temptensor[0].numpy() == 123


@pytest.mark.slow
def test_empty_sample_partial_read(s3_ds):
    with s3_ds as ds:
        ds.create_tensor("xyz")
        ds.xyz.append([1, 2, 3, 4])
        ds.xyz.append(None)
    assert ds.xyz[1].numpy().tolist() == []


def test_htype_config_bug(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc", htype="class_label")
        ds.abc.info.class_names.append("car")
        ds.create_tensor("xyz", htype="class_label")
        assert ds.xyz.info.class_names == []


def test_update_bug(local_ds):
    with local_ds as ds:
        bb = ds.create_tensor("bb", "bbox", dtype="float64")
        arr1 = np.array([[1.0, 2.0, 3.0, 4.0]])
        bb.append(arr1)
        assert_array_equal(bb[0].numpy(), arr1)

        arr2 = np.array([[5.0, 6.0, 7.0, 8.0]])
        bb[0] = arr2
        assert_array_equal(bb[0].numpy(), arr2)


def test_uneven_view(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(10)))
        ds.y.extend(list(range(5)))
        view = ds[list(range(0, 10, 2))]
        np.testing.assert_equal(np.arange(0, 10, 2), view.x.numpy().squeeze())
        with pytest.raises(IndexError):
            np.testing.assert_equal(np.arange(0, 10, 2), view.y.numpy().squeeze())


def test_uneven_iteration(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(10)))
        ds.y.extend(list(range(5)))
        ds._enable_padding()
        assert len(ds) == 10
        for i in range(10):
            x, y = ds[i].x.numpy(), ds[i].y.numpy()
            np.testing.assert_equal(x, i)
            target_y = i if i < 5 else []
            np.testing.assert_equal(y, target_y)

        for i, dsv in enumerate(ds):
            x, y = dsv.x.numpy(), dsv.y.numpy()
            np.testing.assert_equal(x, i)
            target_y = i if i < 5 else []
            np.testing.assert_equal(y, target_y)


def invalid_token_exception_check():
    with pytest.raises(InvalidTokenException):
        ds = deeplake.empty("hub://adilkhan/demo", token="invalid_token")


def user_not_logged_in_exception_check(runner):
    runner.invoke(logout)
    with pytest.raises(UserNotLoggedInException):
        ds = deeplake.load("hub://adilkhan/demo", read_only=False)

    with pytest.raises(UserNotLoggedInException):
        ds = deeplake.dataset("hub://adilkhan/demo", read_only=False)

    with pytest.raises(UserNotLoggedInException):
        ds = deeplake.empty("hub://adilkhan/demo")


def test_incompat_dtype_msg(local_ds, capsys):
    local_ds.create_tensor("abc", dtype="uint32")
    with pytest.raises(SampleAppendError):
        local_ds.abc.append([0.0])
    captured = capsys.readouterr()
    assert "True" not in captured


def test_ellipsis(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        arr = np.random.random((5, 3, 2, 3, 4))
        ds.x.extend(arr)
    assert_array_equal(arr[:3, ..., 1], ds.x[:3, ..., 1])
    assert_array_equal(arr[..., :2], ds.x[..., :2])
    assert_array_equal(arr[2:, ...], ds.x[2:, ...])
    assert_array_equal(arr[2:, ...][...], ds.x[2:, ...][...])
    assert_array_equal(arr[...], ds.x[...])


def test_copy_label_sync_disabled(local_ds, capsys):
    abc = local_ds.create_tensor("abc", htype="class_label")
    abc.extend([1, 2, 3, 4, 5])
    ds = local_ds.copy(
        f"{local_ds.path}_copy", overwrite=True, progressbar=False, num_workers=2
    )
    captured = capsys.readouterr().out
    assert captured.strip() == ""


def test_class_label_bug(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="class_label", class_names=["a", "b"])
        ds.abc.append([0, 1])
        ds.abc.append([1, 0])
        ds.commit()
        ds.abc.append("c")
        b = ds.commit()
        ds.checkout(b)
        assert ds.abc.info.class_names == ["a", "b", "c"]


def test_columnar_views(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.create_tensor("z")
        ds.x.extend(list(range(2)))
        ds.y.extend(list(range(3)))
        ds.z.extend(list(range(2)))
    view = ds[["x", "z"]]
    assert list(view.tensors) == ["x", "z"]
    ds.create_tensor("a/b")
    ds.create_tensor("c/d")
    view = ds[[("a", "b"), ("c", "d")]]
    assert list(view.tensors) == ["a/b", "c/d"]
    ds.create_tensor("a/c")
    ds.create_tensor("a/d")
    view = ds["a"][["b", "d"]]
    assert list(view.tensors) == ["b", "d"]
    assert view.group_index == "a"


@pytest.mark.slow
@pytest.mark.parametrize("verify", [True, False])
def test_bad_link(local_ds_generator, verify):
    with local_ds_generator() as ds:
        ds.create_tensor(
            "images", htype="link[image]", sample_compression="jpg", verify=verify
        )
        ds.images.append(deeplake.link("https://picsum.photos/200/200"))

    if verify:
        with pytest.raises(SampleAppendError):
            ds.images.append(deeplake.link("https://picsum.photos/lalala"))

        with local_ds_generator() as ds:
            assert len(ds) == 1
    else:
        ds.images.append(deeplake.link("https://picsum.photos/lalala"))
        with local_ds_generator() as ds:
            assert len(ds) == 2


def test_rich(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
    rich_print(ds)
    rich_print(ds.info)
    rich_print(ds.x.info)


def test_groups_info(local_ds):
    with local_ds as ds:
        ds.create_tensor("group/tensor")
        ds.group.tensor.extend([0, 1, 2])

        with pytest.raises(GroupInfoNotSupportedError):
            ds.group.info["a"] = 1


def test_iter_warning(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(list(range(100)))

        for i in range(10):
            ds[i]

        with pytest.warns(UserWarning):
            ds[10]

        for i in range(10):
            ds.abc[i]

        with pytest.warns(UserWarning):
            ds.abc[10]


@requires_libdeeplake
def test_random_split(local_ds):
    with local_ds as ds:
        ds.create_tensor("label")
        ds.label.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        train, test = ds.random_split([6, 4])
        assert len(train) == 6
        assert len(test) == 4

        train, test = ds.random_split([0.8, 0.2])
        assert len(train) == 8
        assert len(test) == 2

        train, test, val = ds.random_split([0.5, 0.3, 0.2])

        assert len(train) == 5
        assert len(test) == 3
        assert len(val) == 2

        with pytest.raises(ValueError):
            ds.random_split([0.5, 0.5, 0.5])

        with pytest.raises(ValueError):
            ds.random_split([0.5, 1.3])

        ds.create_tensor("label2")
        ds.label2.extend([0, 1])

        with pytest.raises(ValueError):
            ds.random_split([0.5, 0.5])


@requires_libdeeplake
def test_random_split_views(local_ds):
    with local_ds as ds:
        ds.create_tensor("label")
        ds.label.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        views = [ds[:5], ds[[1, 3, 5, 7, 9]]]
        for view in views:
            train, test = view.random_split([3, 2])
            assert len(train) == 3
            assert len(test) == 2

            train, test = view.random_split([0.6, 0.4])
            assert len(train) == 3
            assert len(test) == 2


@pytest.mark.flaky
@pytest.mark.slow
def test_invalid_ds_name():
    with pytest.raises(InvalidDatasetNameException):
        deeplake.dataset("folder/datasets/dataset name *")

    verify_dataset_name("folders/datasets/dataset name")

    with pytest.raises(InvalidDatasetNameException):
        ds = deeplake.dataset("hub://test/Mnist 123")

    with pytest.raises(InvalidDatasetNameException):
        ds = deeplake.empty("hub://test/ Mnist123")

    with pytest.raises(InvalidDatasetNameException):
        ds = deeplake.like("hub://test/Mnist123 ", "hub://activeloop/mnist-train")

    with pytest.raises(InvalidDatasetNameException):
        ds = deeplake.deepcopy(
            "hub://activeloop/mnist-train", "hub://activeloop/mnist$train"
        )

    verify_dataset_name("hub://test/data-set_123")


def test_pickle_bug(local_ds):
    import pickle

    file = BytesIO()

    with local_ds as ds:
        ds.create_tensor("__temp_123")
        ds.__temp_123.extend([1, 2, 3, 4, 5])

    pickle.dump(local_ds, file)

    file.seek(0)
    ds = pickle.load(file)

    np.testing.assert_array_equal(
        ds["__temp_123"].numpy(), np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    )


def test_max_view(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")
        ds.create_tensor("pqr")

        ds.abc.extend([1, 2, 3, 4])
        ds.xyz.extend([1, 2, 3])
        ds.pqr.extend([1, 2])

    expected = {
        "abc": [[1], [2], [3], [4]],
        "xyz": [[1], [2], [3], []],
        "pqr": [[1], [2], [], []],
    }

    for i, sample in enumerate(ds.max_view):
        np.testing.assert_array_equal(sample.abc.numpy(), expected["abc"][i])


def test_min_view(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")
        ds.create_tensor("pqr")

        ds.abc.extend([1, 2, 3, 4])
        ds.xyz.extend([1, 2, 3])
        ds.pqr.extend([1, 2])

    expected = {
        "abc": [[1], [2]],
        "xyz": [[1], [2]],
        "pqr": [[1], [2]],
    }

    for i, sample in enumerate(ds.min_view):
        np.testing.assert_array_equal(sample.abc.numpy(), expected["abc"][i])


def test_extend_with_empty_tensor(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend([None, None, None])

        ds.create_tensor("xyz")
        ds.xyz.extend(ds.abc)
        ds.xyz.extend([ds.abc[0], ds.abc[1]])

        with pytest.raises(EmptyTensorError):
            ds.xyz.numpy()

        ds.xyz.append(1)

        data = ds.xyz.numpy(aslist=True)
        expected = [[]] * 5 + [1]

        for i in range(len(data)):
            np.testing.assert_array_equal(data[i], expected[i])


def test_np_array_in_info():
    info = deeplake.api.info.Info()
    x = np.random.random((3, 4))
    info["x"] = x
    info2 = deeplake.api.info.Info.frombuffer(info.tobytes())
    np.testing.assert_array_equal(x, info2["x"])


def test_sequence_numpy_bug(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="sequence")
        # issue was when number of samples (flattened) was a multiple of length of tensor
        ds.abc.extend([[1, 2], [1, 2, 3], [1, 2, 3, 4]])

        with pytest.raises(DynamicTensorNumpyError):
            ds.abc.numpy()

        assert ds.abc.numpy(aslist=True) == [[1, 2], [1, 2, 3], [1, 2, 3, 4]]


def test_tensor_dtype_bug(local_path):
    from nibabel.testing import data_path

    with deeplake.empty(local_path, overwrite=True) as ds:
        ds.create_tensor("abc", htype="link[nifti]", sample_compression="nii.gz")
        ds.abc.append(deeplake.link(f"{data_path}/standard.nii.gz"))

    assert ds.abc[0].numpy().shape == (4, 5, 7)
    assert ds.abc.dtype == np.dtype("uint8")

    ds2 = ds.copy(f"{local_path}_2", overwrite=True)

    assert ds2.abc[0].numpy().shape == (4, 5, 7)
    assert ds2.abc.dtype == np.dtype("uint8")


def test_iterate_with_groups(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x/y/z")

    ds["x/y/z"].extend(list(range(100)))

    for i, sample in enumerate(ds):
        assert sample["x/y"].z.is_iteration == True

    for i, sample in enumerate(ds):
        assert sample["x/y/z"].is_iteration == True


def test_shapes(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        np.testing.assert_array_equal(ds.abc.shapes(), np.zeros((0, 0)))

        ds.abc.append(np.ones((3, 4)))
        ds.abc.append(np.ones((5, 6)))
        np.testing.assert_array_equal(ds.abc.shapes(), np.array([[3, 4], [5, 6]]))

        ds.abc.append(None)
        np.testing.assert_array_equal(
            ds.abc.shapes(), np.array([[3, 4], [5, 6], [0, 0]])
        )

        ds.abc.append([])
        np.testing.assert_array_equal(
            ds.abc.shapes(), np.array([[3, 4], [5, 6], [0, 0], [0, 0]])
        )

        with pytest.raises(SampleAppendError):
            ds.abc.append(np.ones((3, 4, 5)))

        ds.abc.append(np.ones((4, 6)))
        np.testing.assert_array_equal(
            ds.abc.shapes(), np.array([[3, 4], [5, 6], [0, 0], [0, 0], [4, 6]])
        )

        np.testing.assert_array_equal(ds.abc[0].shapes(), np.array([[3, 4]]))
        np.testing.assert_array_equal(
            ds.abc[1:4].shapes(), np.array([[5, 6], [0, 0], [0, 0]])
        )
        np.testing.assert_array_equal(ds.abc[1::2].shapes(), np.array([[5, 6], [0, 0]]))


def test_shapes_sequence(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="sequence")
        np.testing.assert_array_equal(ds.abc.shapes(), np.zeros((0, 0)))

        ds.abc.append([np.ones((3, 4)), np.ones((4, 5))])
        np.testing.assert_array_equal(ds.abc.shapes(), np.array([[[3, 4], [4, 5]]]))

        ds.abc.append([np.ones((2, 3)), np.ones((3, 4))])
        np.testing.assert_array_equal(
            ds.abc.shapes(), np.array([[[3, 4], [4, 5]], [[2, 3], [3, 4]]])
        )

        ds.abc.append([None, None])
        np.testing.assert_array_equal(
            ds.abc.shapes(),
            np.array([[[3, 4], [4, 5]], [[2, 3], [3, 4]], [[0, 0], [0, 0]]]),
        )

        ds.abc.append([np.ones((2, 3)), np.ones((3, 4)), None])
        shapes = [
            np.array([[3, 4], [4, 5]]),
            np.array([[2, 3], [3, 4]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[2, 3], [3, 4], [0, 0]]),
        ]
        for i, shape in enumerate(ds.abc.shapes()):
            np.testing.assert_array_equal(shape, shapes[i])

        np.testing.assert_array_equal(ds.abc[0].shapes(), np.array([[[3, 4], [4, 5]]]))
        np.testing.assert_array_equal(
            ds.abc[:3].shapes(),
            np.array([[[3, 4], [4, 5]], [[2, 3], [3, 4]], [[0, 0], [0, 0]]]),
        )
        np.testing.assert_array_equal(
            ds.abc[1:3].shapes(),
            np.array([[[2, 3], [3, 4]], [[0, 0], [0, 0]]]),
        )


def test_shape_squeeze(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(np.ones((5, 10, 10, 10)))
        ds.abc.extend(np.ones((5, 10, 12, 20)))

    assert ds.abc[5:, :, 9].shape == (5, 10, 20)


def test_slice_shape_interval(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(np.ones((2, 1, 2, 3)))
        ds.abc.extend(np.ones((2, 3, 4, 5)))
        ds.abc.extend(np.ones((2, 5, 6, 2)))

    assert ds.abc.shape_interval == ShapeInterval((6, 1, 2, 2), (6, 5, 6, 5))
    assert ds[:4].abc.shape_interval == ShapeInterval((4, 1, 2, 3), (4, 3, 4, 5))
    assert ds[2:].abc.shape_interval == ShapeInterval((4, 3, 4, 2), (4, 5, 6, 5))

    with memory_ds as ds:
        ds.create_tensor("regular_seq", htype="sequence")
        ds.regular_seq.extend(np.ones((2, 2, 1, 2, 3)))
        ds.regular_seq.extend(np.ones((2, 2, 3, 4, 5)))
        ds.regular_seq.extend(np.ones((2, 2, 5, 6, 2)))

    assert ds.regular_seq.shape_interval == ShapeInterval(
        (6, 2, 1, 2, 2), (6, 2, 5, 6, 5)
    )
    assert ds.regular_seq[:4].shape_interval == ShapeInterval(
        (4, 2, 1, 2, 3), (4, 2, 3, 4, 5)
    )
    assert ds.regular_seq[2:].shape_interval == ShapeInterval(
        (4, 2, 3, 4, 2), (4, 2, 5, 6, 5)
    )

    with memory_ds as ds:
        ds.create_tensor("irregular_seq", htype="sequence")
        ds.irregular_seq.extend([np.ones((2, 1, 2, 3)), np.ones((3, 3, 4, 5))])
        ds.irregular_seq.extend([np.ones((3, 5, 6, 2)), np.ones((4, 7, 8, 9))])
        ds.irregular_seq.extend(np.ones((2, 1, 5, 5, 3)))

    assert ds.irregular_seq.shape_interval == ShapeInterval(
        (6, 1, 1, 2, 2), (6, 4, 7, 8, 9)
    )
    assert ds.irregular_seq[:4].shape_interval == ShapeInterval(
        (4, 2, 1, 2, 2), (4, 4, 7, 8, 9)
    )
    assert ds.irregular_seq[2:].shape_interval == ShapeInterval(
        (4, 1, 5, 5, 2), (4, 4, 7, 8, 9)
    )


def test_non_local_org_id():
    with pytest.raises(InvalidTokenException):
        ds = deeplake.dataset("hub://test/test_dataset", org_id="test")

    with pytest.raises(InvalidTokenException):
        ds = deeplake.empty("hub://test/test_dataset", org_id="test")

    with pytest.raises(InvalidTokenException):
        ds = deeplake.load("hub://test/test_dataset", org_id="test")

    with pytest.raises(ValueError):
        ds = deeplake.like("hub://test/test_dataset", "test/test_ds", org_id="test")


def test_azure_bad_path():
    with pytest.raises(ValueError):
        ds = deeplake.empty("az://storage_account")


def test_dataset_extend_error_suggestion(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")

    with pytest.raises(SampleExtendError) as e:
        ds.extend({"abc": [1, 2, 3, 4, "abcd", 5]})

    assert (
        "If you wish to skip the samples that cause errors,"
        " please specify `ignore_errors=True`."
    ) in str(e)


@pytest.mark.flaky(retry_count=3)
def test_extend_rollbacks(local_ds, lfpw_links):
    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="jpg")
        ds.extend(
            {"images": [deeplake.read(link) for link in lfpw_links]},
            ignore_errors=True,
        )

    # Commit should work
    ds.commit()


@pytest.mark.slow
@pytest.mark.parametrize(
    "compression_args",
    [
        {"sample_compression": None},
        {"sample_compression": "jpg"},
        {"chunk_compression": "jpg"},
    ],
)
def test_tensor_extend_ignore(local_ds, lfpw_links, compression_args):
    with local_ds as ds:
        ds.create_tensor("images", htype="image", **compression_args)
        ds.create_tensor(
            "tiled_images",
            htype="image",
            tiling_threshold=1 * KB,
            max_chunk_size=1 * KB,
            **compression_args,
        )
        ds.create_tensor("seq_images", htype="sequence[image]", **compression_args)
        ds.create_tensor("link_images", htype="link[image]", **compression_args)

    images = [deeplake.read(link) for link in lfpw_links]
    ds.images.extend(images, ignore_errors=True)
    ds.tiled_images.extend(images, ignore_errors=True)

    seqs = [
        list(map(deeplake.read, lfpw_links[i : i + 2]))
        for i in range(0, len(lfpw_links), 2)
    ]
    ds.seq_images.extend(seqs, ignore_errors=True)

    links = [deeplake.link(link) for link in lfpw_links]
    ds.link_images.extend(links, ignore_errors=True)

    # Commit should work
    ds.commit()


def test_change_htype(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("images", sample_compression="jpg")
        ds.images.extend(np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))

        ds.create_tensor("labels")
        ds.labels.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        ds.create_tensor("boxes")
        ds.boxes.extend(np.random.randn(10, 5, 4))

        ds.create_tensor("boxes_3d")
        ds.boxes_3d.extend(np.random.randn(10, 5, 8))

        ds.create_tensor("embeddings")
        ds.embeddings.extend(np.random.randn(10, 1536).astype(np.float32))

        mask = np.zeros((10, 100, 100, 5), dtype=bool)
        mask[:, :, :512, 1] = 1
        ds.create_tensor("masks")
        ds.masks.extend(mask)
        ds.create_tensor("image_masks", htype="image", sample_compression=None)
        ds.image_masks.extend(mask)

        ds.create_tensor("keypoints")
        ds.keypoints.extend(np.zeros((10, 9, 5)))

        ds.create_tensor("points")
        ds.points.extend(np.zeros((10, 5, 3)))

    ds.images.htype = "image"
    ds.labels.htype = "class_label"
    ds.boxes.htype = "bbox"
    ds.boxes_3d.htype = "bbox.3d"
    ds.embeddings.htype = "embedding"
    ds.masks.htype = "binary_mask"
    ds.image_masks.htype = "binary_mask"
    ds.keypoints.htype = "keypoints_coco"
    ds.points.htype = "point"

    with local_ds_generator() as ds:
        assert ds.images.htype == "image"
        assert ds.labels.htype == "class_label"
        assert ds.boxes.htype == "bbox"
        assert ds.boxes_3d.htype == "bbox.3d"
        assert ds.embeddings.htype == "embedding"
        assert ds.masks.htype == "binary_mask"
        assert ds.image_masks.htype == "binary_mask"
        assert ds.keypoints.htype == "keypoints_coco"
        assert ds.points.htype == "point"


def test_change_htype_fail(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("images")
        ds.images.extend(np.zeros((10, 5, 5, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.images.htype = "image"

        ds.create_tensor("images2")
        ds.images2.extend(np.zeros((10, 5, 5, 6)))
        with pytest.raises(IncompatibleHtypeError):
            ds.images2.htype = "image"

        ds.create_tensor("labels")
        ds.labels.extend(np.ones((10, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.labels.htype = "class_label"

        ds.create_tensor("boxes")
        ds.boxes.extend(np.zeros((10, 5, 5, 2)))
        with pytest.raises(IncompatibleHtypeError):
            ds.boxes.htype = "bbox"
        with pytest.raises(IncompatibleHtypeError):
            ds.boxes.htype = "bbox.3d"

        ds.create_tensor("boxes2")
        ds.boxes2.extend(np.zeros((10, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.boxes2.htype = "bbox"
        with pytest.raises(IncompatibleHtypeError):
            ds.boxes2.htype = "bbox.3d"

        # bad dtype
        ds.create_tensor("embeddings")
        ds.embeddings.extend(np.zeros((1, 1536)))
        with pytest.raises(IncompatibleHtypeError):
            ds.embeddings.htype = "embedding"

        ds.create_tensor("masks")
        ds.masks.extend(np.zeros((10, 5, 5, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.masks.htype = "binary_mask"

        ds.create_tensor("keypoints")
        ds.keypoints.extend(np.zeros((10, 5, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.keypoints.htype = "keypoints_coco"

        ds.create_tensor("keypoints2")
        ds.keypoints2.extend(np.zeros((10, 10, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.keypoints2.htype = "keypoints_coco"

        ds.create_tensor("points")
        ds.points.extend(np.zeros((10, 5, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.points.htype = "point"

        ds.create_tensor("points2")
        ds.points2.extend(np.zeros((10, 5, 5)))
        with pytest.raises(IncompatibleHtypeError):
            ds.points2.htype = "point"

        with pytest.raises(ValueError):
            ds.images.htype = "link[image]"

        with pytest.raises(ValueError):
            ds.images.htype = "sequence[image]"

        ds.create_tensor("boxes3", htype="bbox")
        ds.boxes3.extend(np.zeros((10, 5, 4), dtype=np.float32))
        with pytest.raises(NotImplementedError):
            ds.boxes3.htype = "embedding"

        with pytest.raises(NotImplementedError):
            ds.images.htype = "text"

        ds.create_tensor("images3", htype="image", sample_compression="jpg")
        with pytest.raises(UnsupportedCompressionError):
            ds.images3.htype = "embedding"


def test_append_non_uint8_to_image(local_ds):
    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="png")
        ds.images.append(np.zeros((40, 40, 1), dtype=np.int16))
        ds.images.append(np.zeros((40, 40, 1), dtype=np.uint8))

    assert ds.images.dtype.name == "int16"


@pytest.mark.slow
def test_create_and_load_with_managed_credentials(
    hub_cloud_path: str, hub_cloud_dev_token
):
    old_environ = dict(os.environ)
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_SESSION_TOKEN", None)

    try:
        dir_name = hub_cloud_path.rsplit("/", 1)[1]
        ds = deeplake.empty(
            f"s3://deeplake-tests/{dir_name}",
            creds={"creds_key": "aws_creds"},
            org_id="testingacc2",
            token=hub_cloud_dev_token,
        )
        ds.create_tensor("id", htype="text")

        assert ds.path == f"s3://deeplake-tests/{dir_name}"

        ds = deeplake.load(
            f"s3://deeplake-tests/{dir_name}",
            creds={"creds_key": "aws_creds"},
            org_id="testingacc2",
            token=hub_cloud_dev_token,
        )
        assert "id" in ds.tensors

        deeplake.delete(
            f"s3://deeplake-tests/{dir_name}",
            creds={"creds_key": "aws_creds"},
            org_id="testingacc2",
            token=hub_cloud_dev_token,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
