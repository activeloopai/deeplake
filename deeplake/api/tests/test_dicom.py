import deeplake
import pytest
from pydicom.data import get_testdata_file
from pydicom import dcmread
from deeplake.util.exceptions import UnsupportedCompressionError


def test_dicom_basic(memory_ds):
    ds = memory_ds
    path = get_testdata_file("MR_small.dcm")
    with ds:
        ds.create_tensor("x", htype="dicom")
        with pytest.raises(UnsupportedCompressionError):
            ds.create_tensor("y", htype="dicom", sample_compression="jpg")
        dcm = deeplake.read(path)
        assert dcm.dtype == "int16"
        assert dcm.shape == (64, 64, 1)
        ds.x.append(dcm)
        assert ds.x.data()["value"].shape == (1, 64, 64, 1)
        dcm = deeplake.read(path, verify=True)
        assert dcm.dtype == "int16"
        assert dcm.shape == (64, 64, 1)
        ds.x.append(dcm)
    assert ds.x.dtype == "int16"
    arr = ds.x.numpy()
    assert arr.dtype == "int16"
    assert arr.shape == (2, 64, 64, 1)
    for item in dcmread(path):
        if not isinstance(item.value, bytes):
            assert item.keyword in ds.x[0].sample_info


def test_dicom_mixed_dtype(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="dicom")
        dcm = deeplake.read(get_testdata_file("MR_small.dcm"))
        assert dcm.dtype == "int16"
        ds.x.append(dcm)
        dcm = deeplake.read(get_testdata_file("ExplVR_BigEnd.dcm"))
        assert dcm.dtype == "uint8"
        ds.x.append(dcm)
    arr = ds.x[:, :10, :10, :1].numpy()
    assert arr.dtype == "int16"
