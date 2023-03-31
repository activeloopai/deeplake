from deeplake.util.exceptions import SampleAppendError
from nibabel.testing import data_path  # type: ignore

import nibabel as nib  # type: ignore
import numpy as np

import deeplake
import pytest
import os


def test_nifti(memory_ds):
    with memory_ds as ds:
        # 4D .nii.gz
        ds.create_tensor("nii_gz_4d", htype="nifti", sample_compression="nii.gz")

        nii_gz_4d = os.path.join(data_path, "example4d.nii.gz")
        sample = deeplake.read(nii_gz_4d)
        img = nib.load(nii_gz_4d)

        assert sample.shape == img.shape
        np.testing.assert_array_equal(sample.array, img.get_fdata())

        ds.nii_gz_4d.append(sample)
        np.testing.assert_array_equal(ds.nii_gz_4d.numpy()[0], img.get_fdata())
        assert ds.nii_gz_4d.shape == (1, *sample.shape)

        # 3D .nii.gz
        ds.create_tensor("nii_gz", htype="nifti", sample_compression="nii.gz")

        nii_gz = os.path.join(data_path, "standard.nii.gz")
        sample = deeplake.read(nii_gz)
        img = nib.load(nii_gz)

        assert sample.shape == img.shape
        np.testing.assert_array_equal(sample.array, img.get_fdata())

        ds.nii_gz.append(sample)
        np.testing.assert_array_equal(ds.nii_gz.numpy()[0], img.get_fdata())
        assert ds.nii_gz.shape == (1, *sample.shape)

        # 3D nii
        ds.create_tensor("nii", htype="nifti", sample_compression="nii")

        nii = os.path.join(data_path, "anatomical.nii")
        sample = deeplake.read(nii)
        img = nib.load(nii)

        assert sample.shape == img.shape
        np.testing.assert_array_equal(sample.array, img.get_fdata())

        ds.nii.append(sample)
        np.testing.assert_array_equal(ds.nii.numpy()[0], img.get_fdata())
        assert ds.nii.shape == (1, *sample.shape)


def test_nifti_sample_info(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="nifti", sample_compression="nii.gz")
        ds.abc.append(deeplake.read(os.path.join(data_path, "example4d.nii.gz")))

        sample_info = ds.abc[0].sample_info
        for key in ("affine", "zooms"):
            assert key in sample_info


def test_nifti_2(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("nifti2", htype="nifti", sample_compression="nii.gz")

        nifti2_file = os.path.join(data_path, "example_nifti2.nii.gz")
        sample = deeplake.read(nifti2_file)
        img = nib.load(nifti2_file)

        assert sample.shape == img.shape
        np.testing.assert_array_equal(sample.array, img.get_fdata())

        ds.nifti2.append(sample)
        np.testing.assert_array_equal(ds.nifti2.numpy()[0], img.get_fdata())
        assert ds.nifti2.shape == (1, *sample.shape)


def test_nifti_raw_compress(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("abc", htype="nifti", sample_compression="nii.gz")

        with pytest.raises(SampleAppendError):
            ds.abc.append(np.ones((40, 40, 10)))

        ds.create_tensor("xyz", htype="nifti", sample_compression=None)
        ds.xyz.append(np.ones((40, 40, 10)))

        np.testing.assert_array_equal(ds.xyz[0].numpy(), np.ones((40, 40, 10)))


def test_nifti_cloud(memory_ds, s3_root_storage):
    with memory_ds as ds:
        ds.add_creds_key("ENV")
        ds.populate_creds("ENV", from_environment=True)
        nii_gz_4d = os.path.join(data_path, "example4d.nii.gz")
        img = nib.load(nii_gz_4d)
        with open(nii_gz_4d, "rb") as f:
            data = f.read()
        s3_root_storage["example4d.nii.gz"] = data

        ds.create_tensor("abc", htype="nifti", sample_compression="nii.gz")
        ds.create_tensor(
            "nifti_linked", htype="link[nifti]", sample_compression="nii.gz"
        )
        ds.abc.append(
            deeplake.read(f"{s3_root_storage.root}/example4d.nii.gz", verify=True)
        )
        ds.nifti_linked.append(
            deeplake.link(f"{s3_root_storage.root}/example4d.nii.gz", creds_key="ENV")
        )

        assert ds.abc[0].numpy().shape == img.shape
        assert ds.nifti_linked[0].numpy().shape == img.shape

        del s3_root_storage["example4d.nii.gz"]
