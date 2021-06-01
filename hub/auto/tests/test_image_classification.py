import json
import os
import shutil
import zipfile
from pathlib import PosixPath

import hub
import numpy as np
from hub.auto.tests.util import get_dataset_store
from hub.auto.computer_vision.classification import multiple_image_parse


def assert_conversion(
    tag, num_samples=None, num_classes=None, image_shape=None, max_image_shape=None
):
    """
    tries to create a dataset for the kaggle_tag & then convert it into hub format.
    """

    dataset_store = get_dataset_store(tag)
    hub_dir = dataset_store / "hub"

    # delete hub dataset so conversion test can be done
    if hub_dir.is_dir():
        print("hub_dir was found (%s), deleting..." % hub_dir)
        shutil.rmtree(hub_dir)

    try:
        ds = hub.Dataset.from_path(str(dataset_store))
    except Exception:
        assert False

    print("dataset obj:", ds)
    assert ds is not None

    assert hub_dir.is_dir(), hub_dir

    # validate num samples
    if num_samples is not None:
        assert num_samples == ds.shape[0]

    # validate num classes
    if num_classes is not None:
        actual_num_classes = len(np.unique(ds["label"].compute()))
        assert num_classes == actual_num_classes

    # validate image shape (this is for when all images are the same shape)
    actual_image_shape = ds["image"].shape
    if image_shape is not None:
        expected_image_shape = np.array((num_samples, *image_shape))
        assert np.array_equal(expected_image_shape, actual_image_shape)

    # validate image max shape (this is for when not all images are the same shape)
    if max_image_shape is not None:
        expected_max_image_shape = np.array((*max_image_shape,))
        actual_max_image_shape = np.max(actual_image_shape, axis=0)
        assert np.array_equal(expected_max_image_shape, actual_max_image_shape)


def test_class_sample_same_shapes():
    tag = "image_classification/class_sample_same_shapes"
    assert_conversion(tag, num_samples=9, num_classes=3, image_shape=(256, 256, 4))


def test_class_sample_different_shapes():
    tag = "image_classification/class_sample_different_shapes"
    assert_conversion(
        tag, num_samples=10, num_classes=3, max_image_shape=(768, 1024, 4)
    )


def test_auto_multiple_dataset_parser():
    path_to_data = (
        "hub/auto/tests/dummy_data/image_classification/class_sample_multiple_folder"
    )

    keys = tuple(
        multiple_image_parse(path_to_data, scheduler="single", workers=1).keys()
    )

    assert keys == ("train", "val") or keys == ("val", "train")
