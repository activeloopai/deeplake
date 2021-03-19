import json
import os
import shutil
import zipfile
from pathlib import PosixPath

import hub
from hub.auto.tests.util import get_dataset_store


def assert_conversion(tag):
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

    # TODO: check if the hub dataset was properly uploaded


def test_class_sample():
    tag = "image_classification/class_sample"
    assert_conversion(tag)
