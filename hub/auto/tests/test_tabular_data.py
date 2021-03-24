import json
import os
import shutil
import zipfile
from pathlib import PosixPath

import hub
import numpy as np
import pytest
from hub.auto.tests.util import get_dataset_store
from hub.utils import pandas_loaded


def assert_conversion(tag, dataset_shape, max_review_shape, max_filename_shape):

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

    if dataset_shape is not None:
        assert dataset_shape == ds.shape

    if max_review_shape is not None:
        actual_max_review_shape = np.max(ds["Review"].shape)
        assert max_review_shape == actual_max_review_shape

    if max_filename_shape is not None:
        actual_max_filename_shape = np.max(ds["Filename"].shape)
        assert max_filename_shape == actual_max_filename_shape


@pytest.mark.skipif(not pandas_loaded(), reason="requires pandas to be loaded")
def test_class_sample_single_csv():
    tag = "tabular/single_csv"
    assert_conversion(
        tag, dataset_shape=(7,), max_review_shape=13704, max_filename_shape=14
    )


@pytest.mark.skipif(not pandas_loaded(), reason="requires pandas to be loaded")
def test_class_sample_multiple_csv():
    tag = "tabular/multiple_csv"
    assert_conversion(
        tag, dataset_shape=(25000,), max_review_shape=13704, max_filename_shape=14
    )
