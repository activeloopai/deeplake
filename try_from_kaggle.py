from hub.api.dataset import Dataset
from hub.auto.unstructured.image_classification import ImageClassification
from hub.auto.unstructured.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError
import pytest
import os
import hub


def test_from_kaggle_simple():
    path = "./test_kaggle/source"
    destination = "./test_kaggle/destination"

    ds = hub.dataset.from_kaggle(
        tag="thisiseshan/bird-classes",
        src=path,
        dest=destination,
        src_creds=None,
        overwrite=False,
    )

    assert list(ds.tensors.keys()) == [
        "test/images",
        "test/labels",
        "train/images",
        "train/labels",
    ]
    assert ds["test/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["test/labels"].numpy().shape == (3,)
    assert ds["test/labels"].meta.class_names == ("class0", "class1", "class2")

    assert ds["train/images"].numpy().shape == (3, 200, 200, 3)
    assert ds["train/labels"].numpy().shape == (3,)
    assert ds["train/labels"].meta.class_names == ("class0", "class1", "class2")