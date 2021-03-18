import json
import os
import zipfile
from pathlib import PosixPath

import hub


def get_dataset_store(tag):
    tag = tag.replace("/", "_")
    return PosixPath("PYTEST_TMPDIR/datasets") / tag


def _exec_command(command):
    print("using command", command)
    print("-----------------")
    out = os.system(command)
    print("-----------------")
    print("exit code:", out)
    assert out == 0


def _download_kaggle(tag):
    dataset_store = get_dataset_store(tag)
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    print("got credentials %s@%s" % (username, key))
    print('downloading kaggle dataset "%s" to %s' % (tag, dataset_store))
    os.makedirs(dataset_store, exist_ok=True)
    setup = "cd %s && export KAGGLE_USERNAME=%s && export KAGLE_KEY=%s &&" % (
        dataset_store,
        username,
        key,
    )
    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % (setup))


def test_image_classification():
    kaggle_tag = "coloradokb/dandelionimages"
    _download_kaggle(kaggle_tag)
    # ds = hub.Dataset.from_path(dataset_store / 'Images'
