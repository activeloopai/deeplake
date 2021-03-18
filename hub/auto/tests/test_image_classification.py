import json
import os
import zipfile
from pathlib import PosixPath

dataset_store = PosixPath("PYTEST_TMPDIR/datasets")


def _exec_command(command):
    print("using command", command)
    print("-----------------")
    out = os.system(command)
    print("-----------------")
    print("exit code:", out)
    assert out == 0


def _download_kaggle(tag):
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
    _exec_command("%s unzip *.zip" % (setup))


def test_image_classification():
    kaggle_tag = "coloradokb/dandelionimages"
    _download_kaggle(kaggle_tag)
