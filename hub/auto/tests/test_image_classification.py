import json
import os
import shutil
import zipfile
from pathlib import PosixPath

import hub


def assert_conversion(kaggle_tag, hub_dir, after_download=None):
    """tries to create a dataset for the kaggle_tag & then convert it into hub format."""

    dataset_store = get_dataset_store(kaggle_tag)
    _hub_dir = hub_dir
    hub_dir = dataset_store / hub_dir / "hub"

    # delete hub dataset so conversion test can be done
    if hub_dir.is_dir():
        print("hub_dir was found (%s), deleting..." % hub_dir)
        shutil.rmtree(hub_dir)

    _download_kaggle(kaggle_tag, dataset_store)

    if after_download is not None:
        after_download(dataset_store / _hub_dir)

    try:
        ds = hub.Dataset.from_path(str(dataset_store / _hub_dir))
    except Exception:
        assert False

    print("dataset obj:", ds)
    assert ds is not None

    assert hub_dir.is_dir(), hub_dir

    # TODO: check if the hub dataset was properly uploaded


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


def _download_kaggle(tag, dataset_store):

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


def test_dandelionimages():
    kaggle_tag = "coloradokb/dandelionimages"
    assert_conversion(kaggle_tag, "Images")


def test_flowers_recognition():
    kaggle_tag = "alxmamaev/flowers-recognition"
    assert_conversion(
        kaggle_tag,
        "flowers",
        after_download=lambda path: shutil.rmtree(path / "flowers"),
    )
