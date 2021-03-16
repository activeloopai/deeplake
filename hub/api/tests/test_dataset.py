"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.api.dataset_utils import slice_extract_info, slice_split
from hub.schema.class_label import ClassLabel
import os
import shutil
import cloudpickle
import pickle
from hub.cli.auth import login_fn
from hub.exceptions import DirectoryNotEmptyException
import numpy as np
import pytest
from hub import transform
from hub import load
import hub.api.dataset as dataset
from hub.schema import Tensor, Text, Image, Sequence, BBox, SchemaDict, ClassLabel
from hub.utils import (
    gcp_creds_exist,
    hub_creds_exist,
    s3_creds_exist,
    azure_creds_exist,
    transformers_loaded,
    minio_creds_exist,
)

Dataset = dataset.Dataset

my_schema = {
    "image": Tensor((10, 1920, 1080, 3), "uint8"),
    "label": {
        "a": Tensor((100, 200), "int32", compressor="lz4"),
        "b": Tensor((100, 400), "int64", compressor="zstd"),
        "c": Tensor((5, 3), "uint8"),
        "d": {"e": Tensor((5, 3), "uint8")},
    },
}


def test_dataset_2():
    dt = {"first": "float", "second": "float"}
    ds = Dataset(schema=dt, shape=(2,), url="./data/test/test_dataset2", mode="w")
    ds.meta_information["description"] = "This is my description"

    ds["first"][0] = 2.3
    assert ds.meta_information["description"] == "This is my description"
    assert ds["second"][0].numpy() != 2.3


def test_dataset_append_and_read():
    dt = {"first": "float", "second": "float"}
    os.makedirs("./data/test/test_dataset_append_and_read", exist_ok=True)
    shutil.rmtree("./data/test/test_dataset_append_and_read")

    ds = Dataset(
        schema=dt,
        shape=(2,),
        url="./data/test/test_dataset_append_and_read",
        mode="a",
    )

    ds["first"][0] = 2.3
    ds.meta_information["description"] = "This is my description"
    assert ds.meta_information["description"] == "This is my description"
    assert ds["second"][0].numpy() != 2.3
    ds.close()

    ds = Dataset(
        url="./data/test/test_dataset_append_and_read",
        mode="r",
    )
    assert ds.meta_information["description"] == "This is my description"
    ds.meta_information["hello"] = 5
    ds.delete()
    ds.close()

    # TODO Add case when non existing dataset is opened in read mode


def test_dataset(url="./data/test/dataset", token=None, public=True):
    ds = Dataset(
        url, token=token, shape=(10000,), mode="w", schema=my_schema, public=public
    )

    sds = ds[5]
    sds["label/a", 50, 50] = 2
    assert sds["label", 50, 50, "a"].numpy() == 2

    ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 100:200, 150:300, :].numpy()
        == np.ones((100, 150, 3), "uint8")
    ).all()

    ds["image", 8, 6, 500:550, 700:730] = np.ones((50, 30, 3))
    subds = ds[3:15]
    subsubds = subds[4:9]
    assert (
        subsubds["image", 1, 6, 500:550, 700:730].numpy() == np.ones((50, 30, 3))
    ).all()

    subds = ds[5:7]
    ds["image", 6, 3:5, 100:135, 700:720] = 5 * np.ones((2, 35, 20, 3))

    assert (
        subds["image", 1, 3:5, 100:135, 700:720].numpy() == 5 * np.ones((2, 35, 20, 3))
    ).all()

    ds["label", "c"] = 4 * np.ones((10000, 5, 3), "uint8")
    assert (ds["label/c"].numpy() == 4 * np.ones((10000, 5, 3), "uint8")).all()

    ds["label", "c", 2, 4] = 6 * np.ones((3))
    sds = ds["label", "c"]
    ssds = sds[1:3, 4]
    sssds = ssds[1]
    assert (sssds.numpy() == 6 * np.ones((3))).all()
    ds.save()

    sds = ds["/label", 5:15, "c"]
    sds[2:4, 4, :] = 98 * np.ones((2, 3))
    assert (ds[7:9, 4, "label", "/c"].numpy() == 98 * np.ones((2, 3))).all()

    labels = ds["label", 1:5]
    d = labels["d"]
    e = d["e"]
    e[:] = 77 * np.ones((4, 5, 3))
    assert (e.numpy() == 77 * np.ones((4, 5, 3))).all()
    ds.close()


my_schema_with_chunks = {
    "image": Tensor((10, 1920, 1080, 3), "uint8", chunks=(1, 5, 1080, 1080, 3)),
    "label": {
        "a": Tensor((100, 200), "int32", chunks=(6,)),
        "b": Tensor((100, 400), "int64", chunks=6),
    },
}


def test_dataset_with_chunks():
    ds = Dataset(
        "./data/test/dataset_with_chunks",
        token=None,
        shape=(10000,),
        mode="w",
        schema=my_schema_with_chunks,
    )
    ds["label/a", 5, 50, 50] = 8
    assert ds["label/a", 5, 50, 50].numpy() == 8
    ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 100:200, 150:300, :].numpy()
        == np.ones((100, 150, 3), "uint8")
    ).all()


def test_pickleability(url="./data/test/test_dataset_dynamic_shaped"):
    schema = {
        "first": Tensor(
            shape=(None, None),
            dtype="int32",
            max_shape=(100, 100),
            chunks=(100,),
        )
    }
    ds = Dataset(
        url=url,
        token=None,
        shape=(1000,),
        mode="w",
        schema=schema,
    )

    ds["first"][0] = np.ones((10, 10))

    pickled_ds = cloudpickle.dumps(ds)
    new_ds = pickle.loads(pickled_ds)
    assert np.all(new_ds["first"][0].compute() == ds["first"][0].compute())


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_pickleability_s3():
    test_pickleability("s3://snark-test/test_dataset_pickle_s3")


@pytest.mark.skipif(not gcp_creds_exist(), reason="requires gcp credentials")
def test_pickleability_gcs():
    test_pickleability("gcs://snark-test/test_dataset_gcs")


def test_dataset_dynamic_shaped():
    schema = {
        "first": Tensor(
            shape=(None, None),
            dtype="int32",
            max_shape=(100, 100),
            chunks=(100,),
        )
    }
    ds = Dataset(
        "./data/test/test_dataset_dynamic_shaped",
        token=None,
        shape=(1000,),
        mode="w",
        schema=schema,
    )

    ds["first", 50, 50:60, 50:60] = np.ones((10, 10), "int32")
    assert (ds["first", 50, 50:60, 50:60].numpy() == np.ones((10, 10), "int32")).all()

    ds["first", 0, :10, :10] = np.ones((10, 10), "int32")
    ds["first", 0, 10:20, 10:20] = 5 * np.ones((10, 10), "int32")
    assert (ds["first", 0, 0:10, 0:10].numpy() == np.ones((10, 10), "int32")).all()


def test_dataset_enter_exit():
    with Dataset(
        "./data/test/dataset", token=None, shape=(10000,), mode="w", schema=my_schema
    ) as ds:
        sds = ds[5]
        sds["label/a", 50, 50] = 2
        assert sds["label", 50, 50, "a"].numpy() == 2

        ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
        assert (
            ds["image", 5, 4, 100:200, 150:300, :].numpy()
            == np.ones((100, 150, 3), "uint8")
        ).all()

        ds["image", 8, 6, 500:550, 700:730] = np.ones((50, 30, 3))
        subds = ds[3:15]
        subsubds = subds[4:9]
        assert (
            subsubds["image", 1, 6, 500:550, 700:730].numpy() == np.ones((50, 30, 3))
        ).all()


def test_dataset_bug():
    from hub import Dataset, schema

    Dataset(
        "./data/test/test_dataset_bug",
        shape=(4,),
        mode="w",
        schema={
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        },
    )

    was_except = False
    try:
        Dataset("./data/test/test_dataset_bug", mode="w")
    except Exception:
        was_except = True
    assert was_except

    Dataset(
        "./data/test/test_dataset_bug",
        shape=(4,),
        mode="w",
        schema={
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        },
    )


def test_dataset_bug_1(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor(
            (None, 1920, 1080, None), "uint8", max_shape=(10, 1920, 1080, 4)
        ),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds["image", 1] = np.ones((2, 1920, 1080, 1))


def test_dataset_bug_2(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor((100, 100), "uint8"),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds["image", 0:1] = [np.zeros((100, 100))]


def test_dataset_bug_3(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor((100, 100), "uint8"),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds.close()
    ds = Dataset(url)
    ds["image", 0:1] = [np.zeros((100, 100))]


def test_dataset_wrong_append(url="./data/test/dataset", token=None):
    my_schema = {
        "image": Tensor((100, 100), "uint8"),
    }
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)
    ds.close()
    try:
        ds = Dataset(url, shape=100)
    except Exception as ex:
        assert isinstance(ex, TypeError)

    try:
        ds = Dataset(url, schema={"hello": "uint8"})
    except Exception as ex:
        assert isinstance(ex, TypeError)


def test_dataset_no_shape(url="./data/test/dataset", token=None):
    try:
        Tensor(shape=(120, 120, 3), max_shape=(120, 120, 4))
    except ValueError:
        pass


def test_dataset_batch_write():
    schema = {"image": Image(shape=(None, None, 3), max_shape=(100, 100, 3))}
    ds = Dataset("./data/batch", shape=(10,), mode="w", schema=schema)

    ds["image", 0:4] = 4 * np.ones((4, 67, 65, 3))

    assert (ds["image", 0].numpy() == 4 * np.ones((67, 65, 3))).all()
    assert (ds["image", 1].numpy() == 4 * np.ones((67, 65, 3))).all()
    assert (ds["image", 2].numpy() == 4 * np.ones((67, 65, 3))).all()
    assert (ds["image", 3].numpy() == 4 * np.ones((67, 65, 3))).all()

    ds["image", 5:7] = [2 * np.ones((60, 65, 3)), 3 * np.ones((54, 30, 3))]

    assert (ds["image", 5].numpy() == 2 * np.ones((60, 65, 3))).all()
    assert (ds["image", 6].numpy() == 3 * np.ones((54, 30, 3))).all()


def test_dataset_batch_write_2():
    schema = {"image": Image(shape=(None, None, 3), max_shape=(640, 640, 3))}
    ds = Dataset("./data/batch", shape=(100,), mode="w", schema=schema)

    ds["image", 0:14] = [np.ones((640 - i, 640, 3)) for i in range(14)]


@pytest.mark.skipif(not hub_creds_exist(), reason="requires hub credentials")
def test_dataset_hub():
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    login_fn("testingacc", password)
    test_dataset("testingacc/test_dataset_private_2", public=False)
    test_dataset("testingacc/test_dataset_public_2")


@pytest.mark.skipif(not gcp_creds_exist(), reason="requires gcp credentials")
def test_dataset_gcs():
    test_dataset("gcs://snark-test/test_dataset_gcs")


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_dataset_s3():
    test_dataset("s3://snark-test/test_dataset_s3")


@pytest.mark.skipif(not azure_creds_exist(), reason="requires azure credentials")
def test_dataset_azure():
    import os

    token = {"account_key": os.getenv("ACCOUNT_KEY")}
    test_dataset(
        "https://activeloop.blob.core.windows.net/activeloop-hub/test_dataset_azure",
        token=token,
    )


def test_datasetview_slicing():
    dt = {"first": Tensor((100, 100))}
    ds = Dataset(
        schema=dt, shape=(20,), url="./data/test/datasetview_slicing", mode="w"
    )
    assert ds["first", 0].numpy().shape == (100, 100)
    assert ds["first", 0:1].numpy().shape == (1, 100, 100)
    assert ds[0]["first"].numpy().shape == (100, 100)
    assert ds[0:1]["first"].numpy().shape == (1, 100, 100)


def test_datasetview_get_dictionary():
    ds = Dataset(
        schema=my_schema,
        shape=(20,),
        url="./data/test/datasetview_get_dictionary",
        mode="w",
    )
    ds["label", 5, "a"] = 5 * np.ones((100, 200))
    ds["label", 5, "d", "e"] = 3 * np.ones((5, 3))
    dsv = ds[2:10]
    dsv.disable_lazy()
    dic = dsv[3, "label"]
    assert (dic["a"] == 5 * np.ones((100, 200))).all()
    assert (dic["d"]["e"] == 3 * np.ones((5, 3))).all()
    dsv.enable_lazy()
    ds["label", "a"] = 9 * np.ones((20, 100, 200))
    ds["label", "d", "e"] = 11 * np.ones((20, 5, 3))
    dic2 = dsv["label"]
    assert (dic2["a"].compute() == 9 * np.ones((8, 100, 200))).all()
    assert (dic2["d"]["e"].compute() == 11 * np.ones((8, 5, 3))).all()
    dic3 = ds["label"]
    assert (dic3["a"].compute() == 9 * np.ones((20, 100, 200))).all()
    assert (dic3["d"]["e"].compute() == 11 * np.ones((20, 5, 3))).all()


def test_tensorview_slicing():
    dt = {"first": Tensor(shape=(None, None), max_shape=(250, 300))}
    ds = Dataset(schema=dt, shape=(20,), url="./data/test/tensorivew_slicing", mode="w")
    tv = ds["first", 5:6, 7:10, 9:10]
    tv.disable_lazy()
    tv.enable_lazy()
    assert tv.compute().shape == tuple(tv.shape) == (1, 3, 1)
    tv2 = ds["first", 5:6, 7:10, 9]
    assert tv2.numpy().shape == tuple(tv2.shape) == (1, 3)


def test_tensorview_iter():
    schema = {"abc": "int32"}
    ds = Dataset(
        schema=schema, shape=(20,), url="./data/test/tensorivew_slicing", mode="w"
    )
    for i in range(20):
        ds["abc", i] = i
    tv = ds["abc", 3]
    for item in tv:
        assert item.compute() == 3


def test_text_dataset():
    schema = {
        "names": Text(shape=(None,), max_shape=(1000,), dtype="int64"),
    }
    ds = Dataset("./data/test/testing_text", mode="w", schema=schema, shape=(10,))
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ds["names", 4] = text + "4"
    assert ds["names", 4].numpy() == text + "4"
    ds["names"][5] = text + "5"
    assert ds["names"][5].numpy() == text + "5"
    dsv = ds[7:9]
    dsv["names", 0] = text + "7"
    assert dsv["names", 0].numpy() == text + "7"
    dsv["names"][1] = text + "8"
    assert dsv["names"][1].numpy() == text + "8"

    schema2 = {
        "id": Text(shape=(4,), dtype="int64"),
    }
    ds2 = Dataset("./data/test/testing_text_2", mode="w", schema=schema2, shape=(10,))
    ds2[0:5, "id"] = ["abcd", "efgh", "ijkl", "mnop", "qrst"]
    assert ds2[2:4, "id"].compute() == ["ijkl", "mnop"]


def test_dataset_from_directory():
    def create_image(path_to_direcotry):
        from PIL import Image

        shape = (512, 512, 3)
        for i in range(10):
            img = np.ones(shape, dtype="uint8")
            img = Image.fromarray(img)
            img.save(os.path.join(path_to_direcotry, str(i) + ".png"))

    def data_in_dir(path_to_direcotry):
        if os.path.exists(path_to_direcotry):
            create_image(path_to_direcotry)
        else:
            os.mkdir(os.path.join(path_to_direcotry))
            create_image(path_to_direcotry)

    def root_dir_image(root):
        if os.path.exists(root):
            import shutil

            shutil.rmtree(root)
        os.mkdir(root)
        for i in range(10):
            dir_name = "data_" + str(i)
            data_in_dir(os.path.join(root, dir_name))

    def del_data(*path_to_dir):
        for i in path_to_dir:
            import shutil

            shutil.rmtree(i)

    root_url = "./data/categorical_label_data"
    store_url = "./data/categorical_label_data_store"

    root_dir_image(root_url)

    ds = Dataset.from_directory(root_url)
    ds.store(store_url)

    ds = load(store_url)

    labels = ClassLabel(names=os.listdir(root_url))
    label = os.listdir(root_url)

    assert ds["image", 0].compute().shape == (512, 512, 3)
    assert ds["label", 0].compute() == labels.str2int(label[0])
    del_data(root_url, store_url)


@pytest.mark.skipif(
    not transformers_loaded(), reason="requires transformers to be loaded"
)
def test_text_dataset_tokenizer():
    schema = {
        "names": Text(shape=(None,), max_shape=(1000,), dtype="int64"),
    }
    ds = Dataset(
        "./data/test/testing_text", mode="w", schema=schema, shape=(10,), tokenizer=True
    )
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ds["names", 4] = text + " 4"
    assert ds["names", 4].numpy() == text + " 4"
    ds["names"][5] = text + " 5"
    assert ds["names"][5].numpy() == text + " 5"
    dsv = ds[7:9]
    dsv["names", 0] = text + " 7"
    assert dsv["names", 0].numpy() == text + " 7"
    dsv["names"][1] = text + " 8"
    assert dsv["names"][1].numpy() == text + " 8"

    schema2 = {
        "id": Text(shape=(4,), dtype="int64"),
    }
    ds2 = Dataset(
        "./data/test/testing_text_2",
        mode="w",
        schema=schema2,
        shape=(10,),
        tokenizer=True,
    )
    ds2[0:5, "id"] = ["abcd", "abcd", "abcd", "abcd", "abcd"]
    assert ds2[2:4, "id"].compute() == ["abcd", "abcd"]


def test_append_dataset():
    dt = {"first": Tensor(shape=(250, 300)), "second": "float"}
    url = "./data/test/model"
    ds = Dataset(schema=dt, shape=(100,), url=url, mode="w")
    ds.append_shape(20)
    ds["first"][0] = np.ones((250, 300))

    assert len(ds) == 120
    assert ds["first"].shape[0] == 120
    assert ds["first", 5:10].shape[0] == 5
    assert ds["second"].shape[0] == 120
    ds.flush()

    ds = Dataset(url)
    assert ds["first"].shape[0] == 120
    assert ds["first", 5:10].shape[0] == 5
    assert ds["second"].shape[0] == 120


def test_meta_information():
    description = {"author": "testing", "description": "here goes the testing text"}

    description_changed = {
        "author": "changed author",
        "description": "now it's changed",
    }

    schema = {"text": Text((None,), max_shape=(1000,))}

    ds = Dataset(
        "./data/test_meta",
        shape=(10,),
        schema=schema,
        meta_information=description,
        mode="w",
    )

    some_text = ["hello world", "hello penguin", "hi penguin"]

    for i, text in enumerate(some_text):
        ds["text", i] = text

    assert type(ds.meta["meta_info"]) == dict
    assert ds.meta["meta_info"]["author"] == "testing"
    assert ds.meta["meta_info"]["description"] == "here goes the testing text"

    ds.close()


def test_dataset_compute():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/ds_compute"
    ds = Dataset(schema=dt, shape=(2,), url=url, mode="w")
    ds["text", 1] = "hello world"
    ds["second", 0] = 3.14
    ds["first", 0] = np.array([5, 6])
    comp = ds.compute()
    comp0 = comp[0]
    assert (comp0["first"] == np.array([5, 6])).all()
    assert comp0["second"] == 3.14
    assert comp0["text"] == ""
    comp1 = comp[1]
    assert (comp1["first"] == np.array([0, 0])).all()
    assert comp1["second"] == 0
    assert comp1["text"] == "hello world"


def test_dataset_view_compute():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/dsv_compute"
    ds = Dataset(schema=dt, shape=(4,), url=url, mode="w")
    ds["text", 3] = "hello world"
    ds["second", 2] = 3.14
    ds["first", 2] = np.array([5, 6])
    dsv = ds[2:]
    comp = dsv.compute()
    comp0 = comp[0]
    assert (comp0["first"] == np.array([5, 6])).all()
    assert comp0["second"] == 3.14
    assert comp0["text"] == ""
    comp1 = comp[1]
    assert (comp1["first"] == np.array([0, 0])).all()
    assert comp1["second"] == 0
    assert comp1["text"] == "hello world"


def test_dataset_lazy():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/ds_lazy"
    ds = Dataset(schema=dt, shape=(2,), url=url, mode="w")
    ds["text", 1] = "hello world"
    ds["second", 0] = 3.14
    ds["first", 0] = np.array([5, 6])
    ds.disable_lazy()
    assert ds["text", 1] == "hello world"
    assert ds["second", 0] == 3.14
    assert (ds["first", 0] == np.array([5, 6])).all()
    ds.enable_lazy()
    assert ds["text", 1].compute() == "hello world"
    assert ds["second", 0].compute() == 3.14
    assert (ds["first", 0].compute() == np.array([5, 6])).all()


def test_dataset_view_lazy():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/dsv_lazy"
    ds = Dataset(schema=dt, shape=(4,), url=url, mode="w")
    ds["text", 3] = "hello world"
    ds["second", 2] = 3.14
    ds["first", 2] = np.array([5, 6])
    dsv = ds[2:]
    dsv.disable_lazy()
    assert dsv["text", 1] == "hello world"
    assert dsv["second", 0] == 3.14
    assert (dsv["first", 0] == np.array([5, 6])).all()
    dsv.enable_lazy()
    assert dsv["text", 1].compute() == "hello world"
    assert dsv["second", 0].compute() == 3.14
    assert (dsv["first", 0].compute() == np.array([5, 6])).all()


def test_datasetview_repr():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/dsv_repr"
    ds = Dataset(schema=dt, shape=(9,), url=url, mode="w", lazy=False)
    dsv = ds[2:]
    print_text = "DatasetView(Dataset(schema=SchemaDict({'first': Tensor(shape=(2,), dtype='float64'), 'second': 'float64', 'text': Text(shape=(None,), dtype='int64', max_shape=(12,))}), url='./data/test/dsv_repr', shape=(9,), mode='w'))"
    assert dsv.__repr__() == print_text


def test_datasetview_2():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    ds = Dataset("./data/test/dsv_2/", schema=dt, shape=(9,), mode="w")
    dsv = ds[2:]
    with pytest.raises(ValueError):
        dsv[3] = np.ones((3, 5))

    with pytest.raises(KeyError):
        dsv["abc"] = np.ones((3, 5))
    dsv["second"] = np.array([0, 1, 2, 3, 4, 5, 6])
    for i in range(7):
        assert dsv[i, "second"].compute() == i


def test_dataset_3():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    ds = Dataset("./data/test/ds_3/", schema=dt, shape=(9,), mode="w")
    with pytest.raises(ValueError):
        ds[3, 8] = np.ones((3, 5))

    with pytest.raises(KeyError):
        ds["abc"] = np.ones((3, 5))
    ds["second"] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    for i in range(9):
        assert ds[i, "second"].compute() == i
    with pytest.raises(ValueError):
        ds[3, 8].compute()


def test_dataset_casting():
    my_schema = {
        "a": Tensor(shape=(1,), dtype="float64"),
    }

    @transform(schema=my_schema)
    def my_transform(annotation):
        return {
            "a": 2.4,
        }

    out_ds = my_transform(range(100))
    res_ds = out_ds.store("./data/casting")
    assert res_ds["a", 30].compute() == np.array([2.4])

    ds = Dataset(schema=my_schema, url="./data/casting2", shape=(100,))
    for i in range(100):
        ds["a", i] = 0.2
    assert ds["a", 30].compute() == np.array([0.2])

    ds2 = Dataset(schema=my_schema, url="./data/casting3", shape=(100,))
    ds2["a", 0:100] = np.ones(
        100,
    )
    assert ds2["a", 30].compute() == np.array([1])


def test_dataset_setting_shape():
    schema = {"text": Text(shape=(None,), dtype="int64", max_shape=(10,))}

    url = "./data/test/text_data"
    ds = Dataset(schema=schema, shape=(5,), url=url, mode="w")
    slice_ = slice(0, 5, None)
    key = "text"
    batch = [
        np.array("THTMLY2F9"),
        np.array("QUUVEU2IU"),
        np.array("8ZUFCYWKD"),
        "H9EDFAGHB",
        "WDLDYN6XG",
    ]
    shape = ds._tensors[f"/{key}"].get_shape_from_value([slice_], batch)
    assert shape[0][0] == [1]


def test_dataset_assign_value():
    schema = {"text": Text(shape=(None,), dtype="int64", max_shape=(10,))}
    url = "./data/test/text_data"
    ds = Dataset(schema=schema, shape=(7,), url=url, mode="w")
    slice_ = slice(0, 5, None)
    key = "text"
    batch = [
        np.array("THTMLY2F9"),
        np.array("QUUVEU2IU"),
        np.array("8ZUFCYWKD"),
        "H9EDFAGHB",
        "WDLDYN6XG",
    ]
    ds[key, slice_] = batch
    ds[key][5] = np.array("GHLSGBFF8")
    ds[key][6] = "YGFJN75NF"
    assert ds["text", 0].compute() == "THTMLY2F9"
    assert ds["text", 1].compute() == "QUUVEU2IU"
    assert ds["text", 2].compute() == "8ZUFCYWKD"
    assert ds["text", 3].compute() == "H9EDFAGHB"
    assert ds["text", 4].compute() == "WDLDYN6XG"
    assert ds["text", 5].compute() == "GHLSGBFF8"
    assert ds["text", 6].compute() == "YGFJN75NF"


simple_schema = {"num": "uint8"}


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_dataset_copy_s3_local():
    ds = Dataset(
        "./data/testing/cp_original_data_local", shape=(100,), schema=simple_schema
    )
    for i in range(100):
        ds["num", i] = 2 * i
    ds2 = ds.copy("s3://snark-test/cp_copy_data_s3_1_a")
    ds3 = ds2.copy("./data/testing/cp_copy_data_local_1")
    for i in range(100):
        assert ds2["num", i].compute() == 2 * i
        assert ds3["num", i].compute() == 2 * i
    ds.delete()
    ds2.delete()
    ds3.delete()


@pytest.mark.skipif(not gcp_creds_exist(), reason="requires gcp credentials")
def test_dataset_copy_gcs_local():
    ds = Dataset(
        "./data/testing/cp_original_ds_local_3", shape=(100,), schema=simple_schema
    )
    for i in range(100):
        ds["num", i] = 2 * i
    ds2 = ds.copy("gcs://snark-test/cp_copy_dataset_gcs_1a")
    ds3 = ds2.copy("./data/testing/cp_copy_ds_local_2")
    for i in range(100):
        assert ds2["num", i].compute() == 2 * i
        assert ds3["num", i].compute() == 2 * i
    ds.delete()
    ds2.delete()
    ds3.delete()


@pytest.mark.skipif(not azure_creds_exist(), reason="requires s3 credentials")
def test_dataset_copy_azure_local():
    token = {"account_key": os.getenv("ACCOUNT_KEY")}
    ds = Dataset(
        "https://activeloop.blob.core.windows.net/activeloop-hub/cp_original_test_ds_azure_1",
        token=token,
        shape=(100,),
        schema=simple_schema,
    )
    for i in range(100):
        ds["num", i] = 2 * i
    ds2 = ds.copy("./data/testing/cp_copy_ds_local_4")
    ds3 = ds2.copy(
        "https://activeloop.blob.core.windows.net/activeloop-hub/cp_copy_test_ds_azure_2",
        token=token,
    )
    for i in range(100):
        assert ds2["num", i].compute() == 2 * i
        assert ds3["num", i].compute() == 2 * i
    ds.delete()
    ds2.delete()
    ds3.delete()


@pytest.mark.skipif(not hub_creds_exist(), reason="requires hub credentials")
def test_dataset_copy_hub_local():
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    login_fn("testingacc", password)
    ds = Dataset("testingacc/cp_original_ds_hub_1", shape=(100,), schema=simple_schema)
    for i in range(100):
        ds["num", i] = 2 * i
    ds2 = ds.copy("./data/testing/cp_copy_ds_local_5")
    ds3 = ds2.copy("testingacc/cp_copy_dataset_testing_2")
    for i in range(100):
        assert ds2["num", i].compute() == 2 * i
        assert ds3["num", i].compute() == 2 * i
    ds.delete()
    ds2.delete()
    ds3.delete()


@pytest.mark.skipif(
    not (gcp_creds_exist() and s3_creds_exist()),
    reason="requires s3 and gcs credentials",
)
def test_dataset_copy_gcs_s3():
    ds = Dataset(
        "s3://snark-test/cp_original_ds_s3_2_a", shape=(100,), schema=simple_schema
    )
    for i in range(100):
        ds["num", i] = 2 * i
    ds2 = ds.copy("gcs://snark-test/cp_copy_dataset_gcs_2_a")
    ds3 = ds2.copy("s3://snark-test/cp_copy_ds_s3_3_a")
    for i in range(100):
        assert ds2["num", i].compute() == 2 * i
        assert ds3["num", i].compute() == 2 * i
    ds.delete()
    ds2.delete()
    ds3.delete()


def test_dataset_copy_exception():
    ds = Dataset("./data/test_data_cp", shape=(100,), schema=simple_schema)
    ds2 = Dataset("./data/test_data_cp_2", shape=(100,), schema=simple_schema)
    for i in range(100):
        ds["num", i] = i
        ds2["num", i] = 2 * i
    ds.flush()
    ds2.flush()
    with pytest.raises(DirectoryNotEmptyException):
        ds3 = ds.copy("./data/test_data_cp_2")
    ds.delete()
    ds2.delete()


def test_dataset_filter():
    def abc_filter(sample):
        return sample["ab"].compute().startswith("abc")

    my_schema = {"img": Tensor((100, 100)), "ab": Text((None,), max_shape=(10,))}
    ds = Dataset("./data/new_filter", shape=(10,), schema=my_schema)
    for i in range(10):
        ds["img", i] = i * np.ones((100, 100))
        ds["ab", i] = "abc" + str(i) if i % 2 == 0 else "def" + str(i)

    ds2 = ds.filter(abc_filter)
    assert ds2.indexes == [0, 2, 4, 6, 8]


def test_datasetview_filter():
    def abc_filter(sample):
        return sample["ab"].compute().startswith("abc")

    my_schema = {"img": Tensor((100, 100)), "ab": Text((None,), max_shape=(10,))}
    ds = Dataset("./data/new_filter", shape=(10,), schema=my_schema)
    for i in range(10):
        ds["img", i] = i * np.ones((100, 100))
        ds["ab", i] = "abc" + str(i) if i % 2 == 0 else "def" + str(i)
    dsv = ds[2:7]
    ds2 = dsv.filter(abc_filter)
    assert ds2.indexes == [2, 4, 6]
    dsv2 = ds[2]
    ds3 = dsv2.filter(abc_filter)
    assert ds3.indexes == 2


def test_dataset_filter_2():
    my_schema = {
        "fname": Text((None,), max_shape=(10,)),
        "lname": Text((None,), max_shape=(10,)),
    }
    ds = Dataset("./data/tests/filtering", shape=(100,), schema=my_schema, mode="w")
    for i in range(100):
        ds["fname", i] = "John"
        ds["lname", i] = "Doe"

    for i in [1, 3, 6, 15, 63, 96, 75]:
        ds["fname", i] = "Active"

    for i in [15, 31, 25, 75, 3, 6]:
        ds["lname", i] = "loop"

    dsv_combined = ds.filter(
        lambda x: x["fname"].compute() == "Active" and x["lname"].compute() == "loop"
    )
    tsv_combined_fname = dsv_combined["fname"]
    tsv_combined_lname = dsv_combined["lname"]
    for item in dsv_combined:
        assert item.compute() == {"fname": "Active", "lname": "loop"}
    for item in tsv_combined_fname:
        assert item.compute() == "Active"
    for item in tsv_combined_lname:
        assert item.compute() == "loop"
    dsv_1 = ds.filter(lambda x: x["fname"].compute() == "Active")
    dsv_2 = dsv_1.filter(lambda x: x["lname"].compute() == "loop")
    for item in dsv_1:
        assert item.compute()["fname"] == "Active"
    tsv_1 = dsv_1["fname"]
    tsv_2 = dsv_2["lname"]
    for item in tsv_1:
        assert item.compute() == "Active"
    for item in tsv_2:
        assert item.compute() == "loop"
    for item in dsv_2:
        assert item.compute() == {"fname": "Active", "lname": "loop"}
    assert dsv_combined.indexes == [3, 6, 15, 75]
    assert dsv_1.indexes == [1, 3, 6, 15, 63, 75, 96]
    assert dsv_2.indexes == [3, 6, 15, 75]

    dsv_3 = ds.filter(lambda x: x["lname"].compute() == "loop")
    dsv_4 = dsv_3.filter(lambda x: x["fname"].compute() == "Active")
    for item in dsv_3:
        assert item.compute()["lname"] == "loop"
    for item in dsv_4:
        assert item.compute() == {"fname": "Active", "lname": "loop"}
    assert dsv_3.indexes == [3, 6, 15, 25, 31, 75]
    assert dsv_4.indexes == [3, 6, 15, 75]

    my_schema2 = {
        "fname": Text((None,), max_shape=(10,)),
        "lname": Text((None,), max_shape=(10,)),
        "image": Image((1920, 1080, 3)),
    }
    ds = Dataset("./data/tests/filtering2", shape=(100,), schema=my_schema2, mode="w")
    with pytest.raises(KeyError):
        ds.filter(lambda x: (x["random"].compute() == np.ones((1920, 1080, 3))).all())

    for i in [1, 3, 6, 15, 63, 96, 75]:
        ds["fname", i] = "Active"
    dsv = ds.filter(lambda x: x["fname"].compute() == "Active")
    with pytest.raises(KeyError):
        dsv.filter(lambda x: (x["random"].compute() == np.ones((1920, 1080, 3))).all())


def test_dataset_filter_3():
    schema = {
        "img": Image((None, None, 3), max_shape=(100, 100, 3)),
        "cl": ClassLabel(names=["cat", "dog", "horse"]),
    }
    ds = Dataset("./data/tests/filtering_3", shape=(100,), schema=schema, mode="w")
    for i in range(100):
        ds["cl", i] = 0 if i % 5 == 0 else 1
        ds["img", i] = i * np.ones((5, 6, 3))
    ds["cl", 4] = 2
    ds_filtered = ds.filter(lambda x: x["cl"].compute() == 0)
    assert ds_filtered.indexes == [5 * i for i in range(20)]
    with pytest.raises(ValueError):
        ds_filtered["img"].compute()
    ds_filtered_2 = ds.filter(lambda x: x["cl"].compute() == 2)
    assert (ds_filtered_2["img"].compute() == 4 * np.ones((1, 5, 6, 3))).all()
    for item in ds_filtered_2:
        assert (item["img"].compute() == 4 * np.ones((5, 6, 3))).all()
        assert item["cl"].compute() == 2


def test_dataset_filter_4():
    schema = {
        "img": Image((None, None, 3), max_shape=(100, 100, 3)),
        "cl": ClassLabel(names=["cat", "dog", "horse"]),
    }
    ds = Dataset("./data/tests/filtering_4", shape=(100,), schema=schema, mode="w")
    for i in range(100):
        ds["cl", i] = 0 if i < 10 else 1
        ds["img", i] = i * np.ones((5, 6, 3))
    ds_filtered = ds.filter(lambda x: x["cl"].compute() == 0)
    assert (ds_filtered[3:8, "cl"].compute() == np.zeros((5,))).all()


def test_dataset_utils():
    with pytest.raises(TypeError):
        slice_split([5.3])
    with pytest.raises(IndexError):
        slice_extract_info(5, 3)
    with pytest.raises(ValueError):
        slice_extract_info(slice(2, 10, -2), 3)
    with pytest.raises(IndexError):
        slice_extract_info(slice(20, 100), 3)
    with pytest.raises(IndexError):
        slice_extract_info(slice(1, 20), 3)
    with pytest.raises(IndexError):
        slice_extract_info(slice(4, 1), 10)
    slice_extract_info(slice(None, 10), 20)
    slice_extract_info(slice(20, None), 50)


def test_dataset_name():
    schema = {"temp": "uint8"}
    ds = Dataset(
        "./data/test_ds_name", shape=(10,), schema=schema, name="my_dataset", mode="w"
    )
    ds.flush()
    assert ds.name == "my_dataset"
    ds2 = Dataset("./data/test_ds_name")
    ds2.rename("my_dataset_2")
    assert ds2.name == "my_dataset_2"
    ds3 = Dataset("./data/test_ds_name")
    assert ds3.name == "my_dataset_2"


def test_check_label_name():
    my_schema = {"label": ClassLabel(names=["red", "green", "blue"])}
    ds = Dataset("./data/test/dataset2", shape=(5,), mode="w", schema=my_schema)
    ds["label", 0] = 1
    ds["label", 1] = 2
    ds["label", 0] = 1
    ds["label", 1] = 2
    ds["label", 2] = 0
    assert ds.compute(label_name=True).tolist() == [
        {"label": "green"},
        {"label": "blue"},
        {"label": "red"},
        {"label": "red"},
        {"label": "red"},
    ]
    assert ds.compute().tolist() == [
        {"label": 1},
        {"label": 2},
        {"label": 0},
        {"label": 0},
        {"label": 0},
    ]
    assert ds[1].compute(label_name=True) == {"label": "blue"}
    assert ds[1].compute() == {"label": 2}
    assert ds[1:3].compute(label_name=True).tolist() == [
        {"label": "blue"},
        {"label": "red"},
    ]
    assert ds[1:3].compute().tolist() == [{"label": 2}, {"label": 0}]


@pytest.mark.skipif(not minio_creds_exist(), reason="requires minio credentials")
def test_minio_endpoint():
    token = {
        "aws_access_key_id": os.getenv("ACTIVELOOP_MINIO_KEY"),
        "aws_secret_access_key": os.getenv("ACTIVELOOP_MINIO_SECRET_ACCESS_KEY"),
        "endpoint_url": "https://play.min.io:9000",
        "region": "us-east-1",
    }

    schema = {"abc": Tensor((100, 100, 3))}
    ds = Dataset(
        "s3://bucket/random_dataset", token=token, shape=(10,), schema=schema, mode="w"
    )

    for i in range(10):
        ds["abc", i] = i * np.ones((100, 100, 3))
    ds.flush()
    for i in range(10):
        assert (ds["abc", i].compute() == i * np.ones((100, 100, 3))).all()


if __name__ == "__main__":
    test_dataset_assign_value()
    test_dataset_setting_shape()
    test_datasetview_repr()
    test_datasetview_get_dictionary()
    test_tensorview_slicing()
    test_datasetview_slicing()
    test_dataset()
    test_dataset_batch_write_2()
    test_append_dataset()
    test_dataset_2()

    test_text_dataset()
    test_text_dataset_tokenizer()
    test_dataset_compute()
    test_dataset_view_compute()
    test_dataset_lazy()
    test_dataset_view_lazy()
    test_dataset_hub()
    test_meta_information()
    test_dataset_filter_2()
    test_dataset_filter_3()
    test_pickleability()
    test_dataset_append_and_read()
    test_tensorview_iter()
    test_dataset_filter_4()
    test_datasetview_2()
    test_dataset_3()
    test_dataset_utils()
    test_check_label_name()
