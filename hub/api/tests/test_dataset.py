import os
from hub.cli.auth import login_fn
from hub.exceptions import HubException
import numpy as np
import pytest

import hub.api.dataset as dataset
from hub.schema import Tensor, Text, Image
from hub.utils import (
    gcp_creds_exist,
    hub_creds_exist,
    s3_creds_exist,
    azure_creds_exist,
    transformers_loaded,
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


def test_dataset2():
    dt = {"first": "float", "second": "float"}
    ds = Dataset(schema=dt, shape=(2,), url="./data/test/test_dataset2", mode="w")
    ds.meta_information["description"] = "This is my description"

    ds["first"][0] = 2.3
    assert ds.meta_information["description"] == "This is my description"
    assert ds["second"][0].numpy() != 2.3


def test_dataset_append_and_read():
    dt = {"first": "float", "second": "float"}
    ds = Dataset(
        schema=dt,
        shape=(2,),
        url="./data/test/test_dataset_append_and_read",
        mode="a",
    )

    ds["first"][0] = 2.3
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
    ds.flush()

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
    except HubException:
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
    test_dataset("testingacc/test_dataset_private", public=False)
    test_dataset("testingacc/test_dataset_public")


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
    dic = dsv[3, "label"]
    assert (dic["a"].compute() == 5 * np.ones((100, 200))).all()
    assert (dic["d"]["e"].compute() == 3 * np.ones((5, 3))).all()


def test_tensorview_slicing():
    dt = {"first": Tensor(shape=(None, None), max_shape=(250, 300))}
    ds = Dataset(schema=dt, shape=(20,), url="./data/test/tensorivew_slicing", mode="w")
    tv = ds["first", 5:6, 7:10, 9:10]
    assert tv.numpy().shape == tuple(tv.shape) == (1, 3, 1)
    tv2 = ds["first", 5:6, 7:10, 9]
    assert tv2.numpy().shape == tuple(tv2.shape) == (1, 3)


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
    ds.commit()

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
        "./data/test_meta", shape=(10,), schema=schema, meta_information=description
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
    ds = Dataset(schema=dt, shape=(2,), url=url, mode="w", lazy=False)
    ds["text", 1] = "hello world"
    ds["second", 0] = 3.14
    ds["first", 0] = np.array([5, 6])
    assert ds["text", 1] == "hello world"
    assert ds["second", 0] == 3.14
    assert (ds["first", 0] == np.array([5, 6])).all()


def test_dataset_view_lazy():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/dsv_lazy"
    ds = Dataset(schema=dt, shape=(4,), url=url, mode="w", lazy=False)
    ds["text", 3] = "hello world"
    ds["second", 2] = 3.14
    ds["first", 2] = np.array([5, 6])
    dsv = ds[2:]
    assert dsv["text", 1] == "hello world"
    assert dsv["second", 0] == 3.14
    assert (dsv["first", 0] == np.array([5, 6])).all()


def test_datasetview_repr():
    dt = {
        "first": Tensor(shape=(2,)),
        "second": "float",
        "text": Text(shape=(None,), max_shape=(12,)),
    }
    url = "./data/test/dsv_repr"
    ds = Dataset(schema=dt, shape=(9,), url=url, mode="w", lazy=False)
    dsv = ds[2:]
    print_text = "DatasetView(Dataset(schema=SchemaDict({'first': Tensor(shape=(2,), dtype='float64'), 'second': 'float64', 'text': Text(shape=(None,), dtype='int64', max_shape=(12,))})url='./data/test/dsv_repr', shape=(9,), mode='w'), slice=slice(2, 9, None))"
    assert dsv.__repr__() == print_text


if __name__ == "__main__":
    test_datasetview_repr()
    test_datasetview_get_dictionary()
    test_tensorview_slicing()
    test_datasetview_slicing()
    test_dataset()
    test_dataset_batch_write_2()
    test_append_dataset()
    test_dataset2()
    test_text_dataset()
    test_text_dataset_tokenizer()
    test_dataset_compute()
    test_dataset_view_compute()
    test_dataset_lazy()
    test_dataset_view_lazy()
    test_dataset_hub()
    test_meta_information()
