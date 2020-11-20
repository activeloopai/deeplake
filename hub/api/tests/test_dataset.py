import numpy as np
import pytest

import hub.api.dataset as dataset
from hub.features import Tensor
from hub.utils import gcp_creds_exist, s3_creds_exist, azure_creds_exist

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

    ds["first"][0] = 2.3
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
    assert ds["second"][0].numpy() != 2.3
    ds.close()

    ds = Dataset(
        url="./data/test/test_dataset_append_and_read",
        mode="r",
    )
    ds.delete()
    ds.close()

    # TODO Add case when non existing dataset is opened in read mode


def test_dataset(url="./data/test/dataset", token=None):
    ds = Dataset(url, token=token, shape=(10000,), mode="w", schema=my_schema)

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
    "image": Tensor((10, 1920, 1080, 3), "uint8", chunks=(6, 5, 1080, 1080, 3)),
    "label": {
        "a": Tensor((100, 200), "int32", chunks=(6, 100, 200)),
        "b": Tensor((100, 400), "int64", chunks=(6, 50, 200)),
    },
    "another_thing": Tensor(
        (100, 200), Tensor((100, 200), "uint32", chunks=(6, 100, 100, 100, 100))
    ),
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
            chunks=(100, 100, 100),
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
    from hub import Dataset, features

    Dataset(
        "./data/test/test_dataset_bug",
        shape=(4,),
        mode="w",
        schema={
            "image": features.Tensor((512, 512), dtype="float"),
            "label": features.Tensor((512, 512), dtype="float"),
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
            "image": features.Tensor((512, 512), dtype="float"),
            "label": features.Tensor((512, 512), dtype="float"),
        },
    )


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
    ds = Dataset(schema=dt, shape=(20,), url="./data/test/model", mode="w")

    assert ds["first", 0].numpy().shape == (100, 100)
    assert ds["first", 0:1].numpy().shape == (1, 100, 100)
    assert ds[0]["first"].numpy().shape == (100, 100)
    assert ds[0:1]["first"].numpy().shape == (1, 100, 100)


def test_tensorview_slicing():
    dt = {"first": Tensor(shape=(None, None), max_shape=(250, 300))}
    ds = Dataset(schema=dt, shape=(20,), url="./data/test/model", mode="w")
    tv = ds["first", 5:6, 7:10, 9:10]
    assert tv.numpy().shape == tv.shape == (1, 3, 1)
    tv2 = ds["first", 5:6, 7:10, 9]
    assert tv2.numpy().shape == tv2.shape == (1, 3)
    tv3 = ds["first", 5:10, 2, 3:39]
    tv4 = tv3[3:5, 5:17]
    assert tv4.numpy().shape == (2, 12)
    assert tv4.shape == [
        (12,),
        (12,),
    ]  # for dynamic_tensor multiple shapes are returned as list of shapes


if __name__ == "__main__":
    test_tensorview_slicing()
    test_datasetview_slicing()
    test_dataset()
    test_dataset2()
