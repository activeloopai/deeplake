import hub.api.tests.test_converters
from hub.features.features import Tensor
import numpy as np
from hub.utils import tfds_loaded, tensorflow_loaded, pytorch_loaded
import pytest


@pytest.mark.skipif(not tfds_loaded(), reason="requires tfds to be loaded")
def test_from_tfds_mnist():
    import tensorflow_datasets as tfds

    with tfds.testing.mock_data(num_examples=5):
        ds = hub.Dataset.from_tfds("mnist", num=5)
        res_ds = ds.store(
            "./data/test_tfds/mnist", length=5
        )  # mock data doesn't have length, so explicitly provided
        assert res_ds["label"].numpy().tolist() == [1, 9, 2, 5, 3]


@pytest.mark.skipif(not tfds_loaded(), reason="requires tfds to be loaded")
def test_from_tfds_coco():
    import tensorflow_datasets as tfds

    with tfds.testing.mock_data(num_examples=5):
        ds = hub.Dataset.from_tfds("coco", num=5)

        res_ds = ds.store(
            "./data/test_tfds/coco", length=5
        )  # mock data doesn't have length, so explicitly provided
        assert res_ds["image_filename"].numpy().tolist() == [
            b"f dhgfdgeichbdba",
            b"dhibdaajeghaefeijch",
            b"ghd h afjj igecea",
            b"iccbhgaaehgad",
            b"dahehcihidgaifeicd",
        ]
        assert res_ds["image_id"].numpy().tolist() == [90, 38, 112, 194, 105]
        assert res_ds["objects"].numpy()[0]["label"][0:5].tolist() == [
            12,
            15,
            33,
            23,
            12,
        ]


@pytest.mark.skipif(not tensorflow_loaded(), reason="requires tensorflow to be loaded")
def test_from_tensorflow():
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(tf.range(10))
    out_ds = hub.Dataset.from_tensorflow(ds)
    res_ds = out_ds.store("./data/test_from_tf/ds1")
    assert res_ds["data"].numpy().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ds = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [5, 6]})
    out_ds = hub.Dataset.from_tensorflow(ds)
    res_ds = out_ds.store("./data/test_from_tf/ds2")
    assert res_ds["a"].numpy().tolist() == [1, 2]
    assert res_ds["b"].numpy().tolist() == [5, 6]


@pytest.mark.skipif(not tensorflow_loaded(), reason="requires tensorflow to be loaded")
def test_to_from_tensorflow():
    my_schema = {
        "image": Tensor((10, 1920, 1080, 3), "uint8"),
        "label": {
            "a": Tensor((100, 200), "int32"),
            "b": Tensor((100, 400), "int64"),
            "c": Tensor((5, 3), "uint8"),
            "d": {"e": Tensor((5, 3), "uint8")},
            "f": "float",
        },
        "named_label": "object",
    }

    ds = hub.Dataset(
        schema=my_schema, shape=(10,), url="./data/test_from_tf/ds3", mode="w"
    )
    for i in range(10):
        ds["label", "d", "e", i] = i * np.ones((5, 3))
        ds["named_label", i] = "try" + str(i)
    ds = ds.to_tensorflow()
    out_ds = hub.Dataset.from_tensorflow(ds)
    res_ds = out_ds.store(
        "./data/test_from_tf/ds4", length=10
    )  # generator has no length, argument needed
    for i in range(10):
        assert (res_ds["label", "d", "e", i].numpy() == i * np.ones((5, 3))).all()
        assert res_ds["named_label", i].numpy().decode("utf-8") == "try" + str(i)


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_to_pytorch():
    import torch

    my_schema = {
        "image": Tensor((10, 1920, 1080, 3), "uint8"),
        "label": {
            "a": Tensor((100, 200), "int32"),
            "b": Tensor((100, 400), "int64"),
            "c": Tensor((5, 3), "uint8"),
            "d": {"e": Tensor((5, 3), "uint8")},
            "f": "float",
        },
    }
    ds = hub.Dataset(
        schema=my_schema, shape=(10,), url="./data/test_from_tf/ds5", mode="w"
    )
    for i in range(10):
        ds["label", "d", "e", i] = i * np.ones((5, 3))
    ds = ds.to_pytorch()
    ds = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
    )
    for i, batch in enumerate(ds):
        assert (batch["label"]["d"]["e"].numpy() == i * np.ones((5, 3))).all()


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_from_pytorch():
    from torch.utils.data import Dataset

    class TestDataset(Dataset):
        def __init__(self, transform=None):
            self.transform = transform

        def __len__(self):
            return 12

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

        def __getitem__(self, idx):
            image = 5 * np.ones((50, 50))
            landmarks = 7 * np.ones((10, 10, 10))
            named = "testing text labels"
            sample = {
                "data": {"image": image, "landmarks": landmarks},
                "labels": {"named": named},
            }

            if self.transform:
                sample = self.transform(sample)
            return sample

    tds = TestDataset()
    ds = hub.Dataset.from_pytorch(tds)
    ds = ds.store("./data/test_from_pytorch/test1")
    assert (ds["data", "image", 3].numpy() == 5 * np.ones((50, 50))).all()
    assert (ds["data", "landmarks", 2].numpy() == 7 * np.ones((10, 10, 10))).all()
    assert ds["labels", "named", 5].numpy() == "testing text labels"


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_to_from_pytorch():
    my_schema = {
        "image": Tensor((10, 10, 3), "uint8"),
        "label": {
            # "c": Tensor((5, 3), "uint8"),
            "d": {"e": Tensor((5, 3), "uint8")},
            # "f": "float",
        },
    }
    ds = hub.Dataset(
        schema=my_schema,
        shape=(10,),
        url="./data/test_from_pytorch/test20",
        mode="w",
        cache=False,
    )
    for i in range(10):
        ds["image", i] = i * np.ones((10, 10, 3))
        ds["label", "d", "e", i] = i * np.ones((5, 3))

    ds = ds.to_pytorch()
    out_ds = hub.Dataset.from_pytorch(ds)
    res_ds = out_ds.store("./data/test_from_pytorch/test30")

    for i in range(10):
        assert (res_ds["label", "d", "e", i].numpy() == i * np.ones((5, 3))).all()


if __name__ == "__main__":
    # test_from_tensorflow()
    # test_from_tensorflow()
    # test_to_from_pytorch()
    # test_to_from_tensorflow()
    test_to_from_pytorch()
    test_from_pytorch()
