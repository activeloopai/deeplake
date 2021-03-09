"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import hub.api.tests.test_converters
from hub.schema.features import Tensor
import numpy as np
from hub.utils import tfds_loaded, tensorflow_loaded, pytorch_loaded
import pytest
from hub.utils import Timer


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
        assert res_ds["image_id"].numpy().tolist() == [90, 38, 112, 194, 105]
        assert res_ds["objects"].numpy()[0]["label"][0:5].tolist() == [
            12,
            15,
            33,
            23,
            12,
        ]


@pytest.mark.skipif(not tfds_loaded(), reason="requires tfds to be loaded")
def test_from_tfds_accentdb():
    import tensorflow_datasets as tfds

    with tfds.testing.mock_data(num_examples=5):
        ds = hub.Dataset.from_tfds("accentdb", num=5)

        res_ds = ds.store(
            "./data/test_tfds/accentdb", length=5
        )  # mock data doesn't have length, so explicitly provided
        assert res_ds["audio", 0, :3].compute().tolist() == [47, 117, 192]
        assert res_ds["audio", 2, :3].compute().tolist() == [163, 254, 203]


@pytest.mark.skipif(not tfds_loaded(), reason="requires tfds to be loaded")
def test_from_tfds_robonet():
    import tensorflow_datasets as tfds

    with tfds.testing.mock_data(num_examples=5):
        ds = hub.Dataset.from_tfds("robonet", num=5)

        res_ds = ds.store(
            "./data/test_tfds/robonet", length=5
        )  # mock data doesn't have length, so explicitly provided
        assert res_ds["video", 0, 0:2, 1, 2, 0].compute().tolist() == [59, 177]
        assert res_ds["video", 2, 0:2, 1, 2, 0].compute().tolist() == [127, 62]


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
def test_to_tensorflow():
    schema = {"abc": Tensor((100, 100, 3)), "int": "uint32"}
    ds = hub.Dataset("./data/test_to_tf", shape=(10,), schema=schema)
    for i in range(10):
        ds["abc", i] = i * np.ones((100, 100, 3))
        ds["int", i] = i
    tds = ds.to_tensorflow()
    for i, item in enumerate(tds):
        assert (item["abc"].numpy() == i * np.ones((100, 100, 3))).all()
        assert item["int"] == i


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
    }

    ds = hub.Dataset(
        schema=my_schema, shape=(10,), url="./data/test_from_tf/ds4", mode="w"
    )
    for i in range(10):
        ds["label", "d", "e", i] = i * np.ones((5, 3))

    ds = ds.to_tensorflow(include_shapes=True)
    out_ds = hub.Dataset.from_tensorflow(ds)
    res_ds = out_ds.store(
        "./data/test_from_tf/ds5", length=10
    )  # generator has no length, argument needed

    for i in range(10):
        assert (res_ds["label", "d", "e", i].numpy() == i * np.ones((5, 3))).all()


@pytest.mark.skipif(not tensorflow_loaded(), reason="requires tensorflow to be loaded")
def test_to_from_tensorflow_datasetview():
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
        schema=my_schema, shape=(10,), url="./data/test_from_tf/ds4", mode="w"
    )
    for i in range(10):
        ds["label", "d", "e", i] = i * np.ones((5, 3))
    dsv = ds[5:]
    tds = dsv.to_tensorflow(include_shapes=True)
    out_ds = hub.Dataset.from_tensorflow(tds)
    res_ds = out_ds.store(
        "./data/test_from_tf/ds6", length=5
    )  # generator has no length, argument needed

    for i in range(5):
        assert (res_ds["label", "d", "e", i].numpy() == (5 + i) * np.ones((5, 3))).all()


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
    # pure conversion
    dst = ds.to_pytorch()
    dl = torch.utils.data.DataLoader(
        dst,
        batch_size=1,
    )
    for i, batch in enumerate(dl):
        assert (batch["label"]["d"]["e"].numpy() == i * np.ones((5, 3))).all()

    # with transforms and inplace=False
    def recursive_torch_tensor(label):
        for key, value in label.items():
            if type(value) is dict:
                label[key] = recursive_torch_tensor(value)
            else:
                label[key] = torch.tensor(value)
        return label

    def transform(data):
        image = torch.tensor(data["image"])
        label = data["label"]
        label = recursive_torch_tensor(label)
        return (image, label)

    dst = ds.to_pytorch(transform=transform, inplace=False)
    dl = torch.utils.data.DataLoader(
        dst,
        batch_size=1,
    )
    for i, batch in enumerate(dl):
        assert (batch[1]["d"]["e"].numpy() == i * np.ones((5, 3))).all()

    # output_type = list
    dst = ds.to_pytorch(output_type=list)
    for i, d in enumerate(dst):
        assert type(d) == list

    # output_type = tuple
    dst = ds.to_pytorch(output_type=tuple)
    for i, d in enumerate(dst):
        assert type(d) == tuple


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_to_pytorch_datasetview():
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
    # pure conversion
    dsv = ds[3:]
    ptds = dsv.to_pytorch()
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
    )
    for i, batch in enumerate(dl):
        assert (batch["label"]["d"]["e"].numpy() == (3 + i) * np.ones((5, 3))).all()

    # with transforms and inplace=False
    def recursive_torch_tensor(label):
        for key, value in label.items():
            if type(value) is dict:
                label[key] = recursive_torch_tensor(value)
            else:
                label[key] = torch.tensor(value)
        return label

    def transform(data):
        image = torch.tensor(data["image"])
        label = data["label"]
        label = recursive_torch_tensor(label)
        return (image, label)

    dst = dsv.to_pytorch(transform=transform, inplace=False)
    dl = torch.utils.data.DataLoader(
        dst,
        batch_size=1,
    )
    for i, batch in enumerate(dl):
        assert (batch[1]["d"]["e"].numpy() == (3 + i) * np.ones((5, 3))).all()

    # output_type = list
    dst = ds.to_pytorch(output_type=list)
    for i, d in enumerate(dst):
        assert type(d) == list

    # output_type = tuple
    dst = ds.to_pytorch(output_type=tuple)
    for i, d in enumerate(dst):
        assert type(d) == tuple


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
            image = 5 * np.ones((256, 256, 3))
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
    with Timer("from_pytorch"):
        ds = hub.Dataset.from_pytorch(tds)

    ds = ds.store("./data/test_from_pytorch/test1")

    assert (ds["data", "image", 3].numpy() == 5 * np.ones((256, 256, 3))).all()
    assert (ds["data", "landmarks", 2].numpy() == 7 * np.ones((10, 10, 10))).all()
    assert ds["labels", "named", 5].numpy() == "testing text labels"


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_to_from_pytorch():
    my_schema = {
        "image": Tensor((10, 10, 3), "uint8"),
        "label": {
            "c": Tensor((5, 3), "uint8"),
            "d": {"e": Tensor((5, 3), "uint8")},
            "f": "float",
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
    with Timer("storing"):
        res_ds = out_ds.store("./data/test_from_pytorch/test30")

    for i in range(10):
        assert (res_ds["label", "d", "e", i].numpy() == i * np.ones((5, 3))).all()


@pytest.mark.skipif(not pytorch_loaded(), reason="requires pytorch to be loaded")
def test_to_pytorch_bug():
    ds = hub.Dataset("activeloop/mnist", mode="r")
    data = ds.to_pytorch()


@pytest.mark.skipif(not tensorflow_loaded(), reason="requires tensorflow to be loaded")
def test_to_tensorflow_bug():
    ds = hub.Dataset("activeloop/coco_train")
    data = ds.to_tensorflow()


if __name__ == "__main__":
    with Timer("Test Converters"):
        with Timer("from MNIST"):
            test_from_tfds_mnist()

        with Timer("from COCO"):
            test_from_tfds_coco()

        with Timer("from TF"):
            test_from_tensorflow()

        with Timer("To From TF"):
            test_to_from_tensorflow()

        with Timer("To From PyTorch"):
            test_to_from_pytorch()

        with Timer("From Pytorch"):
            test_from_pytorch()
