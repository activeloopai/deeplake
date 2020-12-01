import numpy as np

import hub
from hub.schema import Tensor

schema = {
    "image": Tensor((10, 1920, 1080, 3), "uint8"),
    "label": {
        "a": Tensor((100, 200), "int32"),
        "b": Tensor((100, 400), "int64"),
    },
}


def test_hub_open():
    ds = hub.Dataset(
        "./data/test/hub_open", token=None, shape=(10000,), mode="w", schema=schema
    )
    ds["label/a", 5, 50, 50] = 9
    assert ds["label/a", 5, 50, 50].numpy() == 9
    ds["image", 5, 4, 120:200, 150:300, :] = 3 * np.ones((80, 150, 3), "uint8")
    assert (
        ds["image", 5, 4, 120:200, 150:300, :].numpy()
        == 3 * np.ones((80, 150, 3), "uint8")
    ).all()


if __name__ == "__main__":
    test_hub_open()
