import numpy as np

import hub
from hub.features import Tensor
import pytest

try:
    import ray
except:
    pass
my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": "<U20",
}


@hub.transform(schema=my_schema)
def my_transform(sample):
    return {
        "image": sample["image"].numpy() * 2,
        "label": sample["label"].numpy(),
    }

@pytest.mark.skipif(True, reason="requires tfds to be loaded")
def test_pipeline_basic():
    # ray.init(local_mode=True)
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic", mode="w", shape=(100,), schema=my_schema
    )

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"

    out_ds = my_transform(ds)
    res_ds = out_ds.store("./data/test/test_pipeline_basic_output")
    assert res_ds["label", 5].numpy() == "hello 5"
    assert (res_ds["image", 4].numpy() == 2 * np.ones((28, 28, 4), dtype="int32")).all()


if __name__ == "__main__":
    # ray.init(address="auto")
    test_pipeline_basic()