import hub
from hub.utils import ray_loaded
from hub.features import Tensor, Text
import pytest

import numpy as np


my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text((None,), "int64", (20,)),
    "confidence": {"confidence": "float"},
}


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_pipeline_ray():
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic",
        mode="w",
        shape=(100,),
        schema=my_schema,
        cache=False,
    )

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"
        ds["confidence/confidence", i] = 0.2

    @hub.transform(schema=my_schema, scheduler="ray")
    def my_transform(sample, multiplier: int = 2):
        return {
            "image": sample["image"].compute() * multiplier,
            "label": sample["label"].compute(),
            "confidence": {
                "confidence": sample["confidence/confidence"].compute() * multiplier
            },
        }

    out_ds = my_transform(ds, multiplier=2)
    assert (out_ds["image", 0].compute() == 2).all()
    assert len(list(out_ds)) == 100
    out_ds.store("./data/test/test_pipeline_basic_output")


if __name__ == "__main__":
    test_pipeline_ray()