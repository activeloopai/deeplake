import numpy as np

import hub
from hub.schema import Image, ClassLabel
from hub.utils import Timer


schema = {
    "image": Image((28, 28), chunks=(1000, 28, 28)),
    "label": ClassLabel(num_classes=10),
}


def main():
    sample_count = 70000
    step = 10
    with Timer("Time"):

        ds = hub.Dataset(
            "./data/examples/mnist_upload_speed_benchmark",
            mode="w",
            schema=schema,
            shape=(sample_count,),
            cache=2 ** 26,
        )

        arr = (np.random.rand(step, 28, 28) * 100).astype("uint8")

        for i in range(0, sample_count, step):
            # with Timer(f"Sample {i}"):
            ds["image", i : i + step] = arr

        ds.flush()


if __name__ == "__main__":
    main()
