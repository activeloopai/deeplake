import numpy as np

from hub import Dataset
from hub.schema import ClassLabel, Image
from hub.utils import Timer


def main():
    schema = {
        "image": Image(shape=(None, None), max_shape=(28, 28)),
        "label": ClassLabel(num_classes=10),
    }
    path = "./data/examples/new_api_intro2"

    ds = Dataset(path, shape=(10,), mode="w", schema=schema)
    print(len(ds))
    for i in range(len(ds)):
        with Timer("writing single element"):
            ds["image", i] = np.ones((28, 28), dtype="uint8")
            ds["label", i] = 3

    ds.resize_shape(200)
    print(ds.shape)
    print(ds["label", 100:110].numpy())
    with Timer("Saving"):
        ds.flush()

    ds = Dataset(path)
    print(ds.schema)
    print(ds["image", 0].compute())


if __name__ == "__main__":
    main()
