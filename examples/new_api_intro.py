import numpy as np

from hub import Dataset
from hub.features import ClassLabel, Image


def main():
    schema = {
        "image": Image((28, 28)),
        "label": ClassLabel(num_classes=10),
    }

    ds = Dataset("./data/examples/new_api_intro", shape=(1000,), schema=schema)

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28), dtype="uint8")
        ds["label", i] = 3

    print(ds["image", 5].numpy())
    print(ds["label", 100:110].numpy())
    ds.commit()


if __name__ == "__main__":
    main()
