import numpy as np

import hub
from hub.schema import Image
from hub.utils import Timer


def main():
    with Timer("Time"):
        schema = {
            "image": Image(
                (None, None, 4),
                dtype="uint8",
                chunks=(1, 2048, 2048, 4),
                max_shape=(100000, 100000, 4),
            )
        }
        ds = hub.Dataset(
            "./data/examples/big_image", mode="w", schema=schema, shape=(10000,)
        )

        print(ds["image"].shape, ds["image"].dtype)

        ds["image", 3, 0:2048, 0:2048] = np.ones(
            (2048, 2048, 4), dtype="uint8"
        )  # single chunk read/write
        print(ds._tensors["/image"].get_shape((3,)))
        ds.flush()


if __name__ == "__main__":
    main()
