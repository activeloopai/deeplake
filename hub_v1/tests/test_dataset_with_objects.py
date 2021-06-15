"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np

import hub_v1
from hub_v1.schema import Tensor


def test_dataset_with_objects():
    schema = {"images": Tensor(shape=(10,), dtype="object", chunks=(5,))}

    ds = hub_v1.Dataset(
        "./data/test/test_dataset_with_objects", mode="w", shape=(100,), schema=schema
    )
    ds["images", 6, 5] = np.ones((20, 30, 4), dtype="uint8")
    ds.close()


if __name__ == "__main__":
    test_dataset_with_objects()
