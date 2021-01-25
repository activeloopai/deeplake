"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.api.sharded_datasetview import ShardedDatasetView
from hub import Dataset


def test_sharded_dataset():
    dt = {"first": "float", "second": "float"}
    datasets = [
        Dataset(schema=dt, shape=(10,), url=f"./data/test/test_dataset/{i}", mode="w")
        for i in range(4)
    ]

    ds = ShardedDatasetView(datasets)

    ds[0]["first"] = 2.3
    assert ds[0]["second"].numpy() != 2.3
    print(ds[30]["first"].numpy())
    assert ds[30]["first"].numpy() == 0
    assert len(ds) == 40


if __name__ == "__main__":
    test_sharded_dataset()
