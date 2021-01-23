from hub.schema.features import SchemaDict
from hub.exceptions import AdvancedSlicingNotSupported
from hub.api.sharded_datasetview import ShardedDatasetView
from hub import Dataset
import pytest


def test_sharded_dataset():
    dt = {"first": "float", "second": "float"}
    datasets = [
        Dataset(schema=dt, shape=(10,), url=f"./data/test/test_dataset/{i}", mode="w")
        for i in range(4)
    ]

    ds = ShardedDatasetView(datasets)

    ds[0]["first"] = 2.3
    assert ds[0]["second"].numpy() != 2.3
    assert ds[30]["first"].numpy() == 0
    assert len(ds) == 40
    assert ds.shape == (40,)
    assert type(ds.schema) == SchemaDict
    assert ds.__repr__() == "ShardedDatasetView(shape=(40,))"
    with pytest.raises(AdvancedSlicingNotSupported):
        ds[5:8]
    ds[4, "first"] = 3
    for _ in ds:
        pass

    ds2 = ShardedDatasetView([])
    assert ds2.identify_shard(5) == (0, 0)


if __name__ == "__main__":
    test_sharded_dataset()
