"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from collections.abc import Iterable

from hub.api.datasetview import DatasetView
from hub.exceptions import AdvancedSlicingNotSupported


class ShardedDatasetView:
    def __init__(self, datasets: list) -> None:
        """
        | Creates a sharded simple dataset.
        | Datasets should have the schema.

        Parameters
        ----------
            datasets: list of Datasets
        """
        # TODO add schema check to make sure all datasets have the same schema

        self.datasets = datasets
        self.num_samples = sum([d.shape[0] for d in self.datasets])

    @property
    def shape(self):
        return (self.num_samples,)

    def __len__(self):
        return self.num_samples

    def __str__(self):
        return f"ShardedDatasetView(shape={str(self.shape)})"

    def __repr__(self):
        return self.__str__()

    def identify_shard(self, index) -> tuple:
        """ Computes shard id and returns the shard index and offset """
        shard_id = 0
        count = 0
        for ds in self.datasets:
            count += len(ds)
            if index < count:
                return shard_id, count - len(ds)
            shard_id += 1
        return 0, 0

    def slicing(self, slice_):
        """
        Identifies the dataset shard that should be used
        Notes:
            Features of advanced slicing are missing as one would expect from a DatasetView
            E.g. cross sharded dataset access is missing
        """
        if not isinstance(slice_, Iterable) or isinstance(slice_, str):
            slice_ = [slice_]

        slice_ = list(slice_)
        if not isinstance(slice_[0], int):
            # TODO add advanced slicing options
            raise AdvancedSlicingNotSupported()

        shard_id, offset = self.identify_shard(slice_[0])
        slice_[0] = slice_[0] - offset

        return slice_, shard_id

    def __getitem__(self, slice_) -> DatasetView:
        slice_, shard_id = self.slicing(slice_)
        return self.datasets[shard_id][slice_]

    def __setitem__(self, slice_, value) -> None:
        slice_, shard_id = self.slicing(slice_)
        self.datasets[shard_id][slice_] = value

    def __iter__(self):
        """ Returns Iterable over samples """
        for i in range(len(self)):
            yield self[i]

    @property
    def schema(self):
        return self.datasets[0].schema
