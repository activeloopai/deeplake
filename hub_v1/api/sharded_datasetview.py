"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from collections.abc import Iterable
from hub_v1.api.dataset_utils import slice_split
from hub_v1.api.compute_list import ComputeList


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

        self._schema = datasets[0].schema if datasets else None
        self.datasets = [
            ds
            if isinstance(ds.indexes, list)
            else ds.dataset[ds.indexes : ds.indexes + 1]
            for ds in datasets
        ]
        self.num_samples = sum([len(d) for d in self.datasets])

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
        """Computes shard id and returns the shard index and offset"""
        shard_id = 0
        count = 0
        for ds in self.datasets:
            count += len(ds)
            if index < count:
                return shard_id, count - len(ds)
            shard_id += 1
        return 0, 0

    def slicing(self, slice_list):
        """
        Identifies the dataset shard that should be used
        """
        shard_id, offset = self.identify_shard(slice_list[0])
        slice_list[0] = slice_list[0] - offset
        return slice_list, shard_id

    def __getitem__(self, slice_):
        if not isinstance(slice_, Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = slice_list or [slice(0, self.num_samples)]
        if isinstance(slice_list[0], int):
            # if integer it fetches the data from the corresponding dataset
            slice_list, shard_id = self.slicing(slice_list)
            slice_ = slice_list + [subpath] if subpath else slice_list
            return self.datasets[shard_id][slice_]
        else:
            # if slice it finds all the corresponding datasets included in the slice and generates tensorviews or datasetviews (depending on slice)
            # these views are stored in a ComputeList, calling compute on which will fetch data from all corresponding datasets and return a single result
            results = []
            cur_index = slice_list[0].start or 0
            cur_index = cur_index + self.num_samples if cur_index < 0 else cur_index
            cur_index = max(cur_index, 0)
            stop_index = slice_list[0].stop or self.num_samples
            stop_index = min(stop_index, self.num_samples)
            while cur_index < stop_index:
                shard_id, offset = self.identify_shard(cur_index)
                end_index = min(offset + len(self.datasets[shard_id]), stop_index)
                cur_slice_list = [
                    slice(cur_index - offset, end_index - offset)
                ] + slice_list[1:]
                current_slice = (
                    cur_slice_list + [subpath] if subpath else cur_slice_list
                )
                results.append(self.datasets[shard_id][current_slice])
                cur_index = end_index
            return ComputeList(results)

    def __setitem__(self, slice_, value) -> None:
        if not isinstance(slice_, Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = slice_list or [slice(0, self.num_samples)]
        if isinstance(slice_list[0], int):
            # if integer it assigns the data to the corresponding dataset
            slice_list, shard_id = self.slicing(slice_list)
            slice_ = slice_list + [subpath] if subpath else slice_list
            self.datasets[shard_id][slice_] = value
        else:
            # if slice it finds all the corresponding datasets and assigns slices of the value one by one
            cur_index = slice_list[0].start or 0
            cur_index = cur_index + self.num_samples if cur_index < 0 else cur_index
            cur_index = max(cur_index, 0)
            start_index = cur_index
            stop_index = slice_list[0].stop or self.num_samples
            stop_index = min(stop_index, self.num_samples)
            while cur_index < stop_index:
                shard_id, offset = self.identify_shard(cur_index)
                end_index = min(offset + len(self.datasets[shard_id]), stop_index)
                cur_slice_list = [
                    slice(cur_index - offset, end_index - offset)
                ] + slice_list[1:]
                current_slice = (
                    cur_slice_list + [subpath] if subpath else cur_slice_list
                )
                self.datasets[shard_id][current_slice] = value[
                    cur_index - start_index : end_index - start_index
                ]
                cur_index = end_index

    def __iter__(self):
        """Returns Iterable over samples"""
        for i in range(len(self)):
            yield self[i]

    @property
    def schema(self):
        return self._schema
