from collections import defaultdict
from hub.api.dataset import Dataset
from hub.constants import MB
from typing import Callable, Dict, List
from pathos.pools import ProcessPool
from itertools import repeat
import math

CHUNK_SIZE = 16 * MB
# TODO Ensure that all outputs have the same schema


class Transform:
    def __init__(
        self,
        data_in,
        transform_fn: Callable,
        scheduler: str = "single",
        workers: int = 1,
        **kwargs
    ):
        if isinstance(data_in, Transform):
            self.data_in = data_in.data_in
            self.transform_fns: List[Callable] = data_in.transform_fns.append(
                transform_fn
            )
            self.transform_args: List = data_in.transform_args.append(kwargs)
        else:
            self.data_in = data_in
            self.transform_fns: List[Callable] = [transform_fn]
            self.transform_args: List = [kwargs]

        assert hasattr(self.data_in, "__getitem__")  # TODO better check
        assert hasattr(self.data_in, "__len__")  # TODO better check
        self.scheduler = scheduler
        self.workers = workers
        self.map = ProcessPool(nodes=workers)

    def verify_transform_output(self, output):
        # TODO better exceptions
        if isinstance(output, list):
            for item in output:
                assert isinstance(item, dict)
        else:
            assert isinstance(output, dict)

    def transform_sample(self, data):
        """Calls all the functions one after the other"""
        result = data
        for index in range(len(self.transform_fns)):
            fn = self.transform_fns[index]
            kwargs = self.transform_args[index]
            if isinstance(result, list) and index != 0:
                result = [fn(sample, **kwargs) for sample in result]
            else:
                result = fn(result, **kwargs)
            self.verify_transform_output(result)
        return result if isinstance(result, list) else [result]
        # result = sample
        # if index < len(self._func):
        #     if as_list:
        #         result = [self.transform_sample(index, it) for it in result]
        #     else:
        #         transform_fn = self.transform_fns[index]
        #         kwargs = self.transform_args[index]
        #         result = transform_fn(result, **kwargs)
        #         result = self.transform_sample(index + 1, result, isinstance(result, list))
        # result = self._unwrap(result) if isinstance(result, list) else result
        # return result

    def get_size(self, sample):
        # TODO, may exist in core
        return 10

    def write_chunk(self, key, samples, last_index):
        """Writes a chunk and returns index map of all of its items"""
        # TODO should be in core
        pass

    def merge(self, all_index_maps):
        """Merges index maps from all workers and merges the corner chunks if required

        Assuming range is between 16-32MB. n workers

        Strategy, keep combining corner chunks till you hit or exceed optimal range.

        * signifies suboptimal

        Eg1
            process 1:- 15MB 18MB 3MB  # 1 single huge 18MB sample, couldn't fit in 15MB chunk
            process 2:- 18MB 19MB 2MB
            process 3:- 31MB 17MB 5MB

            becomes:-
            15MB* 18MB 21MB 19MB 2MB* 31MB 17MB 5MB*

        Eg2
            process 1: 4MB
            process 2: 16MB 1MB
            process 3: 3MB 31MB # huge sample, couldn't fit in first chunk but fits completely in second

            becomes:-
            24MB 31MB

        Eg3
            process 1: 4MB
            process 2: 1MB
            process 3: 2MB

            becomes:-
            7MB*

        Overall from Eg 4 and 5, combine huge image with previous chunk if it reduces total number of chunks, else don't.
        """
        pass

    def store_shard(self, data_shard):
        """Takes a shard of the original data and iterates through it, producing chunks."""
        outputs: Dict[str, list] = defaultdict(list)
        current_sizes: Dict[str, int] = {}
        last_indexes: Dict[str, int] = defaultdict(int)
        local_index_map = {}

        for i in range(len(data_shard)):
            sample = data_shard[i]
            if isinstance(sample, Dataset):
                sample = sample.numpy()

            results = self.transform_sample(sample)  # always a list of dicts
            for result in results:
                for key, value in result.items():
                    outputs[key].append(value)
                    current_sizes[key] += self.get_size(value)
                    if current_sizes[key] >= CHUNK_SIZE or i == len(data_shard) - 1:
                        data = outputs[key]
                        index = last_indexes[key]
                        chunk_index_map = self.write_chunk(key, data, index)

                        # updating all maps
                        local_index_map.update(chunk_index_map)
                        last_indexes[key] += len(outputs[key])
                        outputs[key].clear()
                        current_sizes[key] = 0
        return local_index_map

    def store(self, url: str):
        size = math.ceil(len(self.data_in) / self.workers)
        shards = [self.data_in[i * size : (i + 1) * size] for i in range(self.workers)]
        all_index_maps = self.map(self.store_shard, zip(shards, repeat(url)))
        self.merge(all_index_maps)
