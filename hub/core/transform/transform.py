from hub.core.meta.tensor_meta import TensorMeta
from hub.util.remove_cache import remove_all_cache
from hub.util.path import storage_provider_from_path
from hub.core.storage.lru_cache import LRUCache
from hub.core.storage.memory import MemoryProvider
from hub.core.meta.index_meta import IndexMeta
from hub.core.tensor import add_samples_to_tensor
from hub.api.dataset import Dataset
from hub.constants import CHUNK_MAX_SIZE, CHUNK_MIN_TARGET, MB
from typing import Callable, Dict, List
from pathos.pools import ThreadPool
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
        **kwargs,
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
        self.workers = max(workers, 1)
        self.map = ThreadPool(nodes=workers).map

    def verify_transform_output(self, output):
        # TODO better exceptions
        if isinstance(output, (list, tuple)):
            for item in output:
                assert isinstance(item, dict)
        else:
            assert isinstance(output, dict)

    def transform_sample(self, sample):
        """Calls all the functions one after the other on a single sample.
        Can return 0 or more samples.
        """
        result = sample
        for index in range(len(self.transform_fns)):
            fn = self.transform_fns[index]
            kwargs = self.transform_args[index]
            if isinstance(result, (list, tuple)) and index != 0:
                result = [fn(data, **kwargs) for data in result]
            else:
                result = fn(result, **kwargs)
            self.verify_transform_output(result)
        return result if isinstance(result, list) else [result]

    def merge_corner_chunks(
        self, index_meta_dict, last_chunk_name, last_chunk_size, tensor, storage
    ):
        first_chunk_name, first_chunk_size = self.get_first_chunk(index_meta_dict)
        if (
            last_chunk_name
            and first_chunk_size < CHUNK_MIN_TARGET
            and first_chunk_size + last_chunk_size <= CHUNK_MAX_SIZE
        ):
            last_chunk_content: bytes = storage[f"{tensor}/chunks/{last_chunk_name}"]
            first_chunk_content: bytes = storage[f"{tensor}/chunks/{first_chunk_name}"]
            new_chunk = bytearray(last_chunk_content) + first_chunk_content
            del storage[f"{tensor}/chunks/{first_chunk_name}"]
            storage[f"{tensor}/chunks/{last_chunk_name}"] = new_chunk

            offset = last_chunk_size

            #TODO explain why this fails for sample across multiple chunks 
            for i in range(len(index_meta_dict["entries"])):
                if index_meta_dict["entries"][i]["chunk_names"] == [first_chunk_name]:
                    index_meta_dict["entries"][i]["chunk_names"] = [last_chunk_name]
                    index_meta_dict["entries"][i]["start_byte"] += offset
                    index_meta_dict["entries"][i]["end_byte"] += offset
                else:
                    break

    def get_first_chunk(self, tensor_dict):
        chunk_name = None
        size = None
        if (
            len(tensor_dict["entries"]) > 0
            and len(tensor_dict["entries"][0]["chunk_names"]) > 0
        ):
            chunk_name = tensor_dict["entries"][0]["chunk_names"][0]
            size = 0
            for entry in tensor_dict["entries"]:
                if entry["chunk_names"] == [chunk_name]:
                    size = entry["end_byte"]
                elif (
                    len(entry["chunk_names"]) > 1
                    and entry["chunk_names"][0] == chunk_name
                ):
                    size = CHUNK_MAX_SIZE
                else:
                    break
        return chunk_name, size

    def merge_sub_datasets(self, all_workers_metas, storage, tensors):
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
            20MB 4MB 31MB

        Eg3
            process 1: 4MB
            process 2: 1MB
            process 3: 2MB

            becomes:-
            7MB*
        """
        for tensor in tensors:
            final_index_meta_dict = None
            final_tensor_meta_dict = None

            index_meta_key = None
            tensor_meta_key = None

            last_chunk_name = None
            last_chunk_size = None

            for worker_meta in all_workers_metas:
                all_index_meta, all_tensor_meta = worker_meta
                current_index_meta = all_index_meta[tensor]
                current_index_meta_dict = current_index_meta.to_dict()
                current_tensor_meta = all_tensor_meta[tensor]
                current_tensor_meta_dict = current_tensor_meta.to_dict()

                self.merge_corner_chunks(
                    current_index_meta_dict, last_chunk_name, last_chunk_size, tensor, storage
                )

                if final_index_meta_dict is None:
                    final_index_meta_dict = current_index_meta_dict
                    index_meta_key = current_index_meta.key
                else:
                    final_index_meta_dict["entries"].extend(current_index_meta_dict["entries"])

                if final_tensor_meta_dict is None:
                    final_tensor_meta_dict = current_tensor_meta_dict
                    tensor_meta_key = current_tensor_meta.key
                else:
                    assert final_tensor_meta_dict["dtype"] == current_tensor_meta_dict["dtype"]

                    final_tensor_meta_dict["length"] += current_tensor_meta_dict["length"]

                    final_max_shape = final_tensor_meta_dict["max_shape"]
                    final_min_shape = final_tensor_meta_dict["min_shape"]
                    max_shape = current_tensor_meta_dict["max_shape"]
                    min_shape = current_tensor_meta_dict["min_shape"]
                    assert len(final_max_shape) == len(max_shape)
                    assert len(final_min_shape) == len(min_shape)
                    final_max_shape = [max(size1, size2) for size1, size2 in zip(final_max_shape, max_shape)]
                    final_min_shape = [min(size1, size2) for size1, size2 in zip(final_min_shape, min_shape)]

                    final_tensor_meta_dict["max_shape"] = final_max_shape
                    final_tensor_meta_dict["min_shape"] = final_min_shape

                    

                # if there was atleast one chunk before
                if (
                    len(final_index_meta_dict["entries"]) > 0
                    and len(final_index_meta_dict["entries"][-1]["chunk_names"]) > 0
                ):
                    last_chunk_name = final_index_meta_dict["entries"][-1]["chunk_names"][-1]
                    last_chunk_size = final_index_meta_dict["entries"][-1]["end_byte"]

            del storage[index_meta_key]
            new_index_meta = IndexMeta.create(tensor, storage)
            new_index_meta.from_dict(final_index_meta_dict)

            del storage[tensor_meta_key]
            new_tensor_meta = TensorMeta.create(tensor, storage)
            new_tensor_meta.from_dict(final_tensor_meta_dict)



    def store_shard(self, inp):
        """Takes a shard of the original data and iterates through it, producing chunks."""
        data_shard, storage, tensors, size = inp

        all_index_meta = {
            tensor: IndexMeta.create(tensor, MemoryProvider()) for tensor in tensors
        }

        all_tensor_meta = {
            tensor: TensorMeta.create(tensor, MemoryProvider()) for tensor in tensors
        }

        # keeping a separate cache for each tensor to prevent frequent flushing.
        # 32 MB size ensures that only full chunks are written.
        local_storage_map = {
            tensor: LRUCache(MemoryProvider(), storage, 32 * MB) for tensor in tensors
        }

        # will be simply range(len(data_shard)) after AL 1092
        for i in range(min(len(data_shard), size)):
            sample = data_shard[i]
            if isinstance(sample, Dataset):
                sample = sample.numpy()

            results = self.transform_sample(sample)  # always a list of dicts
            for result in results:
                for key, value in result.items():
                    add_samples_to_tensor(
                        value,
                        key,
                        local_storage_map[key],
                        batched=False,
                        index_meta=all_index_meta[key],
                        tensor_meta=all_tensor_meta[key]
                    )

        for tensor in tensors:
            local_storage_map[tensor].flush()

        return all_index_meta, all_tensor_meta

    def store(self, ds_out):
        ds_out.flush()

        base_storage = remove_all_cache(ds_out.storage)
        tensors = ds_out.meta.tensors

        # this check doesn't work currently. Will work once AL-1092 is merged and can be uncommented
        # for tensor in tensors:
        #     assert len(ds_out[tensor]) == len(ds_out)

        shard_size = math.ceil(len(self.data_in) / self.workers)
        shards = [self.data_in[i * shard_size : (i + 1) * shard_size] for i in range(self.workers)]

        # hacky way to get around improper length of hub dataset slices
        # can be removed once AL-1092 gets done
        size_list = [shard_size for i in range(self.workers)]
        extra = shard_size * self.workers - len(self.data_in)
        if size_list:
            size_list[-1] -= extra
        

        all_workers_metas = self.map(
            self.store_shard, zip(shards, repeat(base_storage), repeat(tensors), size_list)
        )
        all_workers_metas = list(all_workers_metas)
        self.merge_sub_datasets(all_workers_metas, base_storage, tensors)

