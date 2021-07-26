import hub
import numpy as np
from typing import Any, List, Tuple

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import MemoryProvider, LRUCache
from hub.core.chunk_engine import ChunkEngine
from hub.core.dataset import Dataset

from hub.constants import MB

from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import InvalidTransformOutputError, TensorMismatchError
from hub.util.keys import get_tensor_meta_key, get_chunk_id_encoder_key


def transform_sample(
    sample: Any,
    pipeline,
) -> List[dict]:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.
    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline: The Sequence of functions to apply on the sample.
        kwarg_list: A list of kwargs to be used with functions in the pipeline.
    Returns:
        List[Dict]: Containing a dictionary of all the output samples generated.
    """
    result = sample
    for index in range(len(pipeline)):
        fn = pipeline.transform_functions[index].func
        args = pipeline.transform_functions[index].args
        kwargs = pipeline.transform_functions[index].kwargs
        if isinstance(result, (list, tuple)) and index != 0:
            result = [fn(data, *args, **kwargs) for data in result]
        else:
            result = fn(result, *args, **kwargs)
        if isinstance(result, list):
            result = flatten_list_of_list(result)
        verify_transform_output(result)
    return result if isinstance(result, list) else [result]


def flatten_list_of_list(ls: List) -> List:
    """Flattens list of list into 1D list"""
    items = []
    for r in ls:
        if isinstance(r, dict):
            items.append(r)
        else:
            items.extend(r)
    return items


def verify_transform_output(output):
    """Checks whether the output of a transform is valid."""
    if isinstance(output, (list, tuple)):
        for item in output:
            if not isinstance(item, dict):
                raise InvalidTransformOutputError(item)
    elif not isinstance(output, dict):
        raise InvalidTransformOutputError


def store_shard(transform_input: Tuple):
    """Takes a shard of the original data and iterates through it, producing chunks."""

    # TODO: make this function simpler, shift stuff to core
    (
        data_shard,
        output_storage,
        tensors,
        pipeline,
    ) = transform_input

    chunk_engines = {
        t: ChunkEngine(t, LRUCache(MemoryProvider(), output_storage, 32 * MB))
        for t in tensors
    }

    # storing the metas in memory to merge later
    all_chunk_engines = {}
    all_caches = {}
    for tensor in tensors:
        memory_cache = LRUCache(MemoryProvider(), MemoryProvider(), 32 * MB)
        chunk_engine = chunk_engines[tensor]
        existing_meta = chunk_engine.tensor_meta
        new_tensor_meta = TensorMeta(
            htype=existing_meta.htype,
            dtype=existing_meta.dtype,
            sample_compression=existing_meta.sample_compression,
        )
        meta_key = get_tensor_meta_key(tensor)
        memory_cache[meta_key] = new_tensor_meta  # type: ignore
        actual_storage = get_base_storage(chunk_engine.cache)
        new_cache = LRUCache(MemoryProvider(), actual_storage, 32 * MB)
        new_cache.autoflush = False
        chunk_size = chunk_engine.max_chunk_size
        chunk_engine = ChunkEngine(tensor, new_cache, chunk_size, memory_cache)
        all_chunk_engines[tensor] = chunk_engine
        all_caches[tensor] = new_cache

    if isinstance(data_shard, Dataset):
        base_storage = get_base_storage(data_shard.storage)
        cache_size = 32 * len(tensors) * MB
        cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
        data_shard = Dataset(
            cached_store,
            index=data_shard.index,
            read_only=data_shard.read_only,
            log_loading=False,
        )

    for i in range(len(data_shard)):
        sample = data_shard[i]
        if isinstance(sample, hub.core.dataset.Dataset):
            sample = {key: sample[key].numpy() for key in sample.tensors}
        results = transform_sample(sample, pipeline)
        for result in results:
            if set(result.keys()) != set(tensors):
                raise TensorMismatchError(list(tensors), list(result.keys()))
            for key, value in result.items():
                all_chunk_engines[key].append(value)

    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    for tensor in tensors:
        all_caches[tensor].flush()
        all_tensor_metas[tensor] = all_chunk_engines[tensor].tensor_meta
        all_chunk_id_encoders[tensor] = all_chunk_engines[tensor].chunk_id_encoder
    return all_tensor_metas, all_chunk_id_encoders


def merge_tensor_metas(all_workers_tensor_metas, ds_out):
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        tensor_meta = ds_out[tensor].meta
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            # tensor meta is empty, copy attributes from current_meta
            if len(tensor_meta.max_shape) == 0 or tensor_meta.dtype is None:
                tensor_meta.dtype = current_meta.dtype
                tensor_meta.length += current_meta.length
                tensor_meta.max_shape = current_meta.max_shape
                tensor_meta.min_shape = current_meta.min_shape

            # len of min_shape will be 0 if 0 outputs from worker
            elif len(current_meta.min_shape) != 0:
                assert tensor_meta.dtype == current_meta.dtype
                # TODO we can support this once we have ragged tensor support
                assert len(tensor_meta.max_shape) == len(current_meta.max_shape)
                assert len(tensor_meta.min_shape) == len(current_meta.min_shape)
                tensor_meta.length += current_meta.length
                tensor_meta._update_shape_interval(tuple(current_meta.max_shape))
                tensor_meta._update_shape_interval(tuple(current_meta.min_shape))
        meta_key = get_tensor_meta_key(tensor)
        ds_out[tensor].chunk_engine.cache[meta_key] = tensor_meta
    ds_out.flush()


def merge_chunk_id_encoders(all_workers_chunk_id_encoders, ds_out):
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        chunk_id_encoder = ds_out[tensor].chunk_engine.chunk_id_encoder
        offset = chunk_id_encoder.num_samples
        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            num_samples = current_chunk_id_encoder.num_samples
            encoded_ids = current_chunk_id_encoder._encoded_ids
            if encoded_ids is not None:
                for encoded_id in encoded_ids:
                    encoded_id[1] += offset
                    if chunk_id_encoder._encoded_ids is None:
                        chunk_id_encoder._encoded_ids = np.reshape(encoded_id, (-1, 2))
                    else:
                        chunk_id_encoder._encoded_ids = np.vstack(
                            [chunk_id_encoder._encoded_ids, encoded_id]
                        )
            offset += num_samples
        chunk_id_key = get_chunk_id_encoder_key(tensor)
        ds_out[tensor].chunk_engine.cache[chunk_id_key] = chunk_id_encoder
    ds_out.flush()


# def pipeline_to_list(
#     pipeline: Union[Callable, Sequence[Callable]],
#     kwargs: Optional[Union[Dict, Sequence[Dict]]] = None,
# ) -> Tuple[List[Callable], List[Dict]]:
#     """Converts pipeline and kwargs to lists. Also makes the length of both lists equal to length of pipleine."""
#     kwargs = kwargs or []
#     kwargs = list(kwargs) if isinstance(kwargs, Sequence) else [kwargs]
#     pipeline = list(pipeline) if isinstance(pipeline, Sequence) else [pipeline]

#     kwargs = list(kwargs[: len(pipeline)])
#     kwargs += [dict()] * (len(pipeline) - len(kwargs))
#     return pipeline, kwargs


# def load_updated_meta(ds_out: Dataset):
#     """Clears the dataset's cache which may contain outdated meta file and loads updated meta after transform."""
#     ds_out.clear_cache()
#     ds_out._load_meta()
