from hub.core.meta.encode.chunk_id import ChunkIdEncoder
import hub
import numpy as np
from typing import Any, Dict, List, Tuple

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import MemoryProvider, LRUCache
from hub.core.chunk_engine import ChunkEngine
from hub.core.dataset import Dataset

from hub.constants import MB

from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    InvalidTransformOutputError,
    TensorMismatchError,
)
from hub.util.keys import get_tensor_meta_key, get_chunk_id_encoder_key
from hub.core.transform.transform_shard import TransformDatasetShard


def transform_sample(
    sample: Any,
    pipeline,
) -> TransformDatasetShard:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.
    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline: The Sequence of functions to apply on the sample.
        kwarg_list: A list of kwargs to be used with functions in the pipeline.
    Returns:
        TransformDatasetShard: A dataset shard containing all the samples that were generated.
    """
    result = sample
    for index in range(len(pipeline)):
        transform_fn = pipeline.transform_functions[index]
        fn, args, kwargs = transform_fn.func, transform_fn.args, transform_fn.kwargs

        if isinstance(result, TransformDatasetShard):
            all_samples_out = []
            for item in result:
                samples_out = TransformDatasetShard()
                fn(item, samples_out, *args, **kwargs)
                samples_out._check_length_equal()
                all_samples_out.append(samples_out)
            result = combine_shards(all_samples_out)
            result._check_length_equal()  # TODO separate exception for this
        else:
            samples_out = TransformDatasetShard()
            fn(result, samples_out, *args, **kwargs)
            samples_out._check_length_equal()
            result = samples_out
    return result


def combine_shards(shards: List[TransformDatasetShard]):
    """Combines multiple shards into a single dataset shard"""
    final_shard = TransformDatasetShard()
    for shard in shards:
        for tensor in shard.tensors:
            final_shard[tensor].extend(shard[tensor].numpy())
    return final_shard


def store_data_slice(transform_input: Tuple):
    """Takes a slice of the original data and iterates through it, producing chunks."""
    data_slice, output_storage, tensors, pipeline = transform_input
    all_chunk_engines = create_worker_chunk_engines(tensors, output_storage)

    if isinstance(data_slice, Dataset):
        data_slice = add_cache_to_dataset_slice(data_slice)

    transform_data_slice_and_append(data_slice, pipeline, tensors, all_chunk_engines)
    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    for tensor, chunk_engine in all_chunk_engines.items():
        chunk_engine.cache.flush()
        chunk_engine.mem_cache.flush()
        all_tensor_metas[tensor] = chunk_engine.tensor_meta
        all_chunk_id_encoders[tensor] = chunk_engine.chunk_id_encoder
    return all_tensor_metas, all_chunk_id_encoders


def transform_data_slice_and_append(data_slice, pipeline, tensors, all_chunk_engines):
    """Transforms the data_slice with the pipeline and adds the resultant samples to chunk_engines."""
    for sample in data_slice:
        result = transform_sample(sample, pipeline)
        if set(result.tensors.keys()) != set(tensors):
            raise TensorMismatchError(list(tensors), list(result.tensors.keys()))
        for tensor in result.tensors:
            all_chunk_engines[tensor].extend(result[tensor].numpy_compressed())


def create_worker_chunk_engines(tensors, output_storage) -> Dict[str, ChunkEngine]:
    """Creates chunk engines corresponding to each storage for all tensors.
    These are created separately for each worker for parallel uploads.
    """
    all_chunk_engines = {}
    for tensor in tensors:
        # TODO: replace this with simply a MemoryProvider once we get rid of cachable
        memory_cache = LRUCache(MemoryProvider(), MemoryProvider(), 32 * MB)
        memory_cache.autoflush = False

        storage_cache = LRUCache(MemoryProvider(), output_storage, 32 * MB)
        storage_cache.autoflush = False

        # this chunk engine is used to retrieve actual tensor meta and chunk_size
        storage_chunk_engine = ChunkEngine(tensor, storage_cache)
        existing_meta = storage_chunk_engine.tensor_meta
        new_tensor_meta = TensorMeta(
            htype=existing_meta.htype,
            dtype=existing_meta.dtype,
            sample_compression=existing_meta.sample_compression,
        )
        meta_key = get_tensor_meta_key(tensor)
        memory_cache[meta_key] = new_tensor_meta  # type: ignore
        chunk_size = storage_chunk_engine.max_chunk_size
        storage_cache.clear_cache()
        storage_chunk_engine = ChunkEngine(
            tensor, storage_cache, chunk_size, memory_cache
        )
        all_chunk_engines[tensor] = storage_chunk_engine
    return all_chunk_engines


def add_cache_to_dataset_slice(dataset_slice):
    base_storage = get_base_storage(dataset_slice.storage)
    # 64 to account for potentially big encoder corresponding to each tensor
    # TODO: adjust this size once we get rid of cachable
    cache_size = 64 * len(dataset_slice.tensors) * MB
    cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
    dataset_slice = Dataset(
        cached_store,
        index=dataset_slice.index,
        read_only=dataset_slice.read_only,
        log_loading=False,
    )
    return dataset_slice


def merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]], ds_out
):
    """Merges tensor metas from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        tensor_meta = ds_out[tensor].meta
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            combine_metas(tensor_meta, current_meta)
        meta_key = get_tensor_meta_key(tensor)
        ds_out[tensor].chunk_engine.cache[meta_key] = tensor_meta
    ds_out.flush()


def combine_metas(ds_tensor_meta: TensorMeta, worker_tensor_meta: TensorMeta):
    """Combines the dataset's tensor meta with a single worker's tensor meta."""
    # if tensor meta is empty, copy attributes from current_meta
    if len(ds_tensor_meta.max_shape) == 0 or ds_tensor_meta.dtype is None:
        ds_tensor_meta.dtype = worker_tensor_meta.dtype
        ds_tensor_meta.length += worker_tensor_meta.length
        ds_tensor_meta.max_shape = worker_tensor_meta.max_shape
        ds_tensor_meta.min_shape = worker_tensor_meta.min_shape

    # len of min_shape will be 0 if 0 outputs from worker
    elif len(worker_tensor_meta.min_shape) != 0:
        assert ds_tensor_meta.dtype == worker_tensor_meta.dtype
        # TODO we can support this once we have ragged tensor support
        assert len(ds_tensor_meta.max_shape) == len(worker_tensor_meta.max_shape)
        assert len(ds_tensor_meta.min_shape) == len(worker_tensor_meta.min_shape)
        ds_tensor_meta.length += worker_tensor_meta.length
        ds_tensor_meta._update_shape_interval(tuple(worker_tensor_meta.max_shape))
        ds_tensor_meta._update_shape_interval(tuple(worker_tensor_meta.min_shape))


def merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]], ds_out
):
    """Merges chunk_id_encoders from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        chunk_id_encoder = ds_out[tensor].chunk_engine.chunk_id_encoder
        offset = chunk_id_encoder.num_samples

        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            num_samples = current_chunk_id_encoder.num_samples
            combine_chunk_id_encoders(
                chunk_id_encoder, current_chunk_id_encoder, offset
            )
            offset += num_samples

        chunk_id_key = get_chunk_id_encoder_key(tensor)
        ds_out[tensor].chunk_engine.cache[chunk_id_key] = chunk_id_encoder
    ds_out.flush()


def combine_chunk_id_encoders(
    ds_chunk_id_encoder: ChunkIdEncoder,
    worker_chunk_id_encoder: ChunkIdEncoder,
    offset: int,
):
    """Combines the dataset's chunk_id_encoder with a single worker's chunk_id_encoder."""
    encoded_ids = worker_chunk_id_encoder._encoded
    if encoded_ids.size != 0:
        for encoded_id in encoded_ids:
            encoded_id[1] += offset
            if ds_chunk_id_encoder._encoded.size == 0:
                ds_chunk_id_encoder._encoded = np.reshape(encoded_id, (-1, 2))
            else:
                ds_chunk_id_encoder._encoded = np.vstack(
                    [ds_chunk_id_encoder._encoded, encoded_id]
                )


def check_transform_data_in(data_in):
    """Checks whether the data_in for a transform is valid or not."""
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")


def check_transform_ds_out(ds_out):
    """Checks whether the das_out for a transform is valid or not."""
    if ds_out._read_only:
        raise InvalidOutputDatasetError
    tensors = list(ds_out.tensors)
    for tensor in tensors:
        if len(ds_out[tensor]) != len(ds_out):
            raise InvalidOutputDatasetError(
                "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor (This includes empty datasets that have 0 samples per tensor)."
            )
