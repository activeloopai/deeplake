import hub
from typing import Any, Dict, List, Tuple

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import StorageProvider, MemoryProvider, LRUCache
from hub.core.chunk_engine import ChunkEngine
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.transform.transform_dataset import TransformDataset

from hub.constants import MB

from hub.util.remove_cache import get_base_storage
from hub.util.keys import get_tensor_meta_key
from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    InvalidTransformDataset,
    TensorMismatchError,
)


def transform_sample(
    sample: Any,
    pipeline,
) -> TransformDataset:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.
    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline (Pipeline): The Sequence of functions to apply on the sample.
    Returns:
        TransformDataset: A transform dataset containing all the samples that were generated.
    """
    result = sample
    for index in range(len(pipeline)):
        transform_fn = pipeline.transform_functions[index]
        fn, args, kwargs = transform_fn.func, transform_fn.args, transform_fn.kwargs

        if isinstance(result, TransformDataset):
            all_samples_out = []
            for item in result:
                samples_out = TransformDataset()
                fn(item, samples_out, *args, **kwargs)
                validate_transform_dataset(samples_out)
                all_samples_out.append(samples_out)
            result = combine_transform_datasets(all_samples_out)
            try:
                validate_transform_dataset(result)
            except InvalidTransformDataset:
                raise InvalidTransformDataset(
                    "One or more of the TransformDatasets returned had different number of tensors. Always ensure that all outputs have exactly the same tensors and equal number of samples in each tensor."
                )
        else:
            samples_out = TransformDataset()
            fn(result, samples_out, *args, **kwargs)
            validate_transform_dataset(samples_out)
            result = samples_out
    return result


def combine_transform_datasets(datasets: List[TransformDataset]):
    """Combines multiple TransformDataset into a single transform dataset."""
    final_ds = TransformDataset()
    for ds in datasets:
        for tensor in ds.tensors:
            final_ds[tensor].extend(ds[tensor].numpy())
    return final_ds


def validate_transform_dataset(dataset: TransformDataset):
    """Cheks if the length of all the tensors is equal. Raises exception if not equal."""
    lengths = [len(dataset[tensor]) for tensor in dataset.tensors]
    if any(length != lengths[0] for length in lengths):
        raise InvalidTransformDataset


def store_data_slice(
    transform_input: Tuple,
) -> Tuple[Dict[str, TensorMeta], Dict[str, ChunkIdEncoder]]:
    """Takes a slice of the original data and iterates through it and stores it in the actual storage.
    The tensor_meta and chunk_id_encoder are not stored to the storage to prevent overwrites/race conditions b/w workers.
    They are instead stored in memory and returned."""
    data_slice, output_storage, tensors, pipeline = transform_input
    all_chunk_engines = create_worker_chunk_engines(tensors, output_storage)

    if isinstance(data_slice, hub.core.dataset.Dataset):
        data_slice = add_cache_to_dataset_slice(data_slice)

    transform_data_slice_and_append(data_slice, pipeline, tensors, all_chunk_engines)

    # retrieve the tensor metas and chunk_id_encoder from the memory
    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    for tensor, chunk_engine in all_chunk_engines.items():
        chunk_engine.cache.flush()
        chunk_engine.meta_cache.flush()
        all_tensor_metas[tensor] = chunk_engine.tensor_meta
        all_chunk_id_encoders[tensor] = chunk_engine.chunk_id_encoder
    return all_tensor_metas, all_chunk_id_encoders


def transform_data_slice_and_append(
    data_slice, pipeline, tensors: List[str], all_chunk_engines: Dict[str, ChunkEngine]
) -> None:
    """Transforms the data_slice with the pipeline and adds the resultant samples to chunk_engines."""
    for sample in data_slice:
        result = transform_sample(sample, pipeline)
        if set(result.tensors.keys()) != set(tensors):
            raise TensorMismatchError(list(tensors), list(result.tensors.keys()))
        for tensor in result.tensors:
            all_chunk_engines[tensor].extend(result[tensor].numpy_compressed())


def create_worker_chunk_engines(
    tensors: List[str], output_storage: StorageProvider
) -> Dict[str, ChunkEngine]:
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


def add_cache_to_dataset_slice(
    dataset_slice: hub.core.dataset.Dataset,
) -> hub.core.dataset.Dataset:
    base_storage = get_base_storage(dataset_slice.storage)
    # 64 to account for potentially big encoder corresponding to each tensor
    # TODO: adjust this size once we get rid of cachable
    cache_size = 64 * len(dataset_slice.tensors) * MB
    cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
    dataset_slice = hub.core.dataset.Dataset(
        cached_store,
        index=dataset_slice.index,
        read_only=dataset_slice.read_only,
        verbose=False,
    )
    return dataset_slice


def check_transform_data_in(data_in, scheduler: str) -> None:
    """Checks whether the data_in for a transform is valid or not."""
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError(
            f"The data_in to transform is invalid. It should support __getitem__ operation."
        )
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError(
            f"The data_in to transform is invalid. It should support __len__ operation."
        )
    if isinstance(data_in, hub.core.dataset.Dataset):
        base_storage = get_base_storage(data_in.storage)
        if isinstance(base_storage, MemoryProvider) and scheduler != "threaded":
            raise InvalidOutputDatasetError(
                f"Transforms with data_in as a Dataset having base storage as MemoryProvider are only supported in threaded and serial mode. Current mode is {scheduler}."
            )


def check_transform_ds_out(ds_out: hub.core.dataset.Dataset, scheduler: str) -> None:
    """Checks whether the ds_out for a transform is valid or not."""
    if ds_out._read_only:
        raise InvalidOutputDatasetError
    tensors = list(ds_out.tensors)
    for tensor in tensors:
        if len(ds_out[tensor]) != len(ds_out):
            raise InvalidOutputDatasetError(
                "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor (This includes empty datasets that have 0 samples per tensor)."
            )

    output_base_storage = get_base_storage(ds_out.storage)
    if isinstance(output_base_storage, MemoryProvider) and scheduler != "threaded":
        raise InvalidOutputDatasetError(
            f"Transforms with ds_out having base storage as MemoryProvider are only supported in threaded and serial mode. Current mode is {scheduler}."
        )
