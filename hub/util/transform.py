from collections import defaultdict
import math
import warnings
import hub
from typing import Any, Dict, List, Optional, Tuple
from json.decoder import JSONDecodeError
from hub.core.linked_chunk_engine import LinkedChunkEngine
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.storage import StorageProvider, MemoryProvider, LRUCache
from hub.core.chunk_engine import ChunkEngine
from hub.core.transform.transform_dataset import TransformDataset

from hub.constants import MB, TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL
from hub.util.remove_cache import get_base_storage
from hub.util.keys import get_tensor_meta_key
from hub.util.exceptions import (
    InvalidInputDataError,
    InvalidOutputDatasetError,
    InvalidTransformDataset,
    TensorMismatchError,
)

import posixpath
import time


def transform_sample(
    sample: Any,
    pipeline,
) -> TransformDataset:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.

    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline (Pipeline): The Sequence of functions to apply on the sample.

    Raises:
        InvalidTransformDataset: If number of tensors were inconsistent between all transform datasets.

    Returns:
        TransformDataset: A transform dataset containing all the samples that were generated.
    """

    result = sample
    for index in range(len(pipeline)):
        transform_fn = pipeline.functions[index]
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
    """Checks if the length of all the tensors is equal. Raises exception if not equal."""
    lengths = [len(dataset[tensor]) for tensor in dataset.tensors]
    if any(length != lengths[0] for length in lengths):
        raise InvalidTransformDataset


def is_empty_transform_dataset(dataset: TransformDataset):
    """Checks if there is any data in the TransformDataset. Returns True if empty, False otherwise."""
    return all(len(dataset[tensor]) == 0 for tensor in dataset.tensors)


def store_data_slice(transform_input: Tuple) -> Dict:
    """Takes a slice of the original data and iterates through it and stores it in the actual storage.
    The tensor_meta and chunk_id_encoder are not stored to the storage to prevent overwrites/race conditions b/w workers.
    They are instead stored in memory and returned."""
    return store_data_slice_with_pbar(None, transform_input)


def store_data_slice_with_pbar(pg_callback, transform_input: Tuple) -> Dict:
    data_slice, output_storage, inp = transform_input
    (
        group_index,
        tensors,
        visible_tensors,
        label_temp_tensors,
        actual_tensors,
        pipeline,
        version_state,
        link_creds,
        skip_ok,
    ) = inp
    all_chunk_engines = create_worker_chunk_engines(
        tensors, label_temp_tensors, output_storage, version_state, link_creds
    )

    if isinstance(data_slice, hub.Dataset):
        data_slice = add_cache_to_dataset_slice(data_slice, tensors)

    transform_data_slice_and_append(
        data_slice,
        pipeline,
        visible_tensors,
        label_temp_tensors,
        actual_tensors,
        all_chunk_engines,
        group_index,
        pg_callback,
        skip_ok,
    )

    # retrieve relevant objects from memory
    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    all_tile_encoders = {}
    all_sequence_encoders = {}
    all_chunk_sets = {}
    all_commit_diffs = {}
    all_creds_encoders = {}
    all_hash_label_maps = {}
    for tensor, chunk_engine in all_chunk_engines.items():
        chunk_engine.cache.flush()
        chunk_engine.meta_cache.flush()
        all_tensor_metas[tensor] = chunk_engine.tensor_meta
        all_chunk_id_encoders[tensor] = chunk_engine.chunk_id_encoder
        all_tile_encoders[tensor] = chunk_engine.tile_encoder
        all_sequence_encoders[tensor] = chunk_engine.sequence_encoder
        all_chunk_sets[tensor] = chunk_engine.commit_chunk_set
        all_commit_diffs[tensor] = chunk_engine.commit_diff
        all_creds_encoders[tensor] = chunk_engine.creds_encoder
        if chunk_engine._is_temp_label_tensor:
            all_hash_label_maps[tensor] = chunk_engine._hash_label_map

    return {
        "tensor_metas": all_tensor_metas,
        "chunk_id_encoders": all_chunk_id_encoders,
        "sequence_encoders": all_sequence_encoders,
        "tile_encoders": all_tile_encoders,
        "commit_chunk_sets": all_chunk_sets,
        "commit_diffs": all_commit_diffs,
        "creds_encoders": all_creds_encoders,
        "hash_label_maps": all_hash_label_maps,
    }


def _transform_sample_and_update_chunk_engines(
    sample,
    pipeline,
    tensors: List[str],
    label_temp_tensors: Dict[str, str],
    actual_tensors: Optional[List[str]],
    all_chunk_engines: Dict[str, ChunkEngine],
    group_index: str,
    skip_ok: bool = False,
):
    result = transform_sample(sample, pipeline)
    if is_empty_transform_dataset(result):
        return
    result_resolved = {
        posixpath.join(group_index, k): result[k] for k in result.tensors
    }
    result = result_resolved  # type: ignore
    result_keys = set(result.keys())

    # compare with actual tensors if there are temporary tensors
    actual_tensors = actual_tensors or tensors
    if skip_ok:
        if not result_keys.issubset(actual_tensors):
            raise TensorMismatchError(list(actual_tensors), list(result_keys), skip_ok)
    elif set(result_keys) != set(actual_tensors):
        raise TensorMismatchError(list(actual_tensors), list(result_keys), skip_ok)

    for tensor, value in result.items():
        chunk_engine = all_chunk_engines[label_temp_tensors.get(tensor) or tensor]
        callback = chunk_engine._transform_callback
        chunk_engine.extend(value.numpy_compressed(), link_callback=callback)


def transform_data_slice_and_append(
    data_slice,
    pipeline,
    tensors: List[str],
    label_temp_tensors: Dict[str, str],
    actual_tensors: Optional[List[str]],
    all_chunk_engines: Dict[str, ChunkEngine],
    group_index: str,
    pg_callback=None,
    skip_ok=False,
) -> None:
    """Transforms the data_slice with the pipeline and adds the resultant samples to chunk_engines."""

    n = len(data_slice)
    last_reported_time = time.time()
    last_reported_num_samples = 0
    for i, sample in enumerate(data_slice):
        _transform_sample_and_update_chunk_engines(
            sample,
            pipeline,
            tensors,
            label_temp_tensors,
            actual_tensors,
            all_chunk_engines,
            group_index,
            skip_ok,
        )
        if pg_callback is not None:
            curr_time = time.time()
            if (
                curr_time - last_reported_time > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL
                or i == n - 1
            ):
                num_samples = i + 1
                pg_callback(num_samples - last_reported_num_samples)
                last_reported_num_samples = num_samples
                last_reported_time = curr_time


def create_worker_chunk_engines(
    tensors: List[str],
    label_temp_tensors: Dict[str, str],
    output_storage: StorageProvider,
    version_state,
    link_creds,
) -> Dict[str, ChunkEngine]:
    """Creates chunk engines corresponding to each storage for all tensors.
    These are created separately for each worker for parallel uploads.
    """
    all_chunk_engines: Dict[str, ChunkEngine] = {}
    num_tries = 1000
    for tensor in tensors:
        for i in range(num_tries):
            try:
                # TODO: replace this with simply a MemoryProvider once we get rid of cachable
                memory_cache = LRUCache(MemoryProvider(), MemoryProvider(), 64 * MB)
                memory_cache.autoflush = False
                storage_cache = LRUCache(MemoryProvider(), output_storage, 64 * MB)
                storage_cache.autoflush = False

                # this chunk engine is used to retrieve actual tensor meta and chunk_size
                storage_chunk_engine = ChunkEngine(tensor, storage_cache, version_state)
                existing_meta = storage_chunk_engine.tensor_meta

                chunk_size = storage_chunk_engine.max_chunk_size
                tiling_threshold = storage_chunk_engine.tiling_threshold
                new_tensor_meta = TensorMeta(
                    htype=existing_meta.htype,
                    dtype=existing_meta.dtype,
                    sample_compression=existing_meta.sample_compression,
                    chunk_compression=existing_meta.chunk_compression,
                    max_chunk_size=chunk_size,
                    tiling_threshold=tiling_threshold,
                    links=existing_meta.links,
                    is_sequence=existing_meta.is_sequence,
                    is_link=existing_meta.is_link,
                    hidden=existing_meta.hidden,
                    verify=existing_meta.verify,
                )
                meta_key = get_tensor_meta_key(tensor, version_state["commit_id"])
                memory_cache[meta_key] = new_tensor_meta  # type: ignore
                storage_cache.clear_cache()
                if existing_meta.is_link:
                    storage_chunk_engine = LinkedChunkEngine(
                        tensor,
                        storage_cache,
                        version_state,
                        link_creds,
                        memory_cache,
                    )
                else:
                    storage_chunk_engine = ChunkEngine(
                        tensor, storage_cache, version_state, memory_cache
                    )
                storage_chunk_engine._all_chunk_engines = all_chunk_engines
                if tensor in label_temp_tensors.values():
                    storage_chunk_engine._is_temp_label_tensor = True
                all_chunk_engines[tensor] = storage_chunk_engine
                break
            except (JSONDecodeError, KeyError):
                if i == num_tries - 1:
                    raise
    return all_chunk_engines


def add_cache_to_dataset_slice(
    dataset_slice: hub.Dataset,
    tensors: List[str],
) -> hub.Dataset:
    base_storage = get_base_storage(dataset_slice.storage)
    # 64 to account for potentially big encoder corresponding to each tensor
    # TODO: adjust this size once we get rid of cachable
    cache_size = 64 * len(tensors) * MB
    cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
    commit_id = dataset_slice.pending_commit_id
    # don't pass version state to constructor as otherwise all workers will share it, checkout to commit_id instead
    dataset_slice = hub.core.dataset.dataset_factory(
        path=dataset_slice.path,
        storage=cached_store,
        index=dataset_slice.index,
        group_index=dataset_slice.group_index,
        read_only=dataset_slice.read_only,
        token=dataset_slice.token,
        verbose=False,
        link_creds=dataset_slice.link_creds,
        pad_tensors=dataset_slice._pad_tensors,
    )
    dataset_slice.checkout(commit_id)
    return dataset_slice


def check_transform_data_in(data_in, scheduler: str) -> None:
    """Checks whether the data_in for a transform is valid or not."""
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")
    if isinstance(data_in, hub.Dataset):
        input_base_storage = get_base_storage(data_in.storage)
        if isinstance(input_base_storage, MemoryProvider) and scheduler not in [
            "serial",
            "threaded",
        ]:
            raise InvalidInputDataError(
                f"Transforms with data_in as a Dataset having base storage as MemoryProvider are only supported in threaded and serial mode. Current mode is {scheduler}."
            )


def check_transform_ds_out(
    ds_out: hub.Dataset, scheduler: str, check_lengths: bool
) -> None:
    """Checks whether the ds_out for a transform is valid or not."""
    if ds_out._read_only:
        raise InvalidOutputDatasetError
    tensors = list(ds_out.tensors)

    if check_lengths:
        for tensor in tensors:
            if len(ds_out[tensor]) != len(ds_out):
                raise InvalidOutputDatasetError(
                    "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor (This includes empty datasets that have 0 samples per tensor)."
                )

    output_base_storage = get_base_storage(ds_out.storage)
    if isinstance(output_base_storage, MemoryProvider) and scheduler not in [
        "serial",
        "threaded",
    ]:
        raise InvalidOutputDatasetError(
            f"Transforms with ds_out having base storage as MemoryProvider are only supported in threaded and serial mode. Current mode is {scheduler}."
        )


def get_pbar_description(compute_functions: List):
    """Returns the description string for a hub.compute evaluation progress bar. Incoming list should be a list of `ComputeFunction`s."""

    num_funcs = len(compute_functions)
    if num_funcs == 0:
        return "Evaluating"

    func_names: List[str] = [f.name for f in compute_functions]
    if num_funcs == 1:
        return f"Evaluating {func_names[0]}"

    names_desc = ", ".join(func_names)
    return f"Evaluating [{names_desc}]"


def create_slices(data_in, num_workers):
    size = math.ceil(len(data_in) / num_workers)
    return [data_in[i * size : (i + 1) * size] for i in range(num_workers)]


def get_old_chunk_paths(target_ds, generated_tensors, overwrite):
    old_chunk_paths = []
    if overwrite:
        for key in generated_tensors:
            tensor = target_ds[key]
            old_chunk_paths.extend(tensor.chunk_engine.list_all_chunks_path())

    return old_chunk_paths


def delete_overwritten_chunks(old_chunk_paths, storage, overwrite):
    if not overwrite:
        return

    storage.delete_multiple(old_chunk_paths)


def get_lengths_generated(all_tensor_metas, tensors):
    all_num_samples = []
    all_tensors_generated_length = {tensor: 0 for tensor in tensors}
    for tensor_meta_dict in all_tensor_metas:
        num_samples_dict = {}
        for tensor, meta in tensor_meta_dict.items():
            all_tensors_generated_length[tensor] += meta.length
            num_samples_dict[tensor] = meta.length
        all_num_samples.append(num_samples_dict)
    return all_num_samples, all_tensors_generated_length


def check_lengths(all_tensors_generated_length, skip_ok):
    if not skip_ok:
        return

    first_length = None
    for length in all_tensors_generated_length.values():
        if length == 0:
            continue
        if first_length is None:
            first_length = length
        elif length not in [0, first_length]:
            warnings.warn(
                "Length of all tensors generated is not the same, this may lead to unexpected behavior."
            )
            break


def sanitize_workers_scheduler(num_workers, scheduler):
    if num_workers <= 0:
        scheduler = "serial"
    num_workers = max(num_workers, 1)
    return num_workers, scheduler


def process_transform_result(result: List[Dict]):
    if not result:
        return result
    final = defaultdict(list)
    keys = list(result[0].keys())
    for item in result:
        for key in keys:
            final[key].append(item[key])
    return final
