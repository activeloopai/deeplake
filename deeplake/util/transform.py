from collections import defaultdict
import math
import warnings
import deeplake
from typing import Any, Dict, List, Optional, Tuple
from json.decoder import JSONDecodeError
from deeplake.core.linked_chunk_engine import LinkedChunkEngine
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.storage import StorageProvider, MemoryProvider, LRUCache
from deeplake.core.chunk_engine import ChunkEngine
from deeplake.core.transform.transform_dataset import TransformDataset
from deeplake.core.index import Index
from deeplake.core.tensor import Tensor

from deeplake.constants import (
    MB,
    TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL,
    TRANSFORM_RECHUNK_AVG_SIZE_BOUND,
    TRANSFORM_CHUNK_CACHE_SIZE,
)
from deeplake.util.dataset import try_flushing
from deeplake.util.path import relpath
from deeplake.util.remove_cache import (
    get_base_storage,
    get_dataset_with_zero_size_cache,
)
from deeplake.util.keys import get_tensor_meta_key
from deeplake.util.version_control import auto_checkout, load_meta
from deeplake.util.exceptions import (
    AllSamplesSkippedError,
    InvalidInputDataError,
    InvalidOutputDatasetError,
    InvalidTransformDataset,
    TensorMismatchError,
    TensorDoesNotExistError,
    TransformError,
    SampleAppendError,
)

import traceback
import posixpath
import time

import numpy as np


def transform_sample(
    sample: Any,
    pipeline,
    tensors,
) -> TransformDataset:
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.

    Args:
        sample: The sample on which the pipeline of functions is to be applied.
        pipeline (Pipeline): The Sequence of functions to apply on the sample.
        tensors: List of tensors in output.

    Returns:
        TransformDataset: A transform dataset containing all the samples that were generated.
    """
    out = sample
    for index in range(len(pipeline)):
        transform_fn = pipeline.functions[index]
        fn, args, kwargs = transform_fn.func, transform_fn.args, transform_fn.kwargs

        if isinstance(out, TransformDataset):
            result = TransformDataset(tensors)
            for item in out:
                fn(item, result, *args, **kwargs)
                validate_transform_dataset(result)
            out = result
        else:
            result = TransformDataset(tensors)
            fn(out, result, *args, **kwargs)
            validate_transform_dataset(result)
            out = result
    return out


def validate_transform_dataset(dataset: TransformDataset):
    """Checks if the length of all the tensors is equal. Raises exception if not equal."""
    data = dataset.data
    lengths = [
        len(data[tensor])
        for tensor in data
        if (not data[tensor].is_group and len(data[tensor]) > 0)
    ]
    if any(length != lengths[0] for length in lengths):
        raise InvalidTransformDataset(
            "The number of samples added to each tensor in transform should be the same."
        )


def is_empty_transform_dataset(dataset: TransformDataset):
    """Checks if there is any data in the TransformDataset. Returns True if empty, False otherwise."""
    return all(len(dataset[tensor]) == 0 for tensor in dataset.tensors)


def store_data_slice(transform_input: Tuple) -> Dict:
    """Takes a slice of the original data and iterates through it and stores it in the actual storage.
    The tensor_meta and chunk_id_encoder are not stored to the storage to prevent overwrites/race conditions b/w workers.
    They are instead stored in memory and returned."""
    return store_data_slice_with_pbar(None, transform_input)


def _normalize_pg(pg_callback, num_tensors):
    def inner(num_samples):
        return pg_callback(num_samples / num_tensors)

    return inner


def _extend_data_slice(
    data_slice, offset, transform_dataset, transform_fn, pg_callback
):
    extend_fn, args, kwargs = (
        transform_fn.func,
        transform_fn.args,
        transform_fn.kwargs,
    )
    if pg_callback is not None:
        pg_callback = _normalize_pg(pg_callback, len(transform_dataset.tensors))
    transform_dataset.set_pg_callback(pg_callback)
    extend_fn(data_slice, transform_dataset, *args, **kwargs)
    transform_dataset.flush()


def _check_pipeline(out, tensors, skip_ok):
    data = out.data
    result_keys = set(k for k in data if not data[k].is_group and len(data[k]) > 0)

    if skip_ok:
        if not result_keys.issubset(tensors):
            raise TensorMismatchError(list(tensors), list(result_keys), skip_ok)
    elif set(result_keys) != set(tensors):
        raise TensorMismatchError(list(tensors), list(result_keys), skip_ok)


def write_sample_to_transform_dataset(out, transform_dataset):
    if not is_empty_transform_dataset(out):
        for tensor in out.tensors:
            out_tensor = out[tensor]
            transform_tensor = transform_dataset[tensor]
            if transform_tensor.numpy_only and out_tensor.numpy_only:
                for item in out_tensor.items:
                    transform_tensor.extend(item)
            else:
                out_tensor.non_numpy_only()
                transform_tensor.extend(out_tensor.items)
            out_tensor.items.clear()


def _handle_transform_error(
    data_slice,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    end_input_idx,
    ignore_errors,
):
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None

    start_input_idx = transform_dataset.start_input_idx
    skipped_samples = 0
    for i in range(start_input_idx, end_input_idx + 1):
        sample = (
            data_slice[i : i + 1]
            if pd and isinstance(data_slice, pd.DataFrame)
            else data_slice[i]
        )
        try:
            out = transform_sample(sample, pipeline, tensors)

            write_sample_to_transform_dataset(out, transform_dataset)

            transform_dataset.flush()
        except Exception as e:
            if ignore_errors:
                skipped_samples += 1
                continue
            raise TransformError(
                offset + i, sample, suggest=isinstance(e, SampleAppendError)
            ) from e
    return skipped_samples


def _transform_and_append_data_slice(
    data_slice,
    offset,
    transform_dataset,
    pipeline,
    tensors,
    skip_ok,
    pg_callback,
    ignore_errors,
):
    """Appends a data slice. Returns ``True`` if any samples were appended and ``False`` otherwise."""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None

    n = len(data_slice)
    skipped_samples = 0
    skipped_samples_in_current_batch = 0

    pipeline_checked = False

    last_pg_update_time = time.time()
    progress = 0

    for i, sample in enumerate(
        (data_slice[i : i + 1] for i in range(n))
        if pd and isinstance(data_slice, pd.DataFrame)
        else data_slice
    ):
        try:
            transform_dataset.set_start_input_idx(i)

            try:
                out = transform_sample(sample, pipeline, tensors)

                if not pipeline_checked:
                    _check_pipeline(out, tensors, skip_ok)
                    pipeline_checked = True

                write_sample_to_transform_dataset(out, transform_dataset)

            except Exception as e:
                if ignore_errors:
                    skipped_samples += 1
                    skipped_samples_in_current_batch += 1
                else:
                    raise TransformError(
                        offset + i, sample, suggest=isinstance(e, SampleAppendError)
                    ) from e

            finally:
                if i == n - 1:
                    transform_dataset.flush()
                else:
                    transform_dataset.check_flush()

                # dataset flushed, reset skipped sample count
                if transform_dataset.start_input_idx is None:
                    skipped_samples_in_current_batch = 0

                if pg_callback is not None:
                    progress += 1
                    if (
                        time.time() - last_pg_update_time
                        > TRANSFORM_PROGRESSBAR_UPDATE_INTERVAL
                        or i == n - 1
                    ):
                        pg_callback(progress)
                        progress = 0
                        last_pg_update_time = time.time()

        # failure at chunk_engine
        # retry one sample at a time
        except Exception as e:
            # TransformError pass through
            if isinstance(e, TransformError):
                raise e

            # reset skipped sample count to avoid double counting
            skipped_samples -= skipped_samples_in_current_batch
            skipped_samples_in_current_batch = 0

            skipped_samples += _handle_transform_error(
                data_slice,
                offset,
                transform_dataset,
                pipeline,
                tensors,
                i,
                ignore_errors,
            )
            continue

    return {
        "samples_skipped": skipped_samples,
        "all_samples_skipped": skipped_samples == n,
    }


def _retrieve_memory_objects(all_chunk_engines):
    all_tensor_metas = {}
    all_chunk_id_encoders = {}
    all_tile_encoders = {}
    all_sequence_encoders = {}
    all_pad_encoders = {}
    all_chunk_maps = {}
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
        all_pad_encoders[tensor] = chunk_engine.pad_encoder
        all_chunk_maps[tensor] = chunk_engine.commit_chunk_map
        all_commit_diffs[tensor] = chunk_engine.commit_diff
        all_creds_encoders[tensor] = chunk_engine.creds_encoder
        if chunk_engine._is_temp_label_tensor:
            all_hash_label_maps[tensor] = chunk_engine._hash_label_map

    return {
        "tensor_metas": all_tensor_metas,
        "chunk_id_encoders": all_chunk_id_encoders,
        "sequence_encoders": all_sequence_encoders,
        "pad_encoders": all_pad_encoders,
        "tile_encoders": all_tile_encoders,
        "commit_chunk_maps": all_chunk_maps,
        "commit_diffs": all_commit_diffs,
        "creds_encoders": all_creds_encoders,
        "hash_label_maps": all_hash_label_maps,
    }


def store_data_slice_with_pbar(pg_callback, transform_input: Tuple) -> Dict:
    data_slice, offset, output_storage, inp = transform_input
    (
        group_index,
        tensors,
        visible_tensors,
        label_temp_tensors,
        pipeline,
        version_state,
        link_creds,
        skip_ok,
        extend_only,
        cache_size,
        ignore_errors,
    ) = inp
    all_chunk_engines = create_worker_chunk_engines(
        tensors, label_temp_tensors, output_storage, version_state, link_creds
    )

    if isinstance(data_slice, deeplake.Dataset):
        data_slice = add_cache_to_dataset_slice(data_slice, tensors)

    rel_tensors = [posixpath.relpath(tensor, group_index) for tensor in visible_tensors]

    transform_dataset = TransformDataset(
        rel_tensors,
        all_chunk_engines,
        group_index,
        label_temp_tensors,
        cache_size=cache_size,
    )

    ret = {
        "all_samples_skipped": False,
        "samples_skipped": 0,
    }
    err = None
    try:
        if extend_only:
            _extend_data_slice(
                data_slice,
                offset,
                transform_dataset,
                pipeline.functions[0],
                pg_callback,
            )
        else:
            ret = _transform_and_append_data_slice(
                data_slice,
                offset,
                transform_dataset,
                pipeline,
                rel_tensors,
                skip_ok,
                pg_callback,
                ignore_errors,
            )
    except Exception as e:
        try:
            transform_dataset.flush()
        except Exception:
            pass
        err = e
    finally:
        # retrieve relevant objects from memory
        meta = _retrieve_memory_objects(all_chunk_engines)
        meta.update(ret)

        err_dict: Optional[Dict[str, Any]] = None
        if err:
            err_dict = {}
            err_dict["raise"] = err
            cause = err.__cause__
            if cause:
                cause_traceback = "".join(
                    traceback.format_exception(cause.__class__, cause, cause.__traceback__)  # type: ignore
                )
                err_dict["traceback"] = cause_traceback
        meta["error"] = err_dict
        return meta


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
    storage_cache = LRUCache(
        MemoryProvider(), output_storage, TRANSFORM_CHUNK_CACHE_SIZE
    )
    storage_cache.autoflush = False
    # TODO: replace this with simply a MemoryProvider once we get rid of cachable
    memory_cache = LRUCache(
        MemoryProvider(),
        MemoryProvider(),
        64 * MB,
    )
    memory_cache.autoflush = False
    for tensor in tensors:
        for i in range(num_tries):
            try:
                # this chunk engine is used to retrieve actual tensor meta and chunk_size
                storage_chunk_engine = ChunkEngine(tensor, storage_cache, version_state)
                existing_meta = storage_chunk_engine.tensor_meta

                chunk_size = storage_chunk_engine.max_chunk_size
                tiling_threshold = storage_chunk_engine.tiling_threshold
                new_tensor_meta = TensorMeta(
                    htype=existing_meta.htype,
                    dtype=(
                        np.dtype(existing_meta.typestr)
                        if existing_meta.typestr
                        else existing_meta.dtype
                    ),
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
                new_tensor_meta.max_shape = existing_meta.max_shape.copy()
                new_tensor_meta.min_shape = existing_meta.min_shape.copy()
                new_tensor_meta.name = existing_meta.name
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
    dataset_slice: deeplake.Dataset,
    tensors: List[str],
) -> deeplake.Dataset:
    base_storage = get_base_storage(dataset_slice.storage)
    # 64 to account for potentially big encoder corresponding to each tensor
    # TODO: adjust this size once we get rid of cachable
    cache_size = 64 * len(tensors) * MB
    cached_store = LRUCache(MemoryProvider(), base_storage, cache_size)
    commit_id = dataset_slice.pending_commit_id
    # don't pass version state to constructor as otherwise all workers will share it, checkout to commit_id instead
    index = Index.from_json(
        dataset_slice.index.to_json()
    )  # we don't allow checkouts for views
    dataset_slice = deeplake.core.dataset.dataset_factory(
        path=dataset_slice.path,
        storage=cached_store,
        group_index=dataset_slice.group_index,
        read_only=dataset_slice.read_only,
        token=dataset_slice.token,
        verbose=False,
        link_creds=dataset_slice.link_creds,
        pad_tensors=dataset_slice._pad_tensors,
        enabled_tensors=dataset_slice.enabled_tensors,
    )
    dataset_slice.checkout(commit_id)
    dataset_slice.index = index
    return dataset_slice


def check_transform_data_in(data_in, scheduler: str) -> None:
    """Checks whether the data_in for a transform is valid or not."""
    if not hasattr(data_in, "__getitem__"):
        raise InvalidInputDataError("__getitem__")
    if not hasattr(data_in, "__len__"):
        raise InvalidInputDataError("__len__")
    if isinstance(data_in, deeplake.Dataset):
        input_base_storage = get_base_storage(data_in.storage)
        if isinstance(input_base_storage, MemoryProvider) and scheduler not in [
            "serial",
            "threaded",
        ]:
            raise InvalidInputDataError(
                f"Transforms with data_in as a Dataset having base storage as MemoryProvider are only supported in threaded and serial mode. Current mode is {scheduler}."
            )


def check_transform_ds_out(
    ds_out: deeplake.Dataset,
    scheduler: str,
    check_lengths: bool,
    read_only_ok: bool = False,
) -> None:
    """Checks whether the ds_out for a transform is valid or not."""
    if ds_out._read_only and not read_only_ok:
        raise InvalidOutputDatasetError
    tensors = list(ds_out.tensors)

    if check_lengths:
        for tensor in tensors:
            if len(ds_out[tensor]) != len(ds_out):
                raise InvalidOutputDatasetError(
                    "One or more tensors of the ds_out have different lengths. Transform only supports ds_out having same number of samples for each tensor by default."
                    " Set `check_lengths=False` in your `.eval(...)` call to disable this check."
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
    """Returns the description string for a :meth:`deeplake.compute` evaluation progress bar. Incoming list should be a list of `ComputeFunction`s."""

    num_funcs = len(compute_functions)
    if num_funcs == 0:
        return "Evaluating"

    func_names: List[str] = [f.name for f in compute_functions]
    if num_funcs == 1:
        return f"Evaluating {func_names[0]}"

    names_desc = ", ".join(func_names)
    return f"Evaluating [{names_desc}]"


def len_data_in(data_in):
    if isinstance(data_in, deeplake.Dataset):
        return data_in.max_len
    else:
        return len(data_in)


def transform_summary(data_in, result):
    samples_skipped = sum(result["samples_skipped"])
    successful = len_data_in(data_in) - samples_skipped
    successful_percent = round((successful / len_data_in(data_in)) * 100, 2)
    skipped_percent = round(100 - successful_percent, 2)

    print(
        "No. of samples successfully processed:", successful, f"({successful_percent}%)"
    )
    print("No. of samples skipped:", samples_skipped, f"({skipped_percent}%)")


def create_slices(data_in, num_workers):
    size = math.ceil(len_data_in(data_in) / num_workers)

    if isinstance(data_in, Tensor):
        ret = [
            Tensor(data_in.key, data_in.dataset)[i * size : (i + 1) * size]
            for i in range(num_workers)
        ]
    else:
        ret = [data_in[i * size : (i + 1) * size] for i in range(num_workers)]

    if isinstance(data_in, deeplake.Dataset):
        for ds in ret:
            ds.version_state["full_tensors"] = {}
            _tensors = ds.version_state["full_tensors"]
            for tensor_key in data_in.version_state["tensor_names"].values():
                _tensors[tensor_key] = Tensor(tensor_key, ds)

    offsets = list(range(0, len_data_in(data_in), size))
    return ret, offsets


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
    if skip_ok:
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
    if all(res.get("all_samples_skipped") for res in result):
        raise AllSamplesSkippedError
    if not result:
        return result
    final = defaultdict(list)
    keys = list(result[0].keys())
    for item in result:
        for key in keys:
            final[key].append(item[key])
    return final


def rechunk_if_necessary(ds):
    with ds:
        for tensor in ds.tensors:
            try:
                tensor = ds[tensor]
            # temp tensors
            except TensorDoesNotExistError:
                continue
            if not tensor.meta.sample_compression and not tensor.meta.chunk_compression:
                engine = tensor.chunk_engine
                num_chunks = engine.num_chunks
                if num_chunks > 1:
                    max_shape = tensor.meta.max_shape
                    if len(max_shape) > 0:
                        avg_chunk_size = engine.get_avg_chunk_size()
                        if (
                            avg_chunk_size is not None
                            and avg_chunk_size
                            < TRANSFORM_RECHUNK_AVG_SIZE_BOUND * engine.min_chunk_size
                        ):
                            enc = tensor.chunk_engine.chunk_id_encoder
                            row = 0
                            while row < len(enc._encoded) - 1:
                                encoded = enc._encoded
                                chunk_id = encoded[row, 0]
                                chunk = engine.get_chunk_from_chunk_id(chunk_id)
                                engine._check_rechunk(chunk, row)
                                # np.delete will replace enc._encoded with new array
                                # so this check works
                                rechunked = len(encoded) != len(enc._encoded)
                                if not rechunked:
                                    row += 1


def close_states(compute_provider, pbar, pqueue):
    compute_provider.close()
    if pbar and hasattr(pbar, "close"):
        pbar.close()
    if pqueue and hasattr(pqueue, "close"):
        pqueue.close()


def reload_and_rechunk(
    overwrite,
    original_data_in,
    target_ds,
    initial_autoflush,
    pad_data_in,
    initial_padding_state,
    kwargs,
    completed=True,
):
    if overwrite:
        original_data_in.storage.clear_cache_without_flush()
        load_meta(original_data_in)
        if pad_data_in and not initial_padding_state:
            original_data_in._disable_padding()
        if completed and not kwargs.get("disable_rechunk"):
            rechunk_if_necessary(original_data_in)
    else:
        load_meta(target_ds)
        if completed:
            target_ds.storage.autoflush = initial_autoflush
            if not kwargs.get("disable_rechunk"):
                rechunk_if_necessary(target_ds)


def check_checkpoint_interval(
    data_in, checkpoint_interval, num_workers, overwrite, verbose
):
    if num_workers > 0 and checkpoint_interval % num_workers != 0:
        raise ValueError(
            "checkpoint_interval should be a multiple of num_workers if num_workers > 0"
        )
    if checkpoint_interval > len_data_in(data_in):
        raise ValueError(
            "checkpoint_interval should be less than or equal to the length of data_in"
        )
    if checkpoint_interval < len_data_in(data_in) / 10 and verbose:
        warnings.warn(
            "checkpoint_interval is less than 10% of the length of data_in, this can lead to too many commits, consider increasing checkpoint_interval."
        )
    if overwrite:
        raise ValueError(
            "checkpoint_interval > 0 and ds_out is None. Cannot checkpoint during inplace transform."
        )


def prepare_data_in(data_in, pad_data_in, overwrite):
    initial_padding_state = None
    original_data_in = data_in
    if isinstance(data_in, deeplake.Dataset):
        try_flushing(data_in)
        if overwrite:
            auto_checkout(data_in)
        original_data_in = data_in
        data_in = get_dataset_with_zero_size_cache(data_in)
        if pad_data_in:
            initial_padding_state = data_in._pad_tensors
            data_in._enable_padding()
    return data_in, original_data_in, initial_padding_state
