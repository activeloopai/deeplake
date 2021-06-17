from hub.util.exceptions import InvalidTransformOutputError
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple
from hub.util.keys import get_chunk_key, get_index_meta_key, get_tensor_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import CHUNK_MAX_SIZE, CHUNK_MIN_TARGET


def transform_sample(
    sample: Any,
    pipeline: Sequence[Callable],
    kwarg_list: List[dict],
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
        fn = pipeline[index]
        kwargs = kwarg_list[index]
        if isinstance(result, (list, tuple)) and index != 0:
            result = [fn(data, **kwargs) for data in result]
        else:
            result = fn(result, **kwargs)
        result = _unwrap(result)
        verify_transform_output(result)
    return result if isinstance(result, list) else [result]


def _unwrap(ls: List) -> List:
    """If there is any list then unwrap it into its elements"""
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
                raise InvalidTransformOutputError
    else:
        if not isinstance(output, dict):
            raise InvalidTransformOutputError


def get_first_chunk(index_meta: IndexMeta) -> Tuple[str, int]:
    chunk_name = ""
    chunk_size = 0

    if len(index_meta.entries) > 0 and len(index_meta.entries[0]["chunk_names"]) > 0:
        chunk_name = index_meta.entries[0]["chunk_names"][0]
        chunk_size = 0

        for entry in index_meta.entries:
            if entry["chunk_names"] == [chunk_name]:
                chunk_size = entry["end_byte"]
            elif (
                len(entry["chunk_names"]) > 1 and entry["chunk_names"][0] == chunk_name
            ):
                chunk_size = CHUNK_MAX_SIZE
            else:
                break

    return chunk_name, chunk_size


def merge_corner_chunks(
    index_meta: IndexMeta,
    tensor: str,
    storage: StorageProvider,
    last_chunk_name: str = "",
    last_chunk_size: int = 0,
):
    first_chunk_name, first_chunk_size = get_first_chunk(index_meta)
    if (
        last_chunk_name
        and first_chunk_size < CHUNK_MIN_TARGET
        and first_chunk_size + last_chunk_size <= CHUNK_MAX_SIZE
    ):
        first_chunk_key = get_chunk_key(tensor, first_chunk_name)
        last_chunk_key = get_chunk_key(tensor, last_chunk_name)

        last_chunk_content: bytes = storage[last_chunk_key]
        first_chunk_content: bytes = storage[first_chunk_key]

        new_chunk = bytearray(last_chunk_content) + first_chunk_content
        del storage[first_chunk_key]
        storage[last_chunk_key] = new_chunk

        offset = last_chunk_size

        # TODO explain why this fails for sample across multiple chunks
        for i in range(len(index_meta.entries)):
            if index_meta.entries[i]["chunk_names"] == [first_chunk_name]:
                index_meta.entries[i]["chunk_names"] = [last_chunk_name]
                index_meta.entries[i]["start_byte"] += offset
                index_meta.entries[i]["end_byte"] += offset
            else:
                break


def merge_tensor_metas(
    all_workers_tensor_meta: List[Dict[str, TensorMeta]],
    storage: StorageProvider,
    tensors: Set[str],
):
    for tensor in tensors:
        tensor_meta = TensorMeta.load(tensor, storage)

        for all_tensor_meta in all_workers_tensor_meta:
            current_meta = all_tensor_meta[tensor]
            if tensor_meta.dtype is None:
                tensor_meta = current_meta
            else:
                assert tensor_meta.dtype == current_meta.dtype
                tensor_meta.length += current_meta.length

                # TODO we can support this once we have ragged tensor support
                assert len(tensor_meta.max_shape) == len(current_meta.max_shape)
                assert len(tensor_meta.min_shape) == len(current_meta.min_shape)
                tensor_meta._update_shape_interval(tuple(current_meta.max_shape))
                tensor_meta._update_shape_interval(tuple(current_meta.min_shape))

        tensor_meta_key = get_tensor_meta_key(tensor)
        try:
            del storage[tensor_meta_key]
        except KeyError:
            pass
        
        tensor_meta.copy_to(storage)


def merge_index_metas(
    all_workers_index_meta: List[Dict[str, IndexMeta]],
    storage: StorageProvider,
    tensors: Set[str],
):
    """Merges all of the separate index metas generated across workers.
    Also merges "corner chunks" generated by each worker in case the size of those chunks is small.
    """
    for tensor in tensors:
        # if dataset exists, we can append to it. prerequisite for appending is in transfrom/transform.py (commented out assertion)
        index_meta = IndexMeta.load(tensor, storage)
        last_chunk_name = ""
        last_chunk_size = 0

        for all_index_meta in all_workers_index_meta:
            current_meta = all_index_meta[tensor]
            merge_corner_chunks(
                current_meta,
                tensor,
                storage,
                last_chunk_name,
                last_chunk_size,
            )

            if index_meta is None:
                index_meta = current_meta
            else:
                index_meta.entries.extend(current_meta.entries)

            # if there was atleast one chunk before
            if (
                len(index_meta.entries) > 0
                and len(index_meta.entries[-1]["chunk_names"]) > 0
            ):
                last_chunk_name = index_meta.entries[-1]["chunk_names"][-1]
                last_chunk_size = index_meta.entries[-1]["end_byte"]

        index_meta_key = get_index_meta_key(tensor)

        try:
            del storage[index_meta_key]
        except KeyError:
            pass

        index_meta.copy_to(storage)