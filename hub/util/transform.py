from hub.api.dataset import Dataset
from hub.util.exceptions import InvalidTransformOutputError
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from hub.util.keys import get_chunk_key, get_index_meta_key, get_tensor_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import CHUNK_MAX_SIZE


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
                raise InvalidTransformOutputError
    else:
        if not isinstance(output, dict):
            raise InvalidTransformOutputError


def get_first_chunk(index_meta: dict) -> Tuple[str, int]:
    """Finds the name and size of the first chunk in the index_meta."""
    chunk_name = ""
    chunk_size = 0

    if (
        len(index_meta["entries"]) > 0
        and len(index_meta["entries"][0]["chunk_names"]) > 0
    ):
        chunk_name = index_meta["entries"][0]["chunk_names"][0]
        chunk_size = 0

        for entry in index_meta["entries"]:
            if entry["chunk_names"] == [chunk_name]:
                chunk_size = entry["end_byte"]
            elif (
                len(entry["chunk_names"]) > 1 and entry["chunk_names"][0] == chunk_name
            ):
                chunk_size = CHUNK_MAX_SIZE
            else:
                break

    return chunk_name, chunk_size


def merge_chunks(
    chunk_min_target: int,
    tensor: str,
    storage: StorageProvider,
    current_meta: Dict,
    first_chunk_name: str = "",
    first_chunk_size: int = 0,
    last_chunk_name: str = "",
    last_chunk_size: int = 0,
):
    """Merges 2 chunks which are the last chunk of worker n and first chunk of worker n+1 into a single one if possible.
    This is done to reduce the number of suboptimal chunks generated.
    """
    if (
        first_chunk_size < chunk_min_target
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

        for i in range(len(current_meta["entries"])):
            if current_meta["entries"][i]["chunk_names"] == [first_chunk_name]:
                current_meta["entries"][i]["chunk_names"] = [last_chunk_name]
                current_meta["entries"][i]["start_byte"] += offset
                current_meta["entries"][i]["end_byte"] += offset
            else:
                break


def merge_tensor_metas(
    all_workers_tensor_meta: List[Dict[str, dict]],
    storage: StorageProvider,
    tensors: Set[str],
):
    for tensor in tensors:
        tensor_meta = TensorMeta.load(tensor, storage)

        for all_tensor_meta in all_workers_tensor_meta:
            current_meta = all_tensor_meta[tensor]
            # will be None if 0 outputs from worker
            if tensor_meta.dtype is None:
                tensor_meta.dtype = current_meta["dtype"]
                tensor_meta.max_shape = current_meta["max_shape"]
                tensor_meta.min_shape = current_meta["min_shape"]
            if current_meta["dtype"] is not None:
                assert tensor_meta.dtype == current_meta["dtype"]
                # TODO we can support this once we have ragged tensor support
                assert len(tensor_meta.max_shape) == len(current_meta["max_shape"])
                assert len(tensor_meta.min_shape) == len(current_meta["min_shape"])

                tensor_meta._update_shape_interval(tuple(current_meta["max_shape"]))
                tensor_meta._update_shape_interval(tuple(current_meta["min_shape"]))
                tensor_meta.length += current_meta["length"]


def merge_index_metas(
    all_workers_index_meta: List[Dict[str, Dict]],
    storage: StorageProvider,
    tensors: Set[str],
):
    """Merges all of the separate index metas generated across workers.
    Also merges "corner chunks" generated by each worker in case the size of those chunks is small.
    """
    for tensor in tensors:
        index_meta = IndexMeta.load(tensor, storage)
        tensor_meta = TensorMeta.load(tensor, storage)

        last_chunk_name = ""
        last_chunk_size = 0
        chunk_min_target = tensor_meta.chunk_size

        for all_index_meta in all_workers_index_meta:
            current_meta = all_index_meta[tensor]
            first_chunk_name, first_chunk_size = get_first_chunk(current_meta)
            if first_chunk_name and last_chunk_name:
                merge_chunks(
                    chunk_min_target,
                    tensor,
                    storage,
                    current_meta,
                    first_chunk_name,
                    first_chunk_size,
                    last_chunk_name,
                    last_chunk_size,
                )

            index_meta.entries.extend(current_meta["entries"])

            # if there was atleast one chunk before
            if (
                len(index_meta.entries) > 0
                and len(index_meta.entries[-1]["chunk_names"]) > 0
            ):
                last_chunk_name = index_meta.entries[-1]["chunk_names"][-1]
                last_chunk_size = index_meta.entries[-1]["end_byte"]


def pad_or_shrink_kwargs(
    pipeline_kwargs: Optional[Sequence[Dict]], pipeline: Sequence[Callable]
) -> List[Dict]:
    """Makes the number of pipeline_kwargs equal to number of functions in pipeline."""
    pipeline_kwargs = pipeline_kwargs or []
    pipeline_kwargs = list(pipeline_kwargs[0 : len(pipeline)])
    pipeline_kwargs += [{}] * (len(pipeline) - len(pipeline_kwargs))
    return pipeline_kwargs


def load_updated_meta(ds_out: Dataset):
    """Clears the dataset's cache which may contain outdated meta file and loads updated meta after transform."""
    ds_out.clear_cache()
    ds_out._load_meta()
