from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import CHUNK_MAX_SIZE, CHUNK_MIN_TARGET


def transform_sample(sample, fn_list, arg_list, tensors):
    """Calls all the functions one after the other on a single sample.
    Can return 0 or more samples.
    """
    result = sample
    for index in range(len(fn_list)):
        fn = fn_list[index]
        kwargs = arg_list[index]
        if isinstance(result, (list, tuple)) and index != 0:
            result = [fn(data, **kwargs) for data in result]
        else:
            result = fn(result, **kwargs)
        verify_transform_output(result, tensors)
    return result if isinstance(result, list) else [result]


def verify_transform_output(output, tensors):
    # TODO better exceptions
    if isinstance(output, (list, tuple)):
        for item in output:
            assert isinstance(item, dict)
            assert set(item.keys()) == tensors
    else:
        assert isinstance(output, dict)


def get_first_chunk(index_meta):
    chunk_name = None
    chunk_size = None

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


def merge_corner_chunks(index_meta, last_chunk_name, last_chunk_size, tensor, storage):
    first_chunk_name, first_chunk_size = get_first_chunk(index_meta)
    if (
        last_chunk_name
        and first_chunk_size < CHUNK_MIN_TARGET
        and first_chunk_size + last_chunk_size <= CHUNK_MAX_SIZE
    ):
        # TODO get_chunk_key
        last_chunk_content: bytes = storage[f"{tensor}/chunks/{last_chunk_name}"]
        first_chunk_content: bytes = storage[f"{tensor}/chunks/{first_chunk_name}"]
        new_chunk = bytearray(last_chunk_content) + first_chunk_content
        del storage[f"{tensor}/chunks/{first_chunk_name}"]
        storage[f"{tensor}/chunks/{last_chunk_name}"] = new_chunk

        offset = last_chunk_size

        # TODO explain why this fails for sample across multiple chunks
        for i in range(len(index_meta.entries)):
            if index_meta.entries[i]["chunk_names"] == [first_chunk_name]:
                index_meta.entries[i]["chunk_names"] = [last_chunk_name]
                index_meta.entries[i]["start_byte"] += offset
                index_meta.entries[i]["end_byte"] += offset
            else:
                break


def merge_tensor_metas(all_workers_tensor_meta, storage, tensors):
    for tensor in tensors:
        tensor_meta = None
        tensor_meta_key = None

        for all_tensor_meta in all_workers_tensor_meta:
            current_meta = all_tensor_meta[tensor]
            if tensor_meta is None:
                tensor_meta = current_meta
                tensor_meta_key = current_meta.key
            else:
                assert tensor_meta.dtype == current_meta.dtype
                tensor_meta.length += current_meta.length
                assert len(tensor_meta.max_shape) == len(current_meta.max_shape)
                assert len(tensor_meta.min_shape) == len(current_meta.min_shape)
                tensor_meta.update_shape_interval(tuple(current_meta.max_shape))
                tensor_meta.update_shape_interval(tuple(current_meta.min_shape))

        # update in meta tensor_meta.migrate
        del storage[tensor_meta_key]
        tensor_meta_dict = tensor_meta.to_dict()
        new_tensor_meta = TensorMeta.create(tensor, storage)
        new_tensor_meta.from_dict(tensor_meta_dict)


def merge_index_metas(all_workers_index_meta, storage, tensors):
    # TODO by fixing the initial loading part in both merge fn, transforms can support appending in the future
    for tensor in tensors:
        index_meta = None
        index_meta_key = None
        last_chunk_name = None
        last_chunk_size = None

        for all_index_meta in all_workers_index_meta:
            current_meta = all_index_meta[tensor]
            merge_corner_chunks(
                current_meta, last_chunk_name, last_chunk_size, tensor, storage
            )

            if index_meta is None:
                index_meta = current_meta
                index_meta_key = current_meta.key
            else:
                index_meta.entries.extend(current_meta.entries)

            # if there was atleast one chunk before
            if (
                len(index_meta.entries) > 0
                and len(index_meta.entries[-1]["chunk_names"]) > 0
            ):
                last_chunk_name = index_meta.entries[-1]["chunk_names"][-1]
                last_chunk_size = index_meta.entries[-1]["end_byte"]

        del storage[index_meta_key]
        index_meta_dict = index_meta.to_dict()
        new_index_meta = IndexMeta.create(tensor, storage)
        new_index_meta.from_dict(index_meta_dict)
