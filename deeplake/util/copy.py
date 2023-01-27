from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap
from deeplake.util.keys import (
    get_tensor_meta_key,
    get_tensor_info_key,
    get_tensor_tile_encoder_key,
    get_creds_encoder_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_chunk_id_encoder_key,
    get_sequence_encoder_key,
    get_dataset_meta_key,
)
import numpy as np


def _get_meta_files_for_tensor(tensor_name, commit_id):
    fns = [
        get_tensor_meta_key,
        get_tensor_info_key,
        get_chunk_id_encoder_key,
        get_tensor_tile_encoder_key,
        get_creds_encoder_key,
        get_sequence_encoder_key,
    ]
    return [fn(tensor_name, commit_id) for fn in fns]


def _get_chunks_for_tensor(src_tensor, dest_commit_id, dest_key):
    eng = src_tensor.chunk_engine
    enc = eng.chunk_id_encoder

    chunkids = enc._encoded[:, 0]
    ret = []
    for cid in chunkids:
        cname = enc.name_from_id(cid)
        commit, key = eng.get_chunk_commit(cname)
        same_commit = commit == dest_commit_id
        same_key = key == dest_key
        if same_commit and same_key:
            ret.append((cname,))
        elif same_key:
            ret.append((cname, commit))
        else:
            ret.append((cname, commit, key))
    return ret


def _copy_objects(key_pairs, src_storage, dest_storage):
    for src_key, dest_key in zip(*key_pairs):
        try:
            dest_storage[dest_key] = src_storage[src_key]
        except KeyError as ke:
            pass


def copy_tensors(
    src_ds,
    dest_ds,
    src_tensor_names,
    dest_tensor_names=None,
):
    if not src_tensor_names:
        return
    if not src_ds.read_only:
        src_ds.flush()
    dest_ds.flush()
    src_path = src_ds.path
    dest_path = dest_ds.path
    src_tensor_names = list(src_tensor_names)
    src_commit_id = src_ds.pending_commit_id
    dest_commit_id = dest_ds.pending_commit_id
    dest_ds_meta = dest_ds.meta
    hidden_tensors = []
    src_tensor_names_get = {
        v: k for k, v in src_ds.meta.tensor_names.items()
    }.__getitem__
    for i in range(len(src_tensor_names)):
        src_tensor = src_ds[src_tensor_names[i]]
        hidden_tensors += map(src_tensor_names_get, src_tensor.meta.links)
    src_tensor_names += hidden_tensors
    if dest_tensor_names is None:
        dest_tensor_names = src_tensor_names
    else:
        assert len(src_tensor_names) == len(dest_tensor_names)
    src_keys = []
    dest_keys = []
    src_storage = src_ds.base_storage
    dest_storage = dest_ds.base_storage
    updated_dest_keys = []
    for src_tensor_name, dest_tensor_name in zip(src_tensor_names, dest_tensor_names):
        assert dest_tensor_name not in dest_ds._tensors(include_hidden=True)
        src_tensor = src_ds[src_tensor_name]
        src_key = src_tensor.key
        src_tensor_key = src_tensor.key
        chunks = _get_chunks_for_tensor(src_tensor, dest_commit_id, dest_tensor_name)

        dest_chunk_map_key = get_tensor_commit_chunk_map_key(
            dest_tensor_name, dest_commit_id
        )
        dest_chunk_map = CommitChunkMap()
        for chunk in chunks:
            dest_chunk_map.add(*chunk)
        dest_storage[dest_chunk_map_key] = dest_chunk_map.tobytes()
        src_keys += _get_meta_files_for_tensor(src_tensor_key, src_commit_id)
        dest_keys += _get_meta_files_for_tensor(dest_tensor_name, dest_commit_id)
        dest_commit_diff = CommitDiff(0, True)
        dest_commit_diff.add_data(src_tensor.meta.length)
        dest_commit_diff_key = get_tensor_commit_diff_key(
            dest_tensor_name, dest_commit_id
        )
        dest_storage[dest_commit_diff_key] = dest_commit_diff.tobytes()
        updated_dest_keys = [dest_commit_diff_key]
        updated_dest_keys.append(dest_chunk_map_key)
    _copy_objects((src_keys, dest_keys), src_storage, dest_storage)
    dest_ds_meta.tensors += dest_tensor_names
    dest_ds_meta.tensor_names.update({k: k for k in dest_tensor_names})
    dest_ds_meta.hidden_tensors += hidden_tensors
    dest_storage[get_dataset_meta_key(dest_commit_id)] = dest_ds_meta.tobytes()
    dest_ds.storage.clear_cache_without_flush()
    dest_ds._populate_meta()


def _group_ranges(x):
    ret = []
    s = x[0]
    e = s + 1
    for i in range(1, len(x)):
        xi = x[i]
        if xi == e:
            e += 1
        else:
            ret.append((s, e))
            s = xi
            e = s + 1
    ret.append((s, e))
    return ret


def _merge_chunk_id_encodings(enc1, enc2, start, end):
    n1 = len(enc1)
    if not n1:
        return enc2
    n2 = len(enc2)
    if not n2:
        return enc1
    if start == 0:
        old_offset = 0
    else:
        old_offset = enc2[start - 1, 1:2] + 1
    new_offset = enc1[-1, 1:2] + 1
    ret = np.concatenate([enc1, enc2[start:end]], axis=0)
    ret[n1:, 1] += new_offset - old_offset
    return ret


def _get_required_chunks_for_range(tensor, start, end):
    eng = tensor.chunk_engine
    enc = eng.chunk_id_encoder
    arr = enc._encoded
    start_row = enc.translate_index(start)
    end_row = enc.translate_index(end - 1)
    orig_end_row = end_row
    end_chunk_id = arr[end_row, 0]
    nrows = len(arr)
    nxt = end_row + 1
    while nxt < nrows and arr[nxt, 0] == end_chunk_id:
        end_row = nxt
        nxt += 1
    num_required_chunks = end_row + 1 - start_row
    start_chunk_aligned = False
    end_chunk_aligned = False
    if start_row == 0:
        if start == 0:
            start_chunk_aligned = True
    else:
        prev_row = start_row - 1
        if start == arr[prev_row, 1] + 1:
            start_chunk_aligned = True
    if arr[end_row, 1] == end - 1:
        end_chunk_aligned = True
    if num_required_chunks == 1:
        if not (start_chunk_aligned and end_chunk_aligned):
            return None, (start, end), None
        else:
            return (start_row, start_row + 1), None, None
    elif num_required_chunks == 2:
        if not start_chunk_aligned and not end_chunk_aligned:
            return None, (start, end), None
        if start_chunk_aligned:
            return (start_row, start_row + 1), None, (arr[start_row, 1] + 1, end)
        else:
            return (end_row, end_row + 1), (start, arr[start_row, 1] + 1), None
    elif start_chunk_aligned and not end_chunk_aligned:
        return (start_row, end_row), None, (arr[end_row - 1, 1] + 1, end)
    elif end_chunk_aligned and not start_chunk_aligned:
        return (start_row + 1, end_row + 1), (start, arr[start_row, 1] + 1), None
    elif not start_chunk_aligned and not end_chunk_aligned:
        return (
            (start_row + 1, end_row),
            (start, arr[start_row, 1] + 1),
            (arr[end_row - 1, 1] + 1, end),
        )
    else:
        return (start_row, end_row + 1), None, None


def copy_tensor_slice(
    src_ds, dest_ds, src_tensor_name, dest_tensor_name, indices, _copy_links_only=False, _flush=True
):
    if not indices:
        return
    if _flush:
        dest_ds.flush()
    src_tensor = src_ds[src_tensor_name]
    dest_tensor = dest_ds[dest_tensor_name]
    if not _copy_links_only:
        src_key = src_tensor.key
        dest_key = dest_tensor.key
        src_commit = src_ds.pending_commit_id
        dest_commit = dest_ds.pending_commit_id
        src_eng = src_tensor.chunk_engine
        src_enc = src_eng.chunk_id_encoder
        dest_eng = dest_tensor.chunk_engine
        dest_enc = dest_eng.chunk_id_encoder
        src_enc_arr = src_enc._encoded
        ranges = _group_ranges(indices)
        dest_storage = dest_ds.base_storage
        dest_meta_key = get_tensor_meta_key(dest_key, dest_commit)
        src_meta = src_tensor.meta
        dest_meta = dest_tensor.meta
        dest_length = dest_meta.length + len(indices)
        chunk_map_key = get_tensor_commit_chunk_map_key(dest_key, dest_commit)
        chunk_map = dest_eng.commit_chunk_map
        links = dest_tensor.meta.links
        dest_tensor.meta.links = {}
        try:
            for start, end in ranges:
                (
                    chunks_to_copy,
                    left_edge_samples,
                    right_edge_samples,
                ) = _get_required_chunks_for_range(src_tensor, start, end)
                if left_edge_samples:
                    s, e = left_edge_samples
                    dest_tensor.extend(src_tensor[s:e])
                if chunks_to_copy:
                    s, e = chunks_to_copy
                    chunk_ids = src_enc_arr[s:e, 0]
                    chunk_names = list(map(src_enc.name_from_id, chunk_ids))
                    commit_key_pairs = list(map(src_eng.get_chunk_commit, chunk_names))
                    for chunk_name, (commit, key) in zip(chunk_names, commit_key_pairs):
                        if commit == dest_commit:
                            commit = None
                        elif key == dest_key:
                            key = None
                        chunk_map.add(chunk_name, commit, key)
                    dest_enc._encoded = _merge_chunk_id_encodings(
                        dest_enc._encoded, src_enc_arr, s, e
                    )
                if right_edge_samples:
                    s, e = right_edge_samples
                    dest_tensor.extend(src_tensor[s:e])
            dest_ds.flush()
            dest_storage[chunk_map_key] = chunk_map.tobytes()
            if src_meta.min_shape:
                dest_meta.update_shape_interval(src_meta.min_shape)
                dest_meta.update_shape_interval(src_meta.max_shape)
            dest_meta.length = dest_length
            dest_storage[dest_meta_key] = dest_meta.tobytes()
            dest_storage[
                get_chunk_id_encoder_key(dest_key, dest_commit)
            ] = dest_enc.tobytes()
        finally:
            dest_tensor.meta.links = links
    if _flush:
        links = ["_sample_id_tensor", "_sample_shape_tensor", "_sample_info_tensor"]
        for l in links:
            dest_link_tensor = getattr(dest_tensor, l, None)
            if dest_link_tensor:
                src_link_tensor = getattr(src_tensor, l, None)
                if src_link_tensor:
                    copy_tensor_slice(
                        src_ds,
                        dest_ds,
                        src_link_tensor.meta.name,
                        dest_link_tensor.meta.name,
                        indices,
                        _copy_links_only=False,
                        _flush=False,
                    )
        dest_ds.storage.clear_cache_without_flush()
        dest_ds._populate_meta()
