from deeplake.core.version_control.commit_chunk_set import CommitChunkSet
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap
from deeplake.util.compute import get_compute_provider
from deeplake.util.keys import (
    get_tensor_meta_key,
    get_tensor_info_key,
    get_tensor_tile_encoder_key,
    get_creds_encoder_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_chunk_id_encoder_key,
    get_sequence_encoder_key,
    get_chunk_key,
)
from functools import partial
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


def _get_chunks_for_tensor(tensor):
    key = tensor.key
    eng = tensor.chunk_engine
    enc = eng.chunk_id_encoder
    keys = []
    commits = []
    cnames = []
    chunkids = enc._encoded[:, 0]
    for cid in chunkids:
        cname = enc.name_from_id(cid)
        commit = eng.get_chunk_commit(cname)
        keys.append(get_chunk_key(key, cname, commit))
        cnames.append(cname)
        commits.append(commit)
    return keys, cnames, commits


def _copy_objects(key_pairs, src_storage, dest_storage):
    for src_key, dest_key in zip(*key_pairs):
        try:
            dest_storage[dest_key] = src_storage[src_key]
        except KeyError as ke:
            pass


def _copy_objects_with_progressbar(pg_callback, key_pairs, src_storage, dest_storage):
    for src_key, dest_key in zip(*key_pairs):
        try:
            dest_storage[dest_key] = src_storage[src_key]
        except KeyError:
            pass
        pg_callback(1)


def copy_tensors(
    src_ds,
    dest_ds,
    src_tensor_names,
    dest_tensor_names=None,
    scheduler="threaded",
    num_workers=0,
    progressbar=True,
    shared_chunks=True,
):
    if not src_ds.read_only:
        src_ds.flush()
    dest_ds.flush()
    src_path = src_ds.path
    dest_path = dest_ds.path
    same_ds = shared_chunks and src_path == dest_path
    src_tensor_names = list(src_tensor_names)
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
        src_tensor = src_ds[src_tensor_name]
        src_key = src_tensor.key
        same_key = src_tensor.key == dest_tensor_name
        src_tensor_key = src_tensor.key
        dest_commit_id = dest_ds.pending_commit_id

        _src_keys, chunk_names, commits = _get_chunks_for_tensor(src_tensor)
        if same_ds:
            if not same_key:
                commits = [f"{c}@{src_key}" for c in commits]
            dest_chunk_map_key = get_tensor_commit_chunk_map_key(
                dest_tensor_name, dest_commit_id
            )
            dest_chunk_map = CommitChunkMap()
            dest_chunk_map.chunks = dict(zip(chunk_names, commits))
            dest_storage[dest_chunk_map_key] = dest_chunk_map.tobytes()
        else:
            src_keys += _src_keys
            dest_keys += [
                get_chunk_key(dest_tensor_name, chunk_name, dest_commit_id)
                for chunk_name in chunk_names
            ]
        src_keys += _get_meta_files_for_tensor(src_tensor_key, src_ds.pending_commit_id)
        dest_keys += _get_meta_files_for_tensor(dest_tensor_name, dest_commit_id)
        dest_commit_diff = CommitDiff(0, True)
        dest_commit_diff.add_data(src_tensor.meta.length)
        dest_commit_diff_key = get_tensor_commit_diff_key(
            dest_tensor_name, dest_commit_id
        )
        dest_storage[dest_commit_diff_key] = dest_commit_diff.tobytes()
        dest_chunk_set_key = get_tensor_commit_chunk_set_key(
            dest_tensor_name, dest_commit_id
        )
        updated_dest_keys = [dest_commit_diff_key]
        if same_ds:
            updated_dest_keys.append(dest_chunk_map_key)
        else:
            dest_chunk_set = CommitChunkSet()
            dest_chunk_set.chunks = set(chunk_names)
            dest_storage[dest_chunk_set_key] = dest_chunk_set.tobytes()
            updated_dest_keys.append(dest_chunk_set_key)
    total = len(src_keys)
    compute = get_compute_provider(scheduler, num_workers)
    if same_ds:
        _copy_objects((src_keys, dest_keys), src_storage, dest_storage)
    else:
        if num_workers <= 1:
            inputs = [(src_keys, dest_keys)]
        else:
            inputs = [
                (src_keys[i::num_workers], dest_keys[i::num_workers])
                for i in range(num_workers)
            ]
        if progressbar:

            compute.map_with_progressbar(
                partial(
                    _copy_objects_with_progressbar,
                    src_storage=src_storage,
                    dest_storage=dest_storage,
                ),
                inputs,
                total,
                "Copying tensor data",
            )
        else:
            compute.map(
                partial(
                    _copy_objects, src_storage=src_storage, dest_storage=dest_storage
                ),
                inputs,
            )
    dest_ds_meta.tensors += dest_tensor_names
    dest_ds_meta.tensor_names.update({k: k for k in dest_tensor_names})
    dest_ds_meta.hidden_tensors += hidden_tensors
    dest_ds_meta.is_dirty = True
    dest_ds.flush()
    dest_ds.storage.clear_cache()
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
    n2 = len(enc2)
    if not n2:
        return enc1
    if start == 0:
        old_offset = 0
    else:
        old_offset = enc2[start - 1, 1] + 1
    new_offset = enc1[-1, 1] + 1
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
    # start_tiled = start_row != nrows - 1 and arr[start_row, 0] == arr[start_row + 1, 0]
    # end_tiled = orig_end_row != end_row
    num_required_chunks = end_row + 1 - start_row
    start_chunk_aligned = False
    end_chunk_aligned = False
    if start_row == 0:
        if start == 0:
            start_chunk_aligned = True
    else:
        prev_row = start_row - 1
        if start == arr[prev_row] + 1:
            start_chunk_aligned = True
    if arr[end_row, 1] == end - 1:
        end_chunk_aligned = True

    if num_required_chunks == 1 and not (start_chunk_aligned and end_chunk_aligned):
        return None, (start, end), None
    elif num_required_chunks == 2 and not start_chunk_aligned and not end_chunk_aligned:
        return None, (start, end), None
    elif start_chunk_aligned and not end_chunk_aligned:
        return (start_row, end_row), None, (arr[end_row - 1, 1] + 1, end)
    elif end_chunk_aligned and not start_chunk_aligned:
        return (start_row + 1, end_row + 1), (start, arr[start_row, 1] + 1), None
    elif not start_chunk_aligned and not end_chunk_aligned:
        return (start_row + 1, end_row), (start, arr[start_row, 1] + 1), (arr[end_row - 1, 1] + 1, end)


def copy_tensor_slice(src_ds, dest_ds, src_tensor_name, dest_tensor_name, indices):
    src_tensor = src_ds[src_tensor_name]
    dest_tensor = dest_ds[dest_tensor_name]
    src_key = src_tensor.key
    dest_key = dest_tensor.key
    same_key = src_key == dest_key
    dest_commit = dest_ds.pending_commit_id
    src_eng = src_tensor.chunk_engine
    src_enc = src_eng.chunk_id_encoder
    dest_enc = dest_tensor.chunk_engine.chunk_id_encoder
    src_enc_arr = src_enc._encoded
    ranges = _group_ranges(indices)
    orig_length = dest_tensor.meta.length
    dest_storage = dest_ds.base_storage
    links = dest_tensor.meta.links
    dest_tensor.meta.links = {}
    num_added = 0
    for start, end in ranges:
        chunks_to_copy, left_edge_samples, right_edge_samples = _get_required_chunks_for_range(
            src_tensor, start, end
        )
        if left_edge_samples:
            s, e = left_edge_samples
            dest_tensor.extend(dest_tensor[s:e])
        if chunks_to_copy:
            s, e = chunks_to_copy
            chunk_ids = src_enc_arr[s:e, 0]
            chunk_names = list(map(src_enc.name_from_id, chunk_ids))
            chunk_commits = list(map(src_eng.get_chunk_commit, chunk_names))
            if not same_key:
                chunk_commits = [f"{c}@{src_key}" for c in chunk_commits]
            chunk_map_key = get_tensor_commit_chunk_map_key(dest_key, dest_commit)
            try:
                chunk_map = dest_storage[chunk_map_key]
            except KeyError:
                chunk_map = CommitChunkMap()
            chunk_map.chunks.update(dict(zip(chunk_names, chunk_commits)))
            dest_storage[chunk_map_key] = chunk_map
            dest_enc._encoded = _merge_chunk_id_encodings(dest_enc._encoded, src_enc_arr, s, e)
        if right_edge_samples:
            s, e = right_edge_samples
            dest_tensor.extend(dest_tensor[s:e])
        num_added = end - start
    dest_tensor.meta.links = links
    dest_tensor.meta.length = orig_length + num_added
    dest_ds_tensor_names = {v:k for k,v in dest_ds.meta.tensor_names.items()}
    src_ds_tensor_names = {v:k for k,v in src_ds.meta.tensor_names.items()}
    for k in dest_tensor.links:
        src_tname = src_ds_tensor_names[k]
        dest_tname = dest_ds_tensor_names[k]
        copy_tensor_slice(src_ds, dest_ds, src_tname, dest_tname, indices)



        
        
