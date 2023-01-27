from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.commit_node import CommitNode
from deeplake.util.class_label import convert_to_text
from deeplake.util.diff import (
    get_lowest_common_ancestor,
    has_change,
    merge_renamed_deleted,
    sanitize_commit,
)
from deeplake.util.exceptions import (
    MergeConflictError,
    MergeMismatchError,
    MergeNotSupportedError,
    TensorDoesNotExistError,
)
from deeplake.util.remove_cache import create_read_copy_dataset
from deeplake.util.version_control import auto_checkout, auto_commit, commit
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap
from deeplake.util.keys import (
    get_sample_id_tensor_key,
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


def merge(
    dataset,
    target_id: str,
    conflict_resolution: Optional[str] = None,
    delete_removed_tensors: bool = False,
    force: bool = False,
):
    """Merge works by comparing the states of the dataset at the target commit and the current commit.
    The new tensors in the target are added. The deleted tensors in the target are removed if delete_removed_tensors is True.
    For the common tensors, we compare ids of the samples. The samples with newer ids are added to the dataset.
    For samples with the same ids, we compare the changes history of the sample and resolve conflicts according to the conflict_resolution argument.
    """
    version_state = dataset.version_state
    commit_node_map = version_state["commit_node_map"]
    auto_checkout(dataset)
    target_commit_id = sanitize_commit(target_id, version_state)
    target_commit_id = auto_commit_target_commit(dataset, target_commit_id)

    nodes: Dict[str, CommitNode] = {}
    nodes["original"] = original_node = version_state["commit_node"]
    nodes["target"] = target_node = commit_node_map[target_commit_id]
    lca_id = get_lowest_common_ancestor(original_node, target_node)
    target_ds = create_read_copy_dataset(dataset, target_commit_id)

    if lca_id == target_commit_id:
        print("No merge needed, target id is an ancestor of the current commit")
        return
    nodes["lca"] = commit_node_map[lca_id]

    (
        new_tensors,
        common_tensors,
        deleted_tensors,
    ) = get_new_common_deleted_tensors(dataset, target_ds, lca_id, force)

    merge_common_tensors(common_tensors, dataset, target_ds, nodes, conflict_resolution)
    copy_new_tensors(new_tensors, dataset, target_ds)
    delete_tensors(deleted_tensors, dataset, delete_removed_tensors)
    finalize_merge(dataset, nodes)


def get_new_common_deleted_tensors(
    dataset, target_ds, lca_id: str, force: bool
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Gets the names of tensors, that are new, common and deleted in the target commit"""
    original_tensors: Set[str] = set(dataset.tensors)
    all_original_tensors: Set[str] = set(dataset._all_tensors_filtered())
    check_id_tensors_exist(original_tensors, all_original_tensors)

    target_tensors: Set[str] = set(target_ds.tensors)
    all_target_tensors: Set[str] = set(target_ds._all_tensors_filtered())
    check_id_tensors_exist(target_tensors, all_target_tensors)

    lca_tensors = get_lca_tensors(dataset, lca_id)
    new_tensors = target_tensors - original_tensors
    common_tensors = target_tensors & original_tensors

    # present in dataset at lca, but deleted or renamed in target
    target_deleted_tensors = lca_tensors - target_tensors

    # present in dataset at lca, but deleted or renamed in original
    original_deleted_tensors = lca_tensors - original_tensors

    target_changes = target_ds.diff(lca_id, as_dict=True)
    target_tensor_diff, _ = target_changes["tensor"]
    target_dataset_diff, _ = target_changes["dataset"]

    original_dataset_diff, _ = dataset.diff(lca_id, as_dict=True)["dataset"]

    target_renamed_tensors, _ = merge_renamed_deleted(target_dataset_diff)
    original_renamed_tensors, _ = merge_renamed_deleted(original_dataset_diff)

    process_renamed_tensors(
        dataset,
        force,
        new_tensors,
        common_tensors,
        original_deleted_tensors,
        target_deleted_tensors,
        original_renamed_tensors,
        target_renamed_tensors,
    )
    process_deleted_tensors(new_tensors, original_deleted_tensors, target_tensor_diff)
    return new_tensors, common_tensors, target_deleted_tensors


def process_renamed_tensors(
    dataset,
    force,
    new_tensors,
    common_tensors,
    original_deleted_tensors,
    target_deleted_tensors,
    original_renamed_tensors,
    target_renamed_tensors,
):
    for old_tensor, new_tensor in target_renamed_tensors.items():
        if new_tensor in new_tensors:
            if not force:
                if old_tensor in original_renamed_tensors:
                    raise MergeConflictError(
                        message=f"{old_tensor} was renamed in both branches. Rename tensors to the same name to resolve the conflict or use `force=True` to register {new_tensor} as a new tensor on current branch."
                    )
                elif old_tensor in original_deleted_tensors:
                    raise MergeConflictError(
                        message=f"{old_tensor} was renamed to {new_tensor} in target but is missing from current branch. Use `force=True` to register {new_tensor} as a new tensor on current branch."
                    )
                new_tensors.discard(new_tensor)
                target_deleted_tensors.discard(old_tensor)
                dataset.rename_tensor(old_tensor, new_tensor)
                common_tensors.add(new_tensor)

        elif new_tensor in common_tensors:
            # no merge conflict if same tensor was renamed to same name on both branches
            if original_renamed_tensors.get(old_tensor) != new_tensor and not force:
                raise MergeConflictError(
                    message=f"{old_tensor} was renamed to {new_tensor} in target but another {new_tensor} exists on the current branch. Rename tensors to resolve the conflict or use `force=True` to merge {new_tensor} of target with {new_tensor} of current branch."
                )

        target_deleted_tensors.discard(old_tensor)
        original_deleted_tensors.discard(old_tensor)


def process_deleted_tensors(new_tensors, original_deleted_tensors, target_tensor_diff):
    for tensor in original_deleted_tensors:
        tensor_changed = False
        for commit_diff in target_tensor_diff:
            diff = commit_diff[tensor]
            if has_change(diff):
                tensor_changed = True
                break
        if not tensor_changed:
            new_tensors.discard(tensor)


def finalize_merge(dataset, nodes: Dict[str, CommitNode]):
    """Finalizes the merge operation by linking the nodes and subsequently commiting."""
    original_node = nodes["original"]
    target_node = nodes["target"]
    original_node.merge_from(target_node)
    target_id = target_node.commit_id
    commit(dataset, f"Merge {target_id} into {dataset.branch}")


def get_lca_tensors(dataset, lca_id: str) -> Set[str]:
    """Gets the names of tensors present in the lca commit"""
    original_id = dataset.pending_commit_id
    dataset.checkout(lca_id)
    lca_tensors: Set[str] = set(dataset.tensors.keys())
    dataset.checkout(original_id)
    return lca_tensors


def auto_commit_target_commit(dataset, target_commit_id: str) -> str:
    """Automatically commits the dataset at the target id if it is the head of a branch."""
    original_id = dataset.pending_commit_id
    original_branch = dataset.branch
    dataset.checkout(target_commit_id)
    auto_commit(dataset, f"Auto commit before merging into {original_branch}")
    target_commit_id = dataset.pending_commit_id
    dataset.checkout(original_id)
    return target_commit_id


def get_changes_commit_ids_for_node(
    dataset, tensor_name: str, commit_node: Optional[CommitNode], lca_node: CommitNode
):
    changes_commit_map = defaultdict(list)
    current_node = commit_node
    tensor_key = dataset.version_state["tensor_names"][tensor_name]
    while current_node and current_node.commit_id != lca_node.commit_id:
        commit_id = current_node.commit_id
        if current_node.is_merge_node:
            changes = get_changes_commit_ids_for_node(
                dataset, tensor_name, current_node.merge_parent, lca_node
            )
            for idx in changes:
                changes_commit_map[idx].extend(changes[idx])
        else:
            diff = get_tensor_commit_diff(dataset, tensor_key, commit_id)
            if diff is not None:
                data_updated = sorted(diff.data_updated)
                id_tensor_key = get_sample_id_tensor_key(tensor_name)
                id_tensor = dataset[id_tensor_key]
                for idx in data_updated:
                    sample_id = id_tensor[idx].numpy()[0]
                    changes_commit_map[sample_id].append(commit_id)
        current_node = current_node.parent
    return changes_commit_map


def get_tensor_commit_diff(dataset, tensor_key, commit_id):
    diff_key = get_tensor_commit_diff_key(tensor_key, commit_id)
    diff: Optional[CommitDiff]
    try:
        diff = dataset.storage.get_deeplake_object(diff_key, CommitDiff)
    except KeyError:
        diff = None
    return diff


def delete_tensors(tensor_names: Set[str], dataset, delete_removed_tensors: bool):
    """Deletes tensors from the dataset if delete_removed_tensors is True."""
    if delete_removed_tensors:
        for tensor_name in tensor_names:
            try:
                dataset.delete_tensor(tensor_name)
            # tensor could have been renamed.
            except TensorDoesNotExistError:
                pass


def clear_tensors(tensor_names: Set[str], dataset):
    for tensor_name in tensor_names:
        dataset[tensor_name].clear()


def copy_new_tensors(
    tensor_names: Set[str],
    dataset,
    target_dataset,
):
    """Copies tensors from the target_commit to the dataset."""
    copy_tensors(
        target_dataset,
        dataset,
        tensor_names,
    )


def merge_common_tensors(
    tensor_names: Set[str],
    dataset,
    target_dataset,
    nodes: Dict[str, CommitNode],
    conflict_resolution: Optional[str] = None,
):
    check_common_tensor_mismatches(tensor_names, dataset, target_dataset)
    new_samples_dict: Dict[str, List[int]] = {}
    updated_samples_dict: Dict[str, List[Tuple[int, int]]] = {}
    conflict_samples_dict: Dict[str, List[Tuple[int, int]]] = {}
    conflict_tensors = set()
    for tensor_name in tensor_names:
        (
            new_indexes,
            updated_indexes,
            conflict_indexes,
        ) = find_new_updated_and_conflict_indexes(
            tensor_name,
            dataset,
            target_dataset,
            nodes,
        )
        new_samples_dict[tensor_name] = new_indexes
        updated_samples_dict[tensor_name] = updated_indexes
        if conflict_indexes:
            conflict_samples_dict[tensor_name] = conflict_indexes
            conflict_tensors.add(tensor_name)

    if conflict_tensors and conflict_resolution is None:
        # There are conflicts and a conflict resolution strategy has not been specified, unable to merge
        raise MergeConflictError(conflict_tensors)

    for tensor_name in tensor_names:
        merge_tensor_data(
            tensor_name,
            dataset,
            target_dataset,
            new_samples_dict,
            updated_samples_dict,
            conflict_samples_dict,
            conflict_resolution,
        )


def check_common_tensor_mismatches(tensor_names: Set[str], dataset, target_dataset):
    """Checks common tensors for mismatches in htype, sample_compression and chunk_compression."""
    for tensor_name in tensor_names:
        target_meta = target_dataset[tensor_name].meta
        original_meta = dataset[tensor_name].meta
        original_details = {
            "htype": original_meta.htype,
            "sample_compression": original_meta.sample_compression,
            "chunk_compression": original_meta.chunk_compression,
        }
        target_details = {
            "htype": target_meta.htype,
            "sample_compression": target_meta.sample_compression,
            "chunk_compression": target_meta.chunk_compression,
        }
        for key, value in original_details.items():
            if value != target_details[key]:
                raise MergeMismatchError(tensor_name, key, value, target_details[key])


def get_indexes_from_ids(
    elements_ids, id_changes_commit_map, id_to_index_map
) -> List[int]:
    new_indexes: List[int] = []
    for id in elements_ids:
        id_changes_commit_map.pop(id, None)
        idx = id_to_index_map[id]
        new_indexes.append(idx)
    return new_indexes


def find_updated_and_conflicts(
    original_id_changes_commit_map,
    target_id_changes_commit_map,
    original_id_to_index_map,
    target_id_to_index_map,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Finds the conflicts between the original commit and target id.

    Args:
        original_id_changes_commit_map: A dictionary mapping sample ids to a list of commit ids that modified the sample.
        target_id_changes_commit_map: A dictionary mapping sample ids to a list of commit ids that modified the sample.
        original_id_to_index_map: A dictionary mapping sample ids to their index in the original commit.
        target_id_to_index_map: A dictionary mapping sample ids to their index in the target id.

    Returns:
        A tuple of list of tuples of the form (original_idx, target_idx)
    """
    updated_indexes: List[Tuple[int, int]] = []
    conflict_indexes: List[Tuple[int, int]] = []
    for id in target_id_changes_commit_map:
        target_commit_ids = target_id_changes_commit_map[id]
        original_commit_ids = original_id_changes_commit_map[id]
        set_original_commit_ids = set(original_commit_ids)
        idx = None
        for i, item in enumerate(target_commit_ids):
            if item in set_original_commit_ids:
                idx = i
                break

        # this means that the sample was only modified in the target commit, no conflict
        if not original_commit_ids or (
            idx is not None and target_commit_ids[idx] == original_commit_ids[0]
        ):
            target_idx: int = target_id_to_index_map[id]
            original_idx: int = original_id_to_index_map[id]
            updated_indexes.append((original_idx, target_idx))

        # if no id is common or if a commit id other than the most recent commit_id is in common, there's a conflict
        elif idx is None or idx > 0:
            target_idx = target_id_to_index_map[id]
            original_idx = original_id_to_index_map[id]
            conflict_indexes.append((original_idx, target_idx))
    return updated_indexes, conflict_indexes


def find_new_updated_and_conflict_indexes(
    tensor_name: str,
    dataset,
    target_dataset,
    nodes: Dict[str, CommitNode],
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Finds the new, deleted, updated and conflict indexes between the original commit and target commit.

    Args:
        tensor_name (str): The name of the tensor to find the new and conflict indexes for.
        dataset: The original state of the dataset.
        target_dataset: The target state of the dataset.
        nodes (dict): A dictionary containing original, target and lca nodes.

    Returns:
        A tuple of the form (new_indexes, updated_indexes, conflict_indexes)
        - new_indexes is a list of indexes for new samples
        - updated_indexes is a list of tuples of the form (original_idx, target_idx)
        - conflict_indexes is a list of tuples of the form (original_idx, target_idx)
    """
    id_tensor_name = get_sample_id_tensor_key(tensor_name)
    target_id_tensor = target_dataset[id_tensor_name]
    original_id_tensor = dataset[id_tensor_name]

    original_node = nodes["original"]
    target_node = nodes["target"]
    lca_node = nodes["lca"]

    target_id_changes_commit_map = get_changes_commit_ids_for_node(
        target_dataset, tensor_name, target_node, lca_node
    )

    original_id_changes_commit_map = get_changes_commit_ids_for_node(
        dataset, tensor_name, original_node, lca_node
    )

    original_ids = original_id_tensor.numpy().flatten()
    original_id_to_index_map = {id: idx for idx, id in enumerate(original_ids)}

    target_ids = target_id_tensor.numpy().flatten()
    target_id_to_index_map = {id: idx for idx, id in enumerate(target_ids)}

    new_elements_ids = set(target_ids) - set(original_ids)

    new_indexes = get_indexes_from_ids(
        new_elements_ids, target_id_changes_commit_map, target_id_to_index_map
    )

    conflict_indexes: List[Tuple[int, int]] = []
    updated_indexes: List[Tuple[int, int]] = []
    updated_indexes, conflict_indexes = find_updated_and_conflicts(
        original_id_changes_commit_map,
        target_id_changes_commit_map,
        original_id_to_index_map,
        target_id_to_index_map,
    )
    return new_indexes, updated_indexes, conflict_indexes


def get_deleted_ids(original_ids, target_ids, lca_ids):
    deleted_ids_in_target = set(lca_ids) - set(target_ids)
    deleted_ids_in_original = set(lca_ids) - set(original_ids)
    deleted_in_both = deleted_ids_in_target & deleted_ids_in_original
    deleted_ids_in_original = deleted_ids_in_original - deleted_in_both
    deleted_ids_in_target = deleted_ids_in_target - deleted_in_both
    return deleted_ids_in_original, deleted_ids_in_target


def merge_tensor_data(
    tensor_name: str,
    dataset,
    target_dataset,
    new_samples_dict,
    updated_samples_dict,
    conflict_samples_dict,
    conflict_resolution,
):
    """Merges actual data present in 2 versions of a common tensor."""
    if conflict_resolution == "theirs" and tensor_name in conflict_samples_dict:
        updated_samples_dict[tensor_name].extend(conflict_samples_dict[tensor_name])

    original_tensor = dataset[tensor_name]
    target_tensor = target_dataset[tensor_name]
    id_tensor_name = get_sample_id_tensor_key(tensor_name)
    original_id_tensor = dataset[id_tensor_name]
    target_id_tensor = target_dataset[id_tensor_name]

    new_indexes = new_samples_dict[tensor_name]
    new_indexes.sort()
    is_class_label = target_tensor.meta.htype == "class_label"
    if is_class_label:
        class_names = target_tensor.info.class_names
        is_class_label = bool(class_names)
    copy_links_only = False
    if is_class_label:
        links = original_tensor.meta.links
        original_tensor.meta.links = {}
        try:
            for index in new_indexes:
                sample = target_tensor[index]
                sample = convert_to_text(
                    sample.numpy(), class_names, return_original=True
                )
                original_tensor.append(sample)
        finally:
            original_tensor.meta.links = links
        copy_links_only = True
    copy_tensor_slice(
        target_dataset,
        dataset,
        tensor_name,
        tensor_name,
        new_indexes,
        _copy_links_only=copy_links_only,
    )

    updated_indexes = updated_samples_dict[tensor_name]
    for original_idx, target_idx in updated_indexes:
        sample = target_tensor[target_idx]
        if is_class_label and class_names:
            sample = convert_to_text(sample.numpy(), class_names, return_original=True)
        original_tensor[original_idx] = sample


def check_id_tensors_exist(visible_tensors: Set[str], all_tensors: Set[str]):
    """Checks whether hidden id tensors exist for each tensor."""
    for tensor_name in visible_tensors:
        id_tensor = get_sample_id_tensor_key(tensor_name)
        if id_tensor not in all_tensors:
            raise MergeNotSupportedError


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
    src_ds,
    dest_ds,
    src_tensor_name,
    dest_tensor_name,
    indices,
    _copy_links_only=False,
    _flush=True,
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
