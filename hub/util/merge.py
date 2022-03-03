from collections import defaultdict
from typing import List
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode
from hub.util.diff import get_lowest_common_ancestor, sanitize_commit
from hub.util.exceptions import MergeError
from hub.util.keys import get_tensor_commit_diff_key
from hub.util.remove_cache import create_read_copy_dataset
from hub.util.version_control import auto_checkout, auto_commit, commit


def merge(dataset, target_id, strategy=0):
    version_state = dataset.version_state
    auto_checkout(dataset)
    target_commit_id = sanitize_commit(target_id, version_state)
    target_commit_id = auto_commit_target_commit(dataset, target_commit_id)
    target_ds = create_read_copy_dataset(dataset, target_commit_id)

    original_node: CommitNode = version_state["commit_node"]
    target_node: CommitNode = version_state["commit_node_map"][target_commit_id]

    original_tensor_names = dataset.tensors.keys()
    target_tensor_names = target_ds.tensors.keys()

    new_tensor_names = target_tensor_names - original_tensor_names
    common_tensor_names = target_tensor_names & original_tensor_names
    deleted_tensor_names = original_tensor_names - target_tensor_names

    merge_common_tensors(dataset, target_ds, common_tensor_names, strategy)
    copy_new_tensors(dataset, target_ds, new_tensor_names)
    delete_tensors(dataset, deleted_tensor_names)

    original_node.merge_from(target_node)
    commit(dataset, f"Merge {target_id} into {dataset.branch}")


def auto_commit_target_commit(dataset, target_commit_id: str):
    original_id = dataset.pending_commit_id
    original_branch = dataset.branch
    dataset.checkout(target_commit_id)
    auto_commit(dataset, f"Auto commit before merging into {original_branch}")
    target_commit_id = dataset.pending_commit_id
    dataset.checkout(original_id)
    return target_commit_id


def get_changes_commit_ids_for_node(
    dataset, tensor_name: str, commit_node: CommitNode, lca_node: CommitNode
):
    changes_commit_map = defaultdict(list)
    current_node = commit_node
    while current_node.commit_id != lca_node.commit_id:
        commit_id = current_node.commit_id
        if current_node.is_merge_node:
            changes = get_changes_commit_ids_for_node(
                dataset, tensor_name, current_node.merge_parent, lca_node
            )
            for idx in changes:
                changes_commit_map[idx].extend(changes[idx])
        else:
            try:
                diff_key = get_tensor_commit_diff_key(tensor_name, commit_id)
                diff: CommitDiff = dataset.storage.get_cachable(diff_key, CommitDiff)
                data_updated = sorted(diff.data_updated)
                id_tensor_name = tensor_name + "id"
                id_tensor = dataset[id_tensor_name]
                for idx in data_updated:
                    sample_id = id_tensor[idx].numpy()[0]
                    changes_commit_map[sample_id].append(commit_id)
            except KeyError:
                pass
        current_node = current_node.parent
    return changes_commit_map


def delete_tensors(dataset, tensor_names: List[str]):
    for tensor_name in tensor_names:
        dataset.delete_tensor(tensor_name)


def copy_new_tensors(dataset, target_dataset, tensor_names: List[str]):
    for tensor_name in tensor_names:
        target_tensor = target_dataset[tensor_name]
        htype = target_tensor.meta.htype
        sample_compression = target_tensor.meta.sample_compression
        chunk_compression = target_tensor.meta.chunk_compression
        dataset.create_tensor(
            tensor_name,
            htype=htype,
            sample_compression=sample_compression,
            chunk_compression=chunk_compression,
        )
        new_tensor = dataset[tensor_name]
        for item in target_tensor:
            new_tensor.append(item)
        new_tensor.info.update(target_tensor.info)


def merge_common_tensors(
    dataset,
    target_dataset,
    tensor_names: List[str],
    strategy: int = 0,
):
    check_common_tensor_conflicts(dataset, target_dataset, tensor_names)
    version_state = dataset.version_state

    commit_node_map = version_state["commit_node_map"]
    original_node: CommitNode = version_state["commit_node"]
    target_commit_id = target_dataset.pending_commit_id
    target_node: CommitNode = commit_node_map[target_commit_id]
    lca_id = get_lowest_common_ancestor(original_node, target_node)
    lca_node: CommitNode = commit_node_map[lca_id]

    for tensor_name in tensor_names:
        id_tensor_name = tensor_name + "id"
        target_tensor = target_dataset[tensor_name]
        target_id_tensor = target_dataset[id_tensor_name]
        original_tensor = dataset[tensor_name]
        original_id_tensor = dataset[id_tensor_name]

        target_id_changes_commit_map = get_changes_commit_ids_for_node(
            target_dataset, tensor_name, target_node, lca_node
        )

        original_id_changes_commit_map = get_changes_commit_ids_for_node(
            dataset, tensor_name, original_node, lca_node
        )

        original_ids = original_id_tensor.numpy().flatten()
        original_id_to_index_map = {id: i for i, id in enumerate(original_ids)}

        target_ids = target_id_tensor.numpy().flatten()
        target_id_to_index_map = {id: i for i, id in enumerate(target_ids)}

        new_elements_ids = set(target_ids) - set(original_ids)

        add_new_samples_to_tensor(
            original_tensor,
            target_tensor,
            new_elements_ids,
            target_id_changes_commit_map,
            target_id_to_index_map,
        )

        # handle common elements
        merge_common_samples(
            original_tensor,
            target_tensor,
            strategy,
            original_id_changes_commit_map,
            target_id_changes_commit_map,
            original_id_to_index_map,
            target_id_to_index_map,
        )


def check_common_tensor_conflicts(
    dataset,
    target_dataset,
    tensor_names: List[str],
):
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
                raise MergeError(tensor_name, key, value, target_details[key])


def add_new_samples_to_tensor(
    original_tensor,
    target_tensor,
    new_elements_ids,
    target_id_changes_commit_map,
    target_id_to_index_map,
):
    for id in new_elements_ids:
        target_id_changes_commit_map.pop(id, None)
        idx = target_id_to_index_map[id]
        original_tensor.append(target_tensor[idx])


def merge_common_samples(
    original_tensor,
    target_tensor,
    strategy,
    original_id_changes_commit_map,
    target_id_changes_commit_map,
    original_id_to_index_map,
    target_id_to_index_map,
):
    for id in target_id_changes_commit_map:
        target_idx = target_id_to_index_map[id]
        original_idx = original_id_to_index_map[id]
        target_commit_ids = target_id_changes_commit_map[id]
        original_commit_ids = original_id_changes_commit_map[id]
        set_original_commit_ids = set(original_commit_ids)
        crop = len(target_commit_ids)
        most_recent_common_item = None
        for i, item in enumerate(target_commit_ids):
            if item in set_original_commit_ids:
                crop = i
                most_recent_common_item = item
                break
        target_commit_ids = target_commit_ids[:crop]
        if original_commit_ids[0] == most_recent_common_item or strategy == 0:
            original_tensor[original_idx] = target_tensor[target_idx]
