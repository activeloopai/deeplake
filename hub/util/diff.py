from collections import defaultdict, OrderedDict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.storage import LRUCache
from hub.core.version_control.dataset_diff import DatasetDiff
from hub.util.keys import (
    get_dataset_diff_key,
    get_dataset_meta_key,
    get_tensor_commit_diff_key,
)


def get_changes_and_messages(
    version_state, storage, id_1, id_2
) -> Tuple[
    dict, Optional[dict], dict, Optional[dict], Optional[str], str, Optional[str]
]:
    if id_1 is None and id_2 is None:
        return get_changes_and_messages_compared_to_prev(version_state, storage)
    return get_changes_and_message_2_ids(version_state, storage, id_1, id_2)


def get_changes_and_messages_compared_to_prev(
    version_state, storage
) -> Tuple[dict, None, dict, None, None, str, None]:
    commit_node = version_state["commit_node"]
    commit_id = commit_node.commit_id
    head = commit_node.is_head_node

    tensor_changes: Dict[str, Dict] = defaultdict(dict)
    ds_changes: Dict[str, Any] = {}
    s = "HEAD" if head else f"{commit_id} (current commit)"
    msg_1 = f"Diff in {s} relative to the previous commit:\n"
    get_tensor_changes_for_id(commit_id, storage, tensor_changes, ds_changes)
    get_dataset_changes_for_id(commit_id, storage, ds_changes)

    combine_data_deleted(tensor_changes)
    filter_cleared(tensor_changes)
    filter_data_updated(tensor_changes)
    filter_renamed_diff(ds_changes)
    remove_empty_changes(tensor_changes)

    # Order: ds_changes_1, ds_changes_2, tensor_changes_1, tensor_changes_2, msg_0, msg_1, msg_2
    return ds_changes, None, tensor_changes, None, None, msg_1, None


def get_changes_and_message_2_ids(
    version_state, storage, id_1, id_2
) -> Tuple[dict, dict, dict, dict, str, str, str]:
    commit_node = version_state["commit_node"]
    if id_1 is None:
        raise ValueError("Can't specify id_2 without specifying id_1")
    msg_0 = "The 2 diffs are calculated relative to the most recent common ancestor (%s) of the "
    if id_2 is None:
        msg_0 += "current state and the commit passed."
        id_2 = id_1
        id_1 = commit_node.commit_id
        head = commit_node.is_head_node
        msg_1 = "Diff in HEAD:\n" if head else f"Diff in {id_1} (current commit):\n"
        msg_2 = f"Diff in {id_2} (target id):\n"
    else:
        msg_0 += "two commits passed."
        msg_1 = f"Diff in {id_1} (target id 1):\n"
        msg_2 = f"Diff in {id_2} (target id 2):\n"

    ret = compare_commits(id_1, id_2, version_state, storage)
    ds_changes_1, ds_changes_2, tensor_changes_1, tensor_changes_2, lca_id = ret
    msg_0 %= lca_id
    remove_empty_changes(tensor_changes_1)
    remove_empty_changes(tensor_changes_2)
    return (
        ds_changes_1,
        ds_changes_2,
        tensor_changes_1,
        tensor_changes_2,
        msg_0,
        msg_1,
        msg_2,
    )


def compare_commits(
    id_1: str, id_2: str, version_state: Dict[str, Any], storage: LRUCache
) -> Tuple[dict, dict, dict, dict, str]:
    """Compares two commits and returns the differences.

    Args:
        id_1 (str): The first commit_id or branch name.
        id_2 (str): The second commit_id or branch name.
        version_state (dict): The version state.
        storage (LRUCache): The underlying storage of the dataset.

    Returns:
        Tuple[dict, dict, dict, dict, str]: The changes made in the first commit and second commit respectively, followed by lca_id.
    """
    id_1 = sanitize_commit(id_1, version_state)
    id_2 = sanitize_commit(id_2, version_state)
    mp = version_state["commit_node_map"]
    commit_node_1: CommitNode = mp[id_1]
    commit_node_2: CommitNode = mp[id_2]
    lca_id = get_lowest_common_ancestor(commit_node_1, commit_node_2)
    lca_node: CommitNode = mp[lca_id]

    tensor_changes_1: Dict[str, Dict] = defaultdict(dict)
    tensor_changes_2: Dict[str, Dict] = defaultdict(dict)
    dataset_changes_1: Dict[str, Any] = {}
    dataset_changes_2: Dict[str, Any] = {}

    for commit_node, tensor_changes, dataset_changes in [
        (commit_node_1, tensor_changes_1, dataset_changes_1),
        (commit_node_2, tensor_changes_2, dataset_changes_2),
    ]:
        while commit_node.commit_id != lca_node.commit_id:
            commit_id = commit_node.commit_id
            get_tensor_changes_for_id(
                commit_id, storage, tensor_changes, dataset_changes
            )
            get_dataset_changes_for_id(commit_id, storage, dataset_changes)
            commit_node = commit_node.parent  # type: ignore

        combine_data_deleted(tensor_changes)
        filter_cleared(tensor_changes)
        filter_data_updated(tensor_changes)
        filter_renamed_diff(dataset_changes)
        remove_empty_changes(tensor_changes)

    return (
        dataset_changes_1,
        dataset_changes_2,
        tensor_changes_1,
        tensor_changes_2,
        lca_id,
    )


def sanitize_commit(id: str, version_state: Dict[str, Any]) -> str:
    """Checks the id.
    If it's a valid commit_id, it is returned.
    If it's a branch name, the commit_id of the branch's head is returned.
    Otherwise a ValueError is raised.
    """
    if id in version_state["commit_node_map"]:
        return id
    elif id in version_state["branch_commit_map"]:
        return version_state["branch_commit_map"][id]
    raise KeyError(f"The commit/branch {id} does not exist in the dataset.")


def get_lowest_common_ancestor(p: CommitNode, q: CommitNode):
    """Returns the lowest common ancestor of two commits."""
    if p == q:
        return p.commit_id

    p_family = []
    q_family = set()

    while p:
        p_family.append(p.commit_id)
        p = p.parent  # type: ignore

    while q:
        q_family.add(q.commit_id)
        q = q.parent  # type: ignore
    for id in p_family:
        if id in q_family:
            return id


def get_all_changes_string(
    ds_changes_1, ds_changes_2, tensor_changes_1, tensor_changes_2, msg_0, msg_1, msg_2
):
    """Returns a string with all changes."""
    all_changes = ["\n## Hub Diff"]
    if msg_0:
        all_changes.append(msg_0)

    separator = "-" * 120
    if tensor_changes_1 is not None:
        changes1_str = get_changes_str(ds_changes_1, tensor_changes_1, msg_1, separator)
        all_changes.append(changes1_str)
    if tensor_changes_2 is not None:
        changes2_str = get_changes_str(ds_changes_2, tensor_changes_2, msg_2, separator)
        all_changes.append(changes2_str)
    all_changes.append(separator)
    return "\n".join(all_changes)


def get_changes_str(ds_changes, tensor_changes: Dict, message: str, separator: str):
    """Returns a string with changes made."""
    all_changes = [separator, message]
    if ds_changes.get("info_updated", False):
        all_changes.append("- Updated dataset info \n")
    if ds_changes.get("deleted"):
        for name in ds_changes["deleted"]:
            all_changes.append(f"- Deleted:\t{name}")
    if ds_changes.get("renamed"):
        for old, new in ds_changes["renamed"].items():
            all_changes.append(f"- Renamed:\t{old} -> {new}")
    if len(all_changes) > 2:
        all_changes.append("\n")

    tensors = sorted(tensor_changes.keys())
    for tensor in tensors:
        change = tensor_changes[tensor]
        created = change.get("created", False)
        cleared = change.get("cleared", False)
        data_added = change.get("data_added", [0, 0])
        data_added_str = convert_adds_to_string(data_added)
        data_updated = change.get("data_updated", set())
        info_updated = change.get("info_updated", False)

        data_deleted = change.get("data_deleted", set())
        all_changes.append(tensor)
        if created:
            all_changes.append("* Created tensor")

        if cleared:
            all_changes.append("* Cleared tensor")

        if data_added_str:
            all_changes.append(data_added_str)

        if data_updated:
            output = convert_updates_deletes_to_string(data_updated, "Updated")
            all_changes.append(output)

        if data_deleted:
            output = convert_updates_deletes_to_string(data_deleted, "Deleted")
            all_changes.append(output)

        if info_updated:
            all_changes.append("* Updated tensor info")
        all_changes.append("")
    if len(all_changes) == 2:
        all_changes.append("No changes were made.")
    return "\n".join(all_changes)


def has_change(change: Dict):
    created = change.get("created", False)
    cleared = change.get("cleared")
    data_added = change.get("data_added", [0, 0])
    num_samples_added = data_added[1] - data_added[0]
    data_updated = change.get("data_updated", set())
    info_updated = change.get("info_updated", False)
    data_deleted = change.get("data_deleted", set())
    return (
        created
        or cleared
        or num_samples_added > 0
        or data_updated
        or info_updated
        or data_deleted
    )


def get_dataset_changes_for_id(
    commit_id: str,
    storage: LRUCache,
    dataset_changes,
):
    """Returns the changes made in the dataset for a commit."""

    dataset_diff_key = get_dataset_diff_key(commit_id)
    try:
        dataset_diff = storage.get_hub_object(dataset_diff_key, DatasetDiff)
        dataset_changes["info_updated"] = (
            dataset_changes.get("info_updated") or dataset_diff.info_updated
        )

        renamed = dataset_changes.get("renamed")
        deleted = dataset_changes.get("deleted")
        done = []

        merge_renamed = OrderedDict()
        for old, new in dataset_diff.renamed.items():
            if deleted and new in deleted and new not in done:
                deleted[deleted.index(new)] = old
                done.append(old)
                continue
            if renamed and renamed.get(new):
                merge_renamed[old] = renamed[new]
                renamed.pop(new)
            else:
                merge_renamed[old] = new

        try:
            dataset_changes["renamed"].update(merge_renamed)
        except KeyError:
            dataset_changes["renamed"] = merge_renamed

        if dataset_changes.get("deleted"):
            dataset_changes["deleted"].extend(dataset_diff.deleted)
        else:
            dataset_changes["deleted"] = dataset_diff.deleted.copy()
    except KeyError:
        pass


def get_tensor_changes_for_id(
    commit_id: str,
    storage: LRUCache,
    tensor_changes: Dict[str, Dict],
    dataset_changes,
):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    meta_key = get_dataset_meta_key(commit_id)
    meta = storage.get_hub_object(meta_key, DatasetMeta)
    tensors = meta.visible_tensors

    for tensor in tensors:
        key = meta.tensor_names[tensor]
        try:
            commit_diff: CommitDiff
            commit_diff_key = get_tensor_commit_diff_key(key, commit_id)
            commit_diff = storage.get_hub_object(commit_diff_key, CommitDiff)
            renamed = dataset_changes.get("renamed")
            deleted = dataset_changes.get("deleted")

            if deleted and tensor in deleted:
                if commit_diff.created:
                    deleted.remove(tensor)
                continue

            if renamed:
                try:
                    new_name = renamed[tensor]
                    if commit_diff.created:
                        renamed.pop(tensor, None)
                    tensor = new_name
                except KeyError:
                    pass

            change = tensor_changes[tensor]

            change["created"] = change.get("created") or commit_diff.created
            # ignore older diffs if tensor was cleared
            if change.get("cleared"):
                continue

            change["cleared"] = change.get("cleared") or commit_diff.cleared

            change["info_updated"] = (
                change.get("info_updated") or commit_diff.info_updated
            )

            # this means that the data was transformed inplace in a newer commit, so we can ignore older diffs
            if change.get("data_transformed_in_place", False):
                continue

            if "data_added" not in change:
                change["data_added"] = commit_diff.data_added.copy()
            else:
                change["data_added"][0] = (
                    change["data_added"][0] - commit_diff.num_samples_added
                )

            if "data_updated" not in change:
                change["data_updated"] = commit_diff.data_updated.copy()
            else:
                change["data_updated"].update(commit_diff.data_updated)
            change["data_transformed_in_place"] = (
                change.get("data_transformed_in_place") or commit_diff.data_transformed
            )

            if "data_deleted_list" not in change:
                change["data_deleted_list"] = [commit_diff.data_deleted]
            else:
                change["data_deleted_list"].append(commit_diff.data_deleted)

        except KeyError:
            pass


def combine_data_deleted(changes: Dict[str, Dict]):
    """Combines the data deleted list into a single set."""
    for change in changes.values():
        data_deleted: Set[int] = set()
        data_deleted_list = change.pop("data_deleted_list", [])
        for deleted_list in reversed(data_deleted_list):
            for index in deleted_list:
                offset = sum(i < index for i in data_deleted)
                data_deleted.add(index + offset)
        change["data_deleted"] = data_deleted


def filter_data_updated(changes: Dict[str, Dict]):
    """Removes the intersection of data added and data updated from data updated."""
    for change in changes.values():
        deleted_data = set(change.get("data_deleted", set()))
        # only show the elements in data_updated that are not in data_added
        data_added_range = range(change["data_added"][0], change["data_added"][1] + 1)
        upd = {
            data
            for data in change["data_updated"]
            if data not in data_added_range and data not in deleted_data
        }
        change["data_updated"] = upd


def filter_renamed_diff(dataset_changes):
    """Remove deleted tensors and tensors renamed to same name from diff"""
    rm = []
    renamed = dataset_changes.get("renamed")
    deleted = dataset_changes.get("deleted")
    if renamed:
        for old_name, new_name in renamed.items():
            if old_name == new_name:
                rm.append(old_name)

        for name in rm:
            renamed.pop(name)


def filter_cleared(changes: Dict[str, Dict]):
    """Removes cleared flag if created flag is true."""
    for change in changes.values():
        if change["created"] and change["cleared"]:
            change["cleared"] = False


def compress_into_range_intervals(indexes: Set[int]) -> List[Tuple[int, int]]:
    """Compresses the indexes into range intervals.
    Examples:
        compress_into_range_intervals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        >> [(1, 10)]
        compress_into_range_intervals({1,2,3,5,6,8,10})
        >> [(1, 3), (5, 6), (8,8), (10,10)]

    Args:
        indexes (Set[int]): The indexes to compress.

    Returns:
        List[Tuple[int, int]]: The compressed range intervals.
    """

    if not indexes:
        return []

    sorted_indexes = sorted(indexes)
    compressed_indexes: List[Tuple[int, int]] = []
    start = sorted_indexes[0]
    end = sorted_indexes[0]
    for index in sorted_indexes[1:]:
        if index != end + 1:
            compressed_indexes.append((start, end))
            start = index
        end = index
    compressed_indexes.append((start, end))
    return compressed_indexes


def range_interval_list_to_string(range_intervals: List[Tuple[int, int]]) -> str:
    """Converts the range intervals to a string.

    Examples:
        range_interval_list_to_string([(1, 10)])
        >>"1-10"
        range_interval_list_to_string([(1, 3), (5, 6), (8, 8) (10, 10)])
        >>"1-3, 5-6, 8, 10"

    Args:
        range_intervals (List[Tuple[int, int]]): The range intervals to convert.

    Returns:
        str: The string representation of the range intervals.
    """
    if not range_intervals:
        return ""
    output = ""
    for start, end in range_intervals:
        if start == end:
            output += f"{start}, "
        else:
            output += f"{start}-{end}, "
    return output[:-2]


def convert_updates_deletes_to_string(indexes: Set[int], operation: str) -> str:
    range_intervals = compress_into_range_intervals(indexes)
    output = range_interval_list_to_string(range_intervals)

    num_samples = len(indexes)
    sample_string = "sample" if num_samples == 1 else "samples"
    return f"* {operation} {num_samples} {sample_string}: [{output}]"


def convert_adds_to_string(index_range: List[int]) -> str:
    num_samples = index_range[1] - index_range[0]
    if num_samples == 0:
        return ""
    sample_string = "sample" if num_samples == 1 else "samples"
    return f"* Added {num_samples} {sample_string}: [{index_range[0]}-{index_range[1]}]"


def remove_empty_changes(changes: Dict):
    if not changes:
        return
    tensors = list(changes.keys())
    for tensor in tensors:
        change = changes[tensor]
        if not has_change(change):
            del changes[tensor]
