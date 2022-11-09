from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
from deeplake.core.meta.dataset_meta import DatasetMeta
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.commit_node import CommitNode  # type: ignore
from deeplake.core.storage import LRUCache
from deeplake.core.version_control.dataset_diff import DatasetDiff
from deeplake.util.keys import (
    get_dataset_diff_key,
    get_dataset_meta_key,
    get_tensor_commit_diff_key,
)


def get_changes_and_messages(
    version_state, storage, id_1, id_2
) -> Tuple[
    List[dict],
    Optional[List[dict]],
    List[dict],
    Optional[List[dict]],
    Optional[str],
    str,
    Optional[str],
]:
    if id_1 is None and id_2 is None:
        return get_changes_and_messages_compared_to_prev(version_state, storage)
    return get_changes_and_message_2_ids(version_state, storage, id_1, id_2)


def get_changes_and_messages_compared_to_prev(
    version_state, storage
) -> Tuple[List[dict], None, List[dict], None, None, str, None]:
    commit_node = version_state["commit_node"]
    commit_id = commit_node.commit_id
    head = commit_node.is_head_node

    tensor_changes: List[dict] = []
    ds_changes: List[dict] = []
    s = "HEAD" if head else f"{commit_id} (current commit)"
    msg_1 = f"Diff in {s} relative to the previous commit:\n"
    get_tensor_changes_for_id(commit_node, storage, tensor_changes)
    get_dataset_changes_for_id(commit_node, storage, ds_changes)
    return ds_changes, None, tensor_changes, None, None, msg_1, None


def get_changes_and_message_2_ids(
    version_state, storage, id_1, id_2
) -> Tuple[List[dict], List[dict], List[dict], List[dict], str, str, str]:
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
) -> Tuple[List[dict], List[dict], List[dict], List[dict], str]:
    """Compares two commits and returns the differences.

    Args:
        id_1 (str): The first commit_id or branch name.
        id_2 (str): The second commit_id or branch name.
        version_state (dict): The version state.
        storage (LRUCache): The underlying storage of the dataset.

    Returns:
        Tuple[List[dict], List[dict], List[dict], List[dict], str]: The differences between the two commits and the id of the lowest common ancestor.
    """
    id_1 = sanitize_commit(id_1, version_state)
    id_2 = sanitize_commit(id_2, version_state)
    mp = version_state["commit_node_map"]
    commit_node_1: CommitNode = mp[id_1]
    commit_node_2: CommitNode = mp[id_2]
    lca_id = get_lowest_common_ancestor(commit_node_1, commit_node_2)
    lca_node: CommitNode = mp[lca_id]

    tensor_changes_1: List[dict] = []
    tensor_changes_2: List[dict] = []
    dataset_changes_1: List[dict] = []
    dataset_changes_2: List[dict] = []

    for commit_node, tensor_changes, dataset_changes in [
        (commit_node_1, tensor_changes_1, dataset_changes_1),
        (commit_node_2, tensor_changes_2, dataset_changes_2),
    ]:
        while commit_node.commit_id != lca_node.commit_id:
            get_tensor_changes_for_id(commit_node, storage, tensor_changes)
            get_dataset_changes_for_id(commit_node, storage, dataset_changes)
            commit_node = commit_node.parent  # type: ignore

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
    all_changes = ["\n## Deep Lake Diff"]
    if msg_0:
        all_changes.append(msg_0)

    separator = "-" * 120
    if tensor_changes_1 is not None:
        changes1_str = get_changes_str(
            ds_changes_1, tensor_changes_1, colour_string(msg_1, "blue"), separator
        )
        all_changes.append(changes1_str)
    if tensor_changes_2 is not None:
        changes2_str = get_changes_str(
            ds_changes_2, tensor_changes_2, colour_string(msg_2, "blue"), separator
        )
        all_changes.append(changes2_str)
    all_changes.append(separator)
    return "\n".join(all_changes)


def colour_string(string: str, colour: str) -> str:
    """Returns a coloured string."""
    if colour == "yellow":
        return "\033[93m" + string + "\033[0m"
    elif colour == "blue":
        return "\033[94m" + string + "\033[0m"
    return string


def get_changes_str(
    ds_changes: List, tensor_changes: List, message: str, separator: str
):
    """Returns a string with changes made."""
    all_changes = [separator, message]
    local_separator = "*" * 80
    for ds_change, tensor_change in zip(ds_changes, tensor_changes):
        commit_id = ds_change["commit_id"]
        author = ds_change["author"]
        message = ds_change["message"]
        date = ds_change["date"]
        assert commit_id == tensor_change["commit_id"]
        if date is None:
            commit_id = "UNCOMMITTED HEAD"
        else:
            date = str(date)

        all_changes_for_commit = [
            local_separator,
            colour_string(f"commit {commit_id}", "yellow"),
            f"Author: {author}",
            f"Date: {date}",
            f"Message: {message}",
            "",
        ]
        get_dataset_changes_str_list(ds_change, all_changes_for_commit)
        get_tensor_changes_str_list(tensor_change, all_changes_for_commit)
        if len(all_changes_for_commit) == 6:
            all_changes_for_commit.append("No changes were made in this commit.")
        all_changes.extend(all_changes_for_commit)
    if len(all_changes) == 2:
        all_changes.append("No changes were made.\n")
    return "\n".join(all_changes)


def get_dataset_changes_str_list(ds_change: Dict, all_changes_for_commit: List[str]):
    if ds_change.get("info_updated", False):
        all_changes_for_commit.append("- Updated dataset info \n")
    if ds_change.get("deleted"):
        for name in ds_change["deleted"]:
            all_changes_for_commit.append(f"- Deleted:\t{name}")
    if ds_change.get("renamed"):
        for old, new in ds_change["renamed"].items():
            all_changes_for_commit.append(f"- Renamed:\t{old} -> {new}")
    if len(all_changes_for_commit) > 6:
        all_changes_for_commit.append("\n")


def get_tensor_changes_str_list(tensor_change: Dict, all_changes_for_commit: List[str]):
    tensors = sorted(tensor_change.keys())
    for tensor in tensors:
        if tensor == "commit_id":
            continue
        change = tensor_change[tensor]
        if not has_change(change):
            continue
        all_changes_for_commit.append(tensor)
        if change["created"]:
            all_changes_for_commit.append("* Created tensor")

        if change["cleared"]:
            all_changes_for_commit.append("* Cleared tensor")

        data_added = change.get("data_added", [0, 0])
        data_added_str = convert_adds_to_string(data_added)
        if data_added_str:
            all_changes_for_commit.append(data_added_str)

        data_updated = change["data_updated"]
        if data_updated:
            output = convert_updates_deletes_to_string(data_updated, "Updated")
            all_changes_for_commit.append(output)

        data_deleted = change["data_deleted"]
        if data_deleted:
            output = convert_updates_deletes_to_string(data_deleted, "Deleted")
            all_changes_for_commit.append(output)

        info_updated = change["info_updated"]
        if info_updated:
            all_changes_for_commit.append("* Updated tensor info")
        all_changes_for_commit.append("")


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
    commit_node,
    storage: LRUCache,
    dataset_changes,
):
    """Returns the changes made in the dataset for a commit."""
    commit_id = commit_node.commit_id
    dataset_diff_key = get_dataset_diff_key(commit_id)

    time = str(commit_node.commit_time)[:-7] if commit_node.commit_time else None
    dataset_change = {
        "commit_id": commit_id,
        "author": commit_node.commit_user_name,
        "message": commit_node.commit_message,
        "date": time,
    }
    try:
        dataset_diff = storage.get_deeplake_object(dataset_diff_key, DatasetDiff)
    except KeyError:
        changes = {"info_updated": False, "renamed": {}, "deleted": []}
        dataset_change.update(changes)
        dataset_changes.append(dataset_change)
        return

    changes = {
        "info_updated": dataset_diff.info_updated,
        "renamed": dataset_diff.renamed.copy(),
        "deleted": dataset_diff.deleted.copy(),
    }
    dataset_change.update(changes)
    dataset_changes.append(dataset_change)


def get_tensor_changes_for_id(
    commit_node,
    storage: LRUCache,
    tensor_changes: List[Dict],
):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    commit_id = commit_node.commit_id
    meta_key = get_dataset_meta_key(commit_id)
    meta: DatasetMeta = storage.get_deeplake_object(meta_key, DatasetMeta)
    tensors = meta.visible_tensors

    commit_changes = {"commit_id": commit_id}
    for tensor in tensors:
        key = meta.tensor_names[tensor]
        commit_diff_key = get_tensor_commit_diff_key(key, commit_id)
        try:
            commit_diff: CommitDiff = storage.get_deeplake_object(
                commit_diff_key, CommitDiff
            )
        except KeyError:
            tensor_change = {
                "created": False,
                "cleared": False,
                "info_updated": False,
                "data_added": [0, 0],
                "data_updated": set(),
                "data_deleted": set(),
                "data_transformed_in_place": False,
            }

            commit_changes[tensor] = tensor_change
            continue
        tensor_change = {
            "created": commit_diff.created,
            "cleared": commit_diff.cleared,
            "info_updated": commit_diff.info_updated,
            "data_added": commit_diff.data_added.copy(),
            "data_updated": commit_diff.data_updated.copy(),
            "data_deleted": commit_diff.data_deleted.copy(),
            "data_transformed_in_place": commit_diff.data_transformed,
        }
        commit_changes[tensor] = tensor_change

    tensor_changes.append(commit_changes)


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


def merge_renamed_deleted(dataset_changes):
    deleted = []
    renamed = OrderedDict()
    done = set()
    merge_renamed = {}
    for dataset_change in dataset_changes:
        for old, new in dataset_change["renamed"].items():
            if deleted and new in deleted and new not in done:
                deleted[deleted.index(new)] = old
                done.add(new)
                continue
            if renamed and renamed.get(new):
                merge_renamed[old] = renamed[new]
                renamed.pop(new)
            else:
                merge_renamed[old] = new
        deleted.extend(dataset_change["deleted"])
    return merge_renamed, deleted
