from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.storage import LRUCache
from hub.util.keys import get_dataset_meta_key, get_tensor_commit_diff_key


def compare_commits(
    id_1: str, id_2: str, version_state: Dict[str, Any], storage: LRUCache
) -> Tuple[dict, dict]:
    """Compares two commits and returns the differences.

    Args:
        id_1 (str): The first commit_id or branch name.
        id_2 (str): The second commit_id or branch name.
        version_state (dict): The version state.
        storage (LRUCache): The underlying storage of the dataset.

    Returns:
        Tuple[dict, dict]: The changes made in the first commit and second commit respectively.
    """
    id_1 = sanitize_commit(id_1, version_state)
    id_2 = sanitize_commit(id_2, version_state)
    commit_node_1: CommitNode = version_state["commit_node_map"][id_1]
    commit_node_2: CommitNode = version_state["commit_node_map"][id_2]
    lca_id = get_lowest_common_ancestor(commit_node_1, commit_node_2)
    lca = version_state["commit_node_map"][lca_id]

    changes_1: Dict[str, Dict] = defaultdict(dict)
    changes_2: Dict[str, Dict] = defaultdict(dict)

    for commit_node, changes in [
        (commit_node_1, changes_1),
        (commit_node_2, changes_2),
    ]:
        while commit_node != lca:
            commit_id = commit_node.commit_id
            get_changes_for_id(commit_id, storage, changes)
            commit_node = commit_node.parent  # type: ignore
        filter_data_updated(changes)
    return changes_1, changes_2


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


def get_all_changes_string(changes1, message1, changes2, message2):
    """Returns a string with all changes."""
    all_changes = ["\n## Hub Diff"]
    separator = "-" * 120
    if changes1 is not None:
        changes1_str = get_changes_str(changes1, message1, separator)
        all_changes.append(changes1_str)
    if changes2 is not None:
        changes2_str = get_changes_str(changes2, message2, separator)
        all_changes.append(changes2_str)
    all_changes.append(separator)
    return "\n".join(all_changes)


def get_changes_str(changes: Dict, message: str, separator: str):
    """Returns a string with changes made."""
    all_changes = [separator, message]
    for tensor, change in changes.items():

        data_added = change["data_added"]
        data_added_str = convert_adds_to_string(data_added)

        data_updated = change["data_updated"]

        created = change.get("created", False)

        info_updated = change.get("info_updated", False)

        has_change = created or data_added_str or data_updated or info_updated
        if has_change:
            all_changes.append(tensor)
            if created:
                all_changes.append("* Created tensor")

            if data_added_str:
                all_changes.append(data_added_str)

            if data_updated:
                output = convert_updates_to_string(data_updated)
                all_changes.append(output)

            if info_updated:
                all_changes.append("* Info updated")
            all_changes.append("")
    if len(all_changes) == 2:
        all_changes.append("No changes were made.")
    return "\n".join(all_changes)


def get_changes_for_id(commit_id: str, storage: LRUCache, changes: Dict[str, Dict]):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    meta_key = get_dataset_meta_key(commit_id)
    meta = storage.get_cachable(meta_key, DatasetMeta)

    for tensor in meta.tensors:
        try:
            commit_diff_key = get_tensor_commit_diff_key(tensor, commit_id)
            commit_diff: CommitDiff = storage.get_cachable(commit_diff_key, CommitDiff)
            change = changes[tensor]

            change["created"] = change.get("created") or commit_diff.created
            change["info_updated"] = (
                change.get("info_updated") or commit_diff.info_updated
            )

            # this means that the data was transformed inplace in a newer commit, so we can ignore older diffs
            if change.get("data_transformed_in_place", False):
                continue

            if "data_added" not in change:
                change["data_added"] = commit_diff.data_added.copy()
            else:
                change["data_added"][0] = commit_diff.data_added[0]

            if "data_updated" not in change:
                change["data_updated"] = commit_diff.data_updated.copy()
            else:
                change["data_updated"].update(commit_diff.data_updated)
            change["data_transformed_in_place"] = (
                change.get("data_transformed_in_place") or commit_diff.data_transformed
            )
        except KeyError:
            pass


def filter_data_updated(changes: Dict[str, Dict]):
    """Removes the intersection of data added and data updated from data updated."""
    for change in changes.values():
        # only show the elements in data_updated that are not in data_added
        data_added_range = range(change["data_added"][0], change["data_added"][1] + 1)
        upd = {data for data in change["data_updated"] if data not in data_added_range}
        change["data_updated"] = upd


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


def convert_updates_to_string(indexes: Set[int]) -> str:
    range_intervals = compress_into_range_intervals(indexes)
    output = range_interval_list_to_string(range_intervals)

    num_samples = len(indexes)
    sample_string = "sample" if num_samples == 1 else "samples"
    return f"* Updated {num_samples} {sample_string}: [{output}]"


def convert_adds_to_string(index_range: List[int]) -> str:
    num_samples = index_range[1] - index_range[0]
    if num_samples == 0:
        return ""
    sample_string = "sample" if num_samples == 1 else "samples"
    return f"* Added {num_samples} {sample_string}: [{index_range[0]}-{index_range[1]}]"
