from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.storage import LRUCache
from hub.util.keys import get_dataset_meta_key, get_tensor_commit_diff_key


def compare(
    commit1: str, commit2: str, version_state: Dict[str, Any], storage: LRUCache
) -> Tuple[dict, dict]:
    """Compares two commits and returns the differences.

    Args:
        commit1 (str): The first commit to compare.
        commit2 (str): The second commit to compare.
    """
    commit_node_1: CommitNode = version_state["commit_node_map"][commit1]
    commit_node_2: CommitNode = version_state["commit_node_map"][commit2]
    lca_id = get_lowest_common_ancestor(commit_node_1, commit_node_2)
    lca = version_state["commit_node_map"][lca_id]

    changes_1 = create_changes_dict()
    changes_2 = create_changes_dict()

    for commit_node, changes in [
        (commit_node_1, changes_1),
        (commit_node_2, changes_2),
    ]:
        while commit_node != lca:
            commit_id = commit_node.commit_id
            get_changes_for_id(commit_id, storage, changes)
            commit_node = commit_node.parent
        filter_data_updated(changes)
    return changes_1, changes_2


def get_lowest_common_ancestor(p: CommitNode, q: CommitNode):
    """Returns the lowest common ancestor of two commits."""
    if p == q:
        return p

    p_family = []
    q_family = set()

    while p:
        p_family.append(p.commit_id)
        p = p.parent

    while q:
        q_family.add(q.commit_id)
        q = q.parent
    for id in p_family:
        if id in q_family:
            return id


def display_changes(changes: Optional[Dict], message: str):
    """Displays the changes made."""
    if changes is None:
        return
    separator = "-" * 120
    print()
    print(separator)
    print(message)
    tensors_created = changes["tensors_created"]
    del changes["tensors_created"]
    if tensors_created:
        print("Tensors created:")
        for tensor in tensors_created:
            print(f"* {tensor}")
        print()
    elif not changes:
        print("No changes.\n")
        return

    for tensor, change in changes.items():
        if tensor != "tensors_created" and change:
            print(f"Changes in Tensor {tensor}:")
            if change["data_added"]:
                print(f"* Added indexes: {change['data_added']}")
            if change["data_updated"]:
                print(f"* Updated indexes: {change['data_updated']}")
            print()

    print(separator)


def get_changes_for_id(commit_id: str, storage: LRUCache, changes: Dict[str, Any]):
    """Identifies the changes made in the given commit_id and updates them in the changes dict."""
    meta_key = get_dataset_meta_key(commit_id)
    meta = storage.get_cachable(meta_key, DatasetMeta)

    for tensor in meta.tensors:
        try:
            commit_diff_key = get_tensor_commit_diff_key(tensor, commit_id)
            commit_diff: CommitDiff = storage.get_cachable(commit_diff_key, CommitDiff)
            changes[tensor]["data_added"].update(commit_diff.data_added)
            changes[tensor]["data_updated"].update(commit_diff.data_updated)
            if commit_diff.created:
                changes["tensors_created"].add(tensor)
        except KeyError:
            pass


def filter_data_updated(changes: Dict[str, Any]):
    """Removes the intersection of data added and data updated from data updated."""
    for tensor, change in changes.items():
        if tensor != "tensors_created":
            # only show the elements in data_updated that are not in data_added
            change["data_updated"] = change["data_updated"] - change["data_added"]


def create_changes_dict() -> Dict[str, Any]:
    """Creates the dictionary used to store changes."""
    changes = defaultdict(lambda: defaultdict(set))
    changes["tensors_created"] = set()
    return changes
