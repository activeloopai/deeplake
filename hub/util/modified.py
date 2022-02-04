from typing import Any, Dict, List, Optional, Tuple

from hub.core.storage import LRUCache
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.commit_node import CommitNode
from hub.util.diff import sanitize_commit
from hub.util.exceptions import TensorModifiedError
from hub.util.keys import get_tensor_commit_diff_key


def get_modified_indexes(
    tensor: str,
    current_commit_id: str,
    target_id: Optional[str],
    version_state: Dict[str, Any],
    storage: LRUCache,
) -> List[int]:
    if target_id is None:
        indexes, _ = get_modified_indexes_for_commit(tensor, current_commit_id, storage)
        return indexes

    indexes = []
    target_id = sanitize_commit(target_id, version_state)
    commit_node_map = version_state["commit_node_map"]
    current_node: CommitNode = commit_node_map[current_commit_id]
    target_node: CommitNode = commit_node_map[target_id]

    if not check_ancestor(current_node, target_node):
        raise TensorModifiedError

    while current_node.commit_id != target_node.commit_id:
        commit_id = current_node.commit_id
        idxes, stop = get_modified_indexes_for_commit(tensor, commit_id, storage)
        indexes.extend(idxes)
        if stop:
            break
        current_node = current_node.parent
    return indexes


def get_modified_indexes_for_commit(
    tensor: str, commit_id: str, storage: LRUCache
) -> Tuple[List[int], bool]:
    indexes = []
    try:
        commit_diff_key = get_tensor_commit_diff_key(tensor, commit_id)
        commit_diff: CommitDiff = storage.get_cachable(commit_diff_key, CommitDiff)

        data_added = list(range(*commit_diff.data_added))
        data_updated = commit_diff.data_updated

        indexes.extend(data_added)
        indexes.extend(data_updated)

        stop = commit_diff.data_transformed
        return indexes, stop
    except KeyError:
        return [], False


def check_ancestor(current_node: CommitNode, target_node: CommitNode) -> bool:
    """Checks if the target node is an ancestor of the current node."""
    target_id = target_node.commit_id
    while current_node is not None:
        if current_node.commit_id == target_id:
            return True
        current_node = current_node.parent
    return False
