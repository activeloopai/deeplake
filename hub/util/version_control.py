import random
import time
import hashlib
import pickle
from typing import Any, Dict, Optional
import warnings
from hub.client.log import logger
from hub.constants import FIRST_COMMIT_ID
from hub.core.fast_forwarding import ffw_dataset_meta
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.version_control.commit_diff import CommitDiff
from hub.core.version_control.dataset_diff import DatasetDiff
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from hub.core.storage import LRUCache
from hub.core.lock import Lock
from hub.util.exceptions import CheckoutError, CommitError
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_creds_encoder_key,
    get_sequence_encoder_key,
    get_dataset_diff_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_commit_diff_key,
    get_tensor_info_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_version_control_info_key,
    get_version_control_info_key_old,
    get_version_control_info_lock_key,
)
from hub.util.remove_cache import get_base_storage
from hub.hooks import dataset_committed
from datetime import datetime
import json


def _version_info_to_json(info):
    commit_node_map, branch_commit_map = (
        info["commit_node_map"],
        info["branch_commit_map"],
    )
    commits = {}
    for commit, node in commit_node_map.items():
        commits[commit] = {
            "branch": node.branch,
            "parent": node.parent.commit_id if node.parent else None,
            "children": [c.commit_id for c in node.children],
            "commit_message": node.commit_message,
            "commit_time": node.commit_time.timestamp() if node.commit_time else None,
            "commit_user_name": node.commit_user_name,
        }
    return {
        "commits": commits,
        "branches": branch_commit_map,
    }


def _version_info_from_json(info):
    commits, branch_commit_map = info["commits"], info["branches"]
    commit_node_map = {}
    stack = [FIRST_COMMIT_ID]
    while stack:
        commit_id = stack.pop()
        commit_data = commits[commit_id]
        node = CommitNode(commit_data["branch"], commit_id)
        node.commit_message = commit_data["commit_message"]
        commit_time = commit_data["commit_time"]
        node.commit_time = (
            None if commit_time is None else datetime.fromtimestamp(commit_time)
        )
        node.commit_user_name = commit_data["commit_user_name"]
        parent = commit_data["parent"]
        if parent:
            commit_node_map[parent].add_child(node)
        commit_node_map[commit_id] = node
        stack += commit_data["children"]
    return {
        "commit_node_map": commit_node_map,
        "branch_commit_map": branch_commit_map,
    }


def generate_hash() -> str:
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    hsh.update(random.randrange(0, 1000000).to_bytes(4, "big"))
    return hsh.hexdigest()


def commit(dataset, message: str = None, hash: Optional[str] = None) -> None:
    """Modifies the version state to reflect the commit and also copies required data to the new commit directory."""
    storage = dataset.storage
    version_state = dataset.version_state
    storage.check_readonly()
    # if not the head node, checkout to an auto branch that is newly created
    auto_checkout(dataset)
    stored_commit_node: CommitNode = version_state["commit_node"]
    stored_commit_id = version_state["commit_id"]
    if hash:
        if hash in version_state["commit_node_map"]:
            raise CommitError(f"Commit {hash} already exists")
        version_state["commit_id"] = hash
    else:
        version_state["commit_id"] = generate_hash()
    new_node = CommitNode(version_state["branch"], version_state["commit_id"])
    version_state["commit_node"].add_successor(new_node, message)
    version_state["commit_node"] = new_node
    version_state["branch_commit_map"][version_state["branch"]] = version_state[
        "commit_id"
    ]
    version_state["commit_node_map"][version_state["commit_id"]] = new_node
    save_version_info(version_state, storage)
    copy_metas(stored_commit_id, version_state["commit_id"], storage, version_state)
    create_commit_chunk_sets(version_state["commit_id"], storage, version_state)
    discard_old_metas(stored_commit_id, storage, version_state["full_tensors"])
    load_meta(dataset)

    commit_time = stored_commit_node.commit_time
    commit_message = stored_commit_node.commit_message
    author = stored_commit_node.commit_user_name
    dataset._send_commit_event(
        commit_message=commit_message, commit_time=commit_time, author=author
    )
    dataset_committed(dataset)


def checkout(
    dataset,
    address: str,
    create: bool = False,
    hash: Optional[str] = None,
) -> None:
    """Modifies the version state to reflect the checkout and also copies required data to the new branch directory if a new one is being created."""
    storage = dataset.storage
    version_state = dataset.version_state
    original_commit_id = version_state["commit_id"]

    if address in version_state["branch_commit_map"].keys():
        if create:
            raise CheckoutError(f"Can't create new branch, '{address}' already exists.")
        new_commit_id = version_state["branch_commit_map"][address]
        if original_commit_id == new_commit_id:
            return
        if not storage.read_only:
            storage.flush()
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = version_state["commit_node_map"][new_commit_id]
        version_state["branch"] = address

    elif address in version_state["commit_node_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create new branch, commit '{address}' already exists."
            )
        if address == original_commit_id:
            return
        if not storage.read_only:
            storage.flush()
        version_state["commit_id"] = address
        version_state["commit_node"] = version_state["commit_node_map"][address]
        version_state["branch"] = version_state["commit_node"].branch
    elif create:
        storage.check_readonly()
        # if the original commit is head of the branch, auto commit and checkout to original commit before creating new branch
        auto_commit(dataset, f"auto commit before checkout to {address}")
        if hash:
            if hash in version_state["commit_node_map"]:
                raise CommitError(f"Commit {hash} already exists")
            new_commit_id = hash
        else:
            new_commit_id = generate_hash()
        new_node = CommitNode(address, new_commit_id)
        version_state["commit_node"].add_child(new_node)
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = new_node
        version_state["branch"] = address
        version_state["commit_node_map"][new_commit_id] = new_node
        version_state["branch_commit_map"][address] = new_commit_id
        save_version_info(version_state, storage)
        copy_metas(original_commit_id, new_commit_id, storage, version_state)
        create_commit_chunk_sets(new_commit_id, storage, version_state)
        dataset._send_branch_creation_event(address)
    else:
        raise CheckoutError(
            f"Address {address} not found. If you want to create a new branch, use checkout with create=True"
        )

    discard_old_metas(
        original_commit_id,
        storage,
        version_state["full_tensors"],
    )
    load_meta(dataset)


def copy_metas(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
    version_state: Dict[str, Any],
) -> None:
    """Copies meta data from one commit to another."""
    initial_autoflush = storage.autoflush
    storage.autoflush = False

    tensors = version_state["full_tensors"]
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    src_dataset_meta = storage[src_dataset_meta_key]
    dest_dataset_meta = convert_to_bytes(src_dataset_meta)
    storage[dest_dataset_meta_key] = dest_dataset_meta

    try:
        src_dataset_info_key = get_dataset_info_key(src_commit_id)
        dest_dataset_info_key = get_dataset_info_key(dest_commit_id)
        src_dataset_info = storage[src_dataset_info_key]
        dest_dataset_info = convert_to_bytes(src_dataset_info)
        storage[dest_dataset_info_key] = dest_dataset_info
    except KeyError:
        pass

    tensor_list = list(tensors.keys())

    for tensor in tensor_list:
        src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
        dest_tensor_meta_key = get_tensor_meta_key(tensor, dest_commit_id)
        src_tensor_meta = storage[src_tensor_meta_key]
        dest_tensor_meta = convert_to_bytes(src_tensor_meta)
        storage[dest_tensor_meta_key] = dest_tensor_meta

        try:
            src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
            dest_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, dest_commit_id)
            src_chunk_id_encoder = storage[src_chunk_id_encoder_key]
            dest_chunk_id_encoder = convert_to_bytes(src_chunk_id_encoder)
            storage[dest_chunk_id_encoder_key] = dest_chunk_id_encoder
        except KeyError:
            pass

        try:
            src_tile_encoder_key = get_tensor_tile_encoder_key(tensor, src_commit_id)
            dest_tile_encoder_key = get_tensor_tile_encoder_key(tensor, dest_commit_id)
            src_tile_encoder = storage[src_tile_encoder_key]
            dest_tile_encoder = convert_to_bytes(src_tile_encoder)
            storage[dest_tile_encoder_key] = dest_tile_encoder
        except KeyError:
            pass

        try:
            src_sequence_encoder_key = get_sequence_encoder_key(tensor, src_commit_id)
            dest_sequence_encoder_key = get_sequence_encoder_key(tensor, dest_commit_id)
            src_sequence_encoder = storage[src_sequence_encoder_key]
            dest_sequence_encoder = convert_to_bytes(src_sequence_encoder)
            storage[dest_sequence_encoder_key] = dest_sequence_encoder
        except KeyError:
            pass

        try:
            src_creds_encoder_key = get_creds_encoder_key(tensor, src_commit_id)
            dest_creds_encoder_key = get_creds_encoder_key(tensor, dest_commit_id)
            src_creds_encoder = storage[src_creds_encoder_key]
            dest_creds_encoder = convert_to_bytes(src_creds_encoder)
            storage[dest_creds_encoder_key] = dest_creds_encoder
        except KeyError:
            pass

        try:
            src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
            dest_tensor_info_key = get_tensor_info_key(tensor, dest_commit_id)
            src_tensor_info = storage[src_tensor_info_key]
            dest_tensor_info = convert_to_bytes(src_tensor_info)
            storage[dest_tensor_info_key] = dest_tensor_info
        except KeyError:
            pass

    storage.autoflush = initial_autoflush
    storage.flush()


def create_commit_chunk_sets(
    dest_commit_id: str,
    storage: LRUCache,
    version_state: Dict[str, Any],
) -> None:
    """Creates commit chunk sets for all tensors in new commit."""
    tensor_list = version_state["full_tensors"].keys()
    for tensor in tensor_list:
        key = get_tensor_commit_chunk_set_key(tensor, dest_commit_id)
        storage[key] = CommitChunkSet()


def discard_old_metas(
    src_commit_id: str,
    storage: LRUCache,
    tensors: Dict,
):
    """Discards the metas of the previous commit from cache, during checkout or when a new commit is made."""
    all_src_keys = []
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    all_src_keys.append(src_dataset_meta_key)

    src_dataset_info_key = get_dataset_info_key(src_commit_id)
    all_src_keys.append(src_dataset_info_key)

    tensor_list = list(tensors.keys())

    for tensor in tensor_list:
        src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
        all_src_keys.append(src_tensor_meta_key)

        src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
        all_src_keys.append(src_chunk_id_encoder_key)

        src_tile_encoder_key = get_tensor_tile_encoder_key(tensor, src_commit_id)
        all_src_keys.append(src_tile_encoder_key)

        src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
        all_src_keys.append(src_tensor_info_key)

    for key in all_src_keys:
        storage.dirty_keys.pop(key, None)
        if key in storage.lru_sizes:
            size = storage.lru_sizes.pop(key)
            storage.cache_used -= size
        try:
            del storage.cache_storage[key]
        except KeyError:
            pass


def _merge_commit_node_maps(map1, map2):
    merged_map = {}

    def _merge_node(commit_id):
        if commit_id in map1 and commit_id in map2:
            node1 = map1[commit_id]
            node2 = map2[commit_id]
            merged_node = CommitNode(node1.branch, node2.commit_id)

            for attr in ("commit_message", "commit_user_name", "commit_time"):
                setattr(merged_node, attr, getattr(node1, attr) or getattr(node2, attr))
            for child in set(
                [node.commit_id for node in node1.children]
                + [node.commit_id for node in node2.children]
            ):
                merged_node.add_child(_merge_node(child))
        else:
            if commit_id in map1:
                orig_node = map1[commit_id]
            else:
                orig_node = map2[commit_id]
            merged_node = orig_node.copy()
            for child in [node.commit_id for node in orig_node.children]:
                merged_node.add_child(_merge_node(child))
        merged_map[commit_id] = merged_node
        return merged_node

    _merge_node(FIRST_COMMIT_ID)
    return merged_map


def _merge_version_info(info1, info2):
    commit_node_map = _merge_commit_node_maps(
        info1["commit_node_map"], info2["commit_node_map"]
    )
    branch_commit_map = {}
    branch_commit_map.update(info1["branch_commit_map"])
    branch_commit_map.update(info2["branch_commit_map"])
    return {
        "commit_node_map": commit_node_map,
        "branch_commit_map": branch_commit_map,
    }


def save_version_info(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Saves the current version info to the storage."""
    storage = get_base_storage(storage)
    lock = Lock(storage, get_version_control_info_lock_key())
    lock.acquire(timeout=10, force=True)
    key = get_version_control_info_key()
    new_version_info = {
        "commit_node_map": version_state["commit_node_map"],
        "branch_commit_map": version_state["branch_commit_map"],
    }
    try:
        old_version_info = _version_info_from_json(
            json.loads(storage[key].decode("utf-8"))
        )
        version_info = _merge_version_info(old_version_info, new_version_info)
    except KeyError:
        try:
            old_version_info = pickle.loads(
                storage[get_version_control_info_key_old()]
            )  # backward compatiblity
            version_info = _merge_version_info(old_version_info, new_version_info)
        except KeyError:
            version_info = new_version_info
    storage[key] = json.dumps(_version_info_to_json(version_info)).encode("utf-8")
    lock.release()


def load_version_info(storage: LRUCache) -> Dict:
    try:
        return _version_info_from_json(
            json.loads(storage[get_version_control_info_key()].decode("utf-8"))
        )
    except KeyError:
        return pickle.loads(
            storage[get_version_control_info_key_old()]
        )  # backward compatiblity


def auto_checkout(dataset) -> bool:
    """Automatically checks out if current node is not the head node of the branch. This may happen either during commit/setitem/append/extend/create_tensor/delete_tensor/info updates."""
    version_state = dataset.version_state
    if not version_state["commit_node"].is_head_node:
        current_branch = version_state["branch"]
        auto_branch = f"auto_{generate_hash()}"
        logger.info(
            f"Automatically checking out to branch '{auto_branch}' as not currently at the head node of branch '{current_branch}'."
        )
        checkout(dataset, auto_branch, True)
        return True
    return False


def auto_commit(dataset, message: str) -> None:
    """Automatically commits to the current branch before a checkout to a newly created branch if the current node is the head node and has changes."""
    version_state = dataset.version_state
    commit_node = version_state["commit_node"]
    head = commit_node.is_head_node
    if not head:
        return

    if not current_commit_has_change(version_state, dataset.storage):
        parent_id = commit_node.parent.commit_id  # type: ignore
        checkout(dataset, parent_id, False)
        return

    original_commit_id = version_state["commit_id"]
    branch = version_state["branch"]
    logger.info(
        f"Auto commiting to branch '{branch}' before checkout as currently at head node."
    )
    commit(dataset, message)
    checkout(dataset, original_commit_id, False)


def current_commit_has_change(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    return (
        current_commit_has_data(version_state, storage)
        or current_commit_has_info_modified(version_state, storage)
        or version_state["commit_id"] == FIRST_COMMIT_ID
    )


def current_commit_has_data(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    """Checks if the current commit has any data present in it or not."""
    commit_id = version_state["commit_id"]
    try:
        dataset_diff_key = get_dataset_diff_key(commit_id)
        dataset_diff = storage.get_hub_object(dataset_diff_key, DatasetDiff)
        if dataset_diff.deleted or dataset_diff.renamed:
            return True
    except KeyError:
        pass

    for tensor in version_state["full_tensors"].keys():
        if commit_id == FIRST_COMMIT_ID:
            # if the first commit has even a single tensor i.e. it entered the for loop, it has data
            return True
        if commit_diff_exists(version_state, storage, tensor):
            # commit diff is created during tensor creation and append/extend/update
            return True
    return False


def current_commit_has_info_modified(
    version_state: Dict[str, Any], storage: LRUCache
) -> bool:
    commit_id = version_state["commit_id"]
    try:
        dataset_diff_key = get_dataset_diff_key(commit_id)
        dataset_diff = storage.get_hub_object(dataset_diff_key, DatasetDiff)
        if dataset_diff.info_updated:
            return True
    except KeyError:
        pass

    for tensor in version_state["full_tensors"].keys():
        try:
            tensor_diff_key = get_tensor_commit_diff_key(tensor, commit_id)
            tensor_diff = storage.get_hub_object(tensor_diff_key, CommitDiff)
            if tensor_diff.info_updated:
                return True
        except KeyError:
            pass

    return False


def commit_diff_exists(
    version_state: Dict[str, Any], storage: LRUCache, tensor: str
) -> bool:
    """Checks if the commit chunk set exists for the given tensor in the current commit."""
    try:
        commit_id = version_state["commit_id"]
        key = get_tensor_commit_diff_key(tensor, commit_id)
        storage[key]
        return True
    except KeyError:
        return False


def load_meta(dataset):
    """Loads the meta info for the version state."""
    from hub.core.tensor import Tensor

    version_state = dataset.version_state
    storage: LRUCache = dataset.storage
    storage.clear_hub_objects()
    meta_key = get_dataset_meta_key(version_state["commit_id"])
    meta = storage.get_hub_object(meta_key, DatasetMeta)
    if not meta.tensor_names:  # backward compatibility
        meta.tensor_names = {key: key for key in meta.tensors}

    ffw_dataset_meta(meta)
    version_state["meta"] = meta

    storage.register_hub_object(meta_key, meta)
    _tensors = version_state["full_tensors"]
    _tensors.clear()
    _tensor_names = version_state["tensor_names"]
    _tensor_names.clear()
    _tensor_names.update(meta.tensor_names)

    for tensor_key in _tensor_names.values():
        _tensors[tensor_key] = Tensor(tensor_key, dataset)


def warn_node_checkout(commit_node: CommitNode, create: bool):
    """Throws a warning if there are no commits in a branch after checkout.
    This warning isn't thrown if the branch was newly created.
    """
    if not create and commit_node.is_head_node:
        branch = commit_node.branch
        parent = commit_node.parent
        if parent is None or parent.branch != branch:
            warnings.warn(
                f"The branch ({branch}) that you have checked out to, has no commits."
            )


def convert_to_bytes(inp):
    return inp.tobytes() if isinstance(inp, HubMemoryObject) else inp
