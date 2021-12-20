import random
import time
import hashlib
import pickle
from typing import Any, Dict
import warnings
from hub.client.log import logger
from hub.constants import FIRST_COMMIT_ID
from hub.core.fast_forwarding import ffw_dataset_meta
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.storage.cachable import Cachable
from hub.core.version_control.commit_node import CommitNode  # type: ignore
from hub.core.version_control.commit_chunk_set import CommitChunkSet  # type: ignore
from hub.core.storage import LRUCache
from hub.core.lock import Lock
from hub.util.exceptions import CallbackInitializationError, CheckoutError
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_info_key,
    get_tensor_meta_key,
    get_tensor_commit_chunk_set_key,
    get_tensor_tile_encoder_key,
    get_version_control_info_key,
    get_version_control_info_lock_key,
)
from hub.util.remove_cache import get_base_storage


def generate_hash() -> str:
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    hsh.update(random.randrange(0, 1000000).to_bytes(4, "big"))
    return hsh.hexdigest()


def commit(
    version_state: Dict[str, Any], storage: LRUCache, message: str = None
) -> None:
    """Modifies the version state to reflect the commit and also copies required data to the new commit directory."""
    storage.check_readonly()
    # if not the head node, checkout to an auto branch that is newly created
    auto_checkout(version_state, storage)
    stored_commit_id = version_state["commit_id"]
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
    discard_old_metas(stored_commit_id, storage, version_state["full_tensors"])
    load_meta(storage, version_state)


def checkout(
    version_state: Dict[str, Any],
    storage: LRUCache,
    address: str,
    create: bool = False,
) -> None:
    """Modifies the version state to reflect the checkout and also copies required data to the new branch directory if a new one is being created."""
    original_commit_id = version_state["commit_id"]

    if address in version_state["branch_commit_map"].keys():
        if create:
            raise CheckoutError(f"Can't create new branch, '{address}' already exists.")
        version_state["branch"] = address
        new_commit_id = version_state["branch_commit_map"][address]
        if original_commit_id == new_commit_id:
            return
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = version_state["commit_node_map"][new_commit_id]
        if not storage.read_only:
            storage.flush()
    elif address in version_state["commit_node_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create new branch, commit '{address}' already exists."
            )
        if address == original_commit_id:
            return
        version_state["commit_id"] = address
        version_state["commit_node"] = version_state["commit_node_map"][address]
        version_state["branch"] = version_state["commit_node"].branch
        if not storage.read_only:
            storage.flush()
    elif create:
        storage.check_readonly()
        # if the original commit is head of the branch, auto commit and checkout to original commit before creating new branch
        auto_commit(version_state, storage, address)
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
    else:
        raise CheckoutError(
            f"Address {address} not found. If you want to create a new branch, use checkout with create=True"
        )

    discard_old_metas(
        original_commit_id,
        storage,
        version_state["full_tensors"],
    )
    load_meta(storage, version_state)


def copy_metas(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
    version_state: Dict[str, Any],
) -> None:
    """Copies meta data from one commit to another."""
    tensors = version_state["full_tensors"]
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    src_dataset_meta = storage[src_dataset_meta_key]
    if isinstance(src_dataset_meta, Cachable):
        storage[dest_dataset_meta_key] = src_dataset_meta.copy()
    else:
        storage[dest_dataset_meta_key] = src_dataset_meta

    try:
        src_dataset_info_key = get_dataset_info_key(src_commit_id)
        dest_dataset_info_key = get_dataset_info_key(dest_commit_id)
        src_dataset_info = storage[src_dataset_info_key]
        if isinstance(src_dataset_info, Cachable):
            new_info = src_dataset_info.copy()
            new_info.initialize_callback_location(
                dest_dataset_info_key, storage, version_state
            )
            storage[dest_dataset_info_key] = new_info
        else:
            storage[dest_dataset_info_key] = src_dataset_info
    except KeyError:
        pass

    tensor_list = list(tensors.keys())

    for tensor in tensor_list:
        src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
        dest_tensor_meta_key = get_tensor_meta_key(tensor, dest_commit_id)
        src_tensor_meta = storage[src_tensor_meta_key]
        if isinstance(src_tensor_meta, Cachable):
            storage[dest_tensor_meta_key] = src_tensor_meta.copy()
        else:
            storage[dest_tensor_meta_key] = src_tensor_meta

        try:
            src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
            dest_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, dest_commit_id)
            src_chunk_id_encoder = storage[src_chunk_id_encoder_key]
            if isinstance(src_chunk_id_encoder, Cachable):
                storage[dest_chunk_id_encoder_key] = src_chunk_id_encoder.copy()
            else:
                storage[dest_chunk_id_encoder_key] = src_chunk_id_encoder
        except KeyError:
            pass

        try:
            src_tile_encoder_key = get_tensor_tile_encoder_key(tensor, src_commit_id)
            dest_tile_encoder_key = get_tensor_tile_encoder_key(tensor, dest_commit_id)
            src_tile_encoder = storage[src_tile_encoder_key]
            if isinstance(src_tile_encoder, Cachable):
                storage[dest_tile_encoder_key] = src_tile_encoder.copy()
            else:
                storage[dest_tile_encoder_key] = src_tile_encoder
        except KeyError:
            pass

        try:
            src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
            dest_tensor_info_key = get_tensor_info_key(tensor, dest_commit_id)
            src_tensor_info = storage[src_tensor_info_key]
            if isinstance(src_tensor_info, Cachable):
                new_info = src_tensor_info.copy()
                new_info.initialize_callback_location(
                    dest_tensor_info_key, storage, version_state
                )
                storage[dest_tensor_info_key] = new_info
            else:
                storage[dest_tensor_info_key] = src_tensor_info
        except KeyError:
            pass

    storage.flush()


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
        storage.dirty_keys.discard(key)
        if key in storage.lru_sizes:
            size = storage.lru_sizes.pop(key)
            storage.cache_used -= size
        try:
            del storage.cache_storage[key]
        except (KeyError, CallbackInitializationError):
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
        old_version_info = pickle.loads(storage[key])
        version_info = _merge_version_info(old_version_info, new_version_info)
    except KeyError:
        version_info = new_version_info
    storage[key] = pickle.dumps(version_info)
    lock.release()


def auto_checkout(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Automatically checks out if current node is not the head node of the branch. This may happen either during commit/setitem/append/extend/create_tensor/delete_tensor/info updates."""
    if not version_state["commit_node"].is_head_node:
        current_branch = version_state["branch"]
        auto_branch = f"auto_{generate_hash()}"
        logger.info(
            f"Automatically checking out to branch '{auto_branch}' as not currently at the head node of branch '{current_branch}'."
        )
        checkout(version_state, storage, auto_branch, True)


def auto_commit(version_state: Dict[str, Any], storage: LRUCache, address: str) -> None:
    """Automatically commits to the current branch before a checkout to a newly created branch if the current node is the head node."""
    commit_node = version_state["commit_node"]
    if commit_node.is_head_node:
        original_commit_id = version_state["commit_id"]
        branch = version_state["branch"]
        logger.info(
            f"Auto commiting to branch '{branch}' before checkout as currently at head node."
        )
        commit(
            version_state,
            storage,
            f"auto commit before checkout to {address}",
        )
        checkout(version_state, storage, original_commit_id, False)


def commit_has_data(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    """Checks if the current commit has any data present in it or not."""
    commit_id = version_state["commit_id"]
    for tensor in version_state["full_tensors"].keys():
        if commit_id == FIRST_COMMIT_ID:
            # if the first commit has even a single tensor i.e. it entered the for loop, it has data
            return True
        key = get_tensor_commit_chunk_set_key(tensor, commit_id)
        if commit_chunk_set_exists(version_state, storage, tensor):
            enc = storage.get_cachable(key, CommitChunkSet)
            if enc.chunks:
                return True
    return False


def commit_chunk_set_exists(
    version_state: Dict[str, Any], storage: LRUCache, tensor: str
) -> bool:
    """Checks if the commit chunk set exists for the given tensor in the current commit."""
    try:
        commit_id = version_state["commit_id"]
        key = get_tensor_commit_chunk_set_key(tensor, commit_id)
        storage[key]
        return True
    except KeyError:
        return False


def load_meta(storage, version_state):
    """Loads the meta info for the version state."""
    from hub.core.tensor import Tensor

    meta_key = get_dataset_meta_key(version_state["commit_id"])
    meta = storage.get_cachable(meta_key, DatasetMeta)
    ffw_dataset_meta(meta)
    version_state["meta"] = meta
    _tensors = version_state["full_tensors"]
    _tensors.clear()

    for tensor_name in meta.tensors:
        _tensors[tensor_name] = Tensor(tensor_name, storage, version_state)


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
