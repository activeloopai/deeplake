from hub.core.meta.dataset_meta import DatasetMeta
import random
import time
import hashlib
import pickle
from typing import Any, Dict

from hub.core.version_control.version_node import VersionNode  # type: ignore
from hub.core.version_control.version_chunk_list import VersionChunkList  # type: ignore
from hub.core.storage import LRUCache
from hub.util.exceptions import CallbackInitializationError, CheckoutError
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_info_key,
    get_tensor_meta_key,
    get_tensor_version_chunk_list_key,
    get_version_control_info_key,
)

# TODO: replace prints with logging
def generate_hash() -> str:
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    hsh.update(random.randrange(0, 1000000).to_bytes(4, "big"))
    return hsh.hexdigest()


def commit(
    version_state: Dict[str, Any], storage: LRUCache, message: str = None
) -> None:
    """Modifies the version state to reflect the commit and also copies required data to the new commit directory."""

    # if not the head node, checkout to an auto branch that is newly created
    auto_checkout(version_state, storage)
    stored_commit_id = version_state["commit_id"]
    version_state["commit_id"] = generate_hash()
    new_node = VersionNode(version_state["branch"], version_state["commit_id"])
    version_state["commit_node"].add_successor(new_node, message)
    version_state["commit_node"] = new_node
    version_state["branch_commit_map"][version_state["branch"]] = version_state[
        "commit_id"
    ]
    version_state["commit_node_map"][version_state["commit_id"]] = new_node
    save_version_info(version_state, storage)
    copy_metas(
        stored_commit_id, version_state["commit_id"], storage, version_state["tensors"]
    )
    storage.flush()
    load_meta(storage, version_state)


def checkout(
    version_state: Dict[str, Any],
    storage: LRUCache,
    address: str,
    create: bool = False,
) -> None:
    """Modifies the version state to reflect the checkout and also copies required data to the new branch directory if a new one is being created."""
    if address in version_state["branch_commit_map"].keys():
        if create:
            raise CheckoutError(f"Can't create new branch, '{address}' already exists.")
        version_state["branch"] = address
        version_state["commit_id"] = version_state["branch_commit_map"][address]
        version_state["commit_node"] = version_state["commit_node_map"][
            version_state["commit_id"]
        ]
    elif address in version_state["commit_node_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create new branch, commit '{address}' already exists."
            )
        version_state["commit_id"] = address
        version_state["commit_node"] = version_state["commit_node_map"][address]
        version_state["branch"] = version_state["commit_node"].branch
    elif create:
        # if the original commit is head of the branch and has data, auto commit and checkout to original commit before creating new branch
        auto_commit(version_state, storage, address)
        original_commit_id = version_state["commit_id"]
        new_commit_id = generate_hash()
        new_node = VersionNode(address, new_commit_id)
        new_node.parent = version_state["commit_node"]
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = new_node
        version_state["branch"] = address
        version_state["commit_node_map"][new_commit_id] = new_node
        version_state["branch_commit_map"][address] = new_commit_id
        save_version_info(version_state, storage)
        copy_metas(
            original_commit_id,
            new_commit_id,
            storage,
            version_state["tensors"],
        )
        storage.flush()
    else:
        raise CheckoutError(
            f"Address {address} not found. If you want to create a new branch, use checkout with create=True"
        )
    load_meta(storage, version_state)


def copy_metas(
    src_commit_id: str, dest_commit_id: str, storage: LRUCache, tensors: Dict
) -> None:
    """Copies meta data from one commit to another."""
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    storage[dest_dataset_meta_key] = storage[src_dataset_meta_key].copy()

    try:
        src_dataset_info_key = get_dataset_info_key(src_commit_id)
        dest_dataset_info_key = get_dataset_info_key(dest_commit_id)
        storage[dest_dataset_info_key] = storage[src_dataset_info_key].copy()
    except (KeyError, CallbackInitializationError):
        pass

    tensor_list = list(tensors.keys())

    for tensor in tensor_list:
        src_tensor_meta_key = get_tensor_meta_key(tensor, src_commit_id)
        dest_tensor_meta_key = get_tensor_meta_key(tensor, dest_commit_id)
        storage[dest_tensor_meta_key] = storage[src_tensor_meta_key].copy()

        try:
            src_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, src_commit_id)
            dest_chunk_id_encoder_key = get_chunk_id_encoder_key(tensor, dest_commit_id)
            storage[dest_chunk_id_encoder_key] = storage[
                src_chunk_id_encoder_key
            ].copy()
        except (KeyError, CallbackInitializationError):
            pass

        try:
            src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
            dest_tensor_info_key = get_tensor_info_key(tensor, dest_commit_id)
            storage[dest_tensor_info_key] = storage[src_tensor_info_key].copy()
        except (KeyError, CallbackInitializationError):
            pass


def save_version_info(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Saves the current version info to the storage."""
    version_info = {
        "commit_node_map": version_state["commit_node_map"],
        "branch_commit_map": version_state["branch_commit_map"],
    }
    storage[get_version_control_info_key()] = pickle.dumps(version_info)


def auto_checkout(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Automatically checks out if current node is not the head node of the branch. This may happen either during commit or during setitem."""
    if version_state["commit_node"].children:
        current_branch = version_state["branch"]
        auto_branch = f"auto_{generate_hash()}"
        print(
            f"Automatically checking out to branch '{auto_branch}' as not currently at the head node of branch '{current_branch}'."
        )
        checkout(version_state, storage, auto_branch, True)


def auto_commit(version_state: Dict[str, Any], storage: LRUCache, address: str) -> None:
    """Automatically commits to the current branch before a checkout to a new branch if the current node is the head node and has uncommitted data."""
    commit_node = version_state["commit_node"]
    if not commit_node.children and commit_has_data(version_state, storage):
        original_commit_id = version_state["commit_id"]
        branch = version_state["branch"]
        print(
            f"Auto commiting to branch '{branch}' as currently at head node with uncommitted changes."
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
    for tensor in version_state["tensors"].keys():
        key = get_tensor_version_chunk_list_key(tensor, commit_id)
        if version_chunk_list_exists(version_state, storage, tensor):
            enc = storage.get_cachable(key, VersionChunkList)
            if enc.chunks:
                return True
    return False


def version_chunk_list_exists(
    version_state: Dict[str, Any], storage: LRUCache, tensor: str
) -> bool:
    """Checks if the version chunk list exists for the given tensor in the current commit."""
    try:
        commit_id = version_state["commit_id"]
        key = get_tensor_version_chunk_list_key(tensor, commit_id)
        storage[key]
        return True
    except KeyError:
        return False


def load_meta(storage, version_state):
    """Loads the meta info for the version state."""
    from hub.core.tensor import Tensor

    meta_key = get_dataset_meta_key(version_state["commit_id"])
    meta = storage.get_cachable(meta_key, DatasetMeta)
    version_state["meta"] = meta
    tensors = version_state["tensors"]
    tensors.clear()

    for tensor_name in meta.tensors:
        tensors[tensor_name] = Tensor(tensor_name, storage, version_state)
