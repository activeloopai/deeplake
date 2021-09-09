import time
import hashlib
import pickle
from typing import Any, Dict

from hub.core.version_control.version_node import VersionNode
from hub.core.storage import StorageProvider
from hub.util.exceptions import CheckoutError
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_info_key,
    get_tensor_meta_key,
    get_version_control_info_key,
)


def generate_hash() -> str:
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    return hsh.hexdigest()


def commit(
    version_state: Dict[str, Any], storage: StorageProvider, message: str = None
) -> None:

    # if not the head node, checkout to an auto branch that is newly created
    if version_state["commit_node"].children:
        checkout(version_state, storage, f"auto_branch_{generate_hash()}", True)
        commit(version_state, storage, message)
        return
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


def checkout(
    version_state: Dict[str, Any],
    storage: StorageProvider,
    address: str,
    create: bool = False,
) -> None:
    if address in version_state["branch_commit_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create a new branch. Branch '{address}' already exists."
            )
        version_state["branch"] = address
        version_state["commit_id"] = version_state["branch_commit_map"][address]
        version_state["commit_node"] = version_state["commit_node_map"][
            version_state["commit_id"]
        ]
    elif address in version_state["commit_node_map"].keys():
        if create:
            raise CheckoutError(
                f"Can't create a new branch. Commit '{address}' already exists."
            )
        version_state["commit_id"] = address
        version_state["commit_node"] = version_state["commit_node_map"][address]
        version_state["branch"] = version_state["commit_node"].branch
    elif create:
        # if the commit is head of the branch and has no children, create a new commit and checkout from the previous commit to this one
        if not version_state["commit_node"].children:
            original_commit_id = version_state["commit_id"]
            commit(
                version_state,
                storage,
                f"auto commit before checkout to {address}",
            )
            checkout(version_state, storage, original_commit_id, False)
        original_commit_id = version_state["commit_id"]
        version_state["branch"] = address
        new_commit_id = generate_hash()
        new_node = VersionNode(version_state["branch"], new_commit_id)
        version_state["commit_node"] = new_node
        version_state["commit_id"] = new_commit_id
        version_state["commit_node_map"][version_state["commit_id"]] = new_node
        version_state["branch_commit_map"][version_state["branch"]] = version_state[
            "commit_id"
        ]
        save_version_info(version_state, storage)
        copy_metas(
            original_commit_id,
            version_state["commit_id"],
            storage,
            version_state["tensors"],
        )
    else:
        raise CheckoutError(
            f"Address {address} not found. If you want to create a new branch, use checkout with create=True"
        )


def copy_metas(
    src_commit_id: str, dest_commit_id: str, storage: StorageProvider, tensors: Dict
) -> None:
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    storage[dest_dataset_meta_key] = storage[src_dataset_meta_key].copy()

    try:
        src_dataset_info_key = get_dataset_info_key(src_commit_id)
        dest_dataset_info_key = get_dataset_info_key(dest_commit_id)
        storage[dest_dataset_info_key] = storage[src_dataset_info_key].copy()
    except Exception:
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
        except Exception:
            pass

        try:
            src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
            dest_tensor_info_key = get_tensor_info_key(tensor, dest_commit_id)
            storage[dest_tensor_info_key] = storage[src_tensor_info_key].copy()
        except Exception:
            pass

    storage.flush()


def save_version_info(version_state: Dict[str, Any], storage: StorageProvider) -> None:
    version_info = {
        "commit_node_map": version_state["commit_node_map"],
        "branch_commit_map": version_state["branch_commit_map"],
    }
    storage[get_version_control_info_key()] = pickle.dumps(version_info)
