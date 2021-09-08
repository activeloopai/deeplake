import hashlib
from hub.core.storage.cachable import Cachable
from hub.util.keys import (
    get_chunk_id_encoder_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_info_key,
    get_tensor_meta_key,
)
from hub.constants import VERSION_CONTROL_FILE
import pickle
from hub.core.version_control.version_node import VersionNode
import time


def generate_hash():
    hsh = hashlib.sha1()
    hsh.update(str(time.time()).encode("utf-8"))
    return hsh.hexdigest()


def commit(version_state, storage, message: str = None) -> None:
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


def checkout(version_state, storage, address: str, create: bool = False) -> None:
    if address in version_state["branch_commit_map"].keys():
        if create:
            raise Exception("Branch already exists")  # TODO: better exception
        version_state["branch"] = address
        version_state["commit_id"] = version_state["branch_commit_map"][address]
        version_state["commit_node"] = version_state["commit_node_map"][
            version_state["commit_id"]
        ]
    elif address in version_state["commit_node_map"].keys():
        if create:
            raise Exception("Commit already exists")  # TODO: better exception
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
        # raise AddressNotFound(address)
        raise Exception  # TODO: better exception


def copy_metas(src_commit_id: str, dest_commit_id: str, storage, tensors):
    src_dataset_meta_key = get_dataset_meta_key(src_commit_id)
    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
    storage[dest_dataset_meta_key] = storage[src_dataset_meta_key].copy()

    try:
        src_dataset_info_key = get_dataset_info_key(src_commit_id)
        dest_dataset_info_key = get_dataset_info_key(dest_commit_id)
        storage[dest_dataset_info_key] = storage[src_dataset_info_key].copy()
    except:
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
        except:
            pass

        try:
            src_tensor_info_key = get_tensor_info_key(tensor, src_commit_id)
            dest_tensor_info_key = get_tensor_info_key(tensor, dest_commit_id)
            storage[dest_tensor_info_key] = storage[src_tensor_info_key].copy()
        except:
            pass

    storage.flush()


def save_version_info(version_state, storage):
    version_info = {
        "commit_node_map": version_state["commit_node_map"],
        "branch_commit_map": version_state["branch_commit_map"],
    }
    storage[VERSION_CONTROL_FILE] = pickle.dumps(version_info)
