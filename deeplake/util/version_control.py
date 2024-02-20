import random
import time
import hashlib
import pickle
from typing import Any, Dict, Optional, List
import warnings

from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder

from deeplake.client.log import logger
from deeplake.constants import FIRST_COMMIT_ID
from deeplake.core import lock, StorageProvider
from deeplake.core.fast_forwarding import ffw_dataset_meta
from deeplake.core.meta.dataset_meta import DatasetMeta
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from deeplake.core.storage.lru_cache import LRUCache
from deeplake.core.storage.memory import MemoryProvider
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.version_control.dataset_diff import DatasetDiff
from deeplake.core.version_control.commit_node import CommitNode  # type: ignore
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap  # type: ignore
from deeplake.core.storage import LRUCache
from deeplake.core.lock import Lock, PersistentLock
from deeplake.util.exceptions import (
    CheckoutError,
    CommitError,
    DatasetCorruptError,
    VersionControlError,
)
from deeplake.util.keys import (
    get_chunk_id_encoder_key,
    get_creds_encoder_key,
    get_sequence_encoder_key,
    get_dataset_diff_key,
    get_dataset_info_key,
    get_dataset_meta_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_tensor_info_key,
    get_tensor_meta_key,
    get_tensor_tile_encoder_key,
    get_version_control_info_key,
    get_version_control_info_key_old,
    get_version_control_info_lock_key,
    get_commit_info_key,
    get_pad_encoder_key,
)
from deeplake.constants import COMMIT_INFO_FILENAME
from deeplake.util.path import relpath
from deeplake.util.remove_cache import get_base_storage
from deeplake.hooks import dataset_committed
from datetime import datetime
import deeplake.core.dataset

import posixpath
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
            "is_checkpoint": node.is_checkpoint,
            "total_samples_processed": node.total_samples_processed,
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
        branch_commit_map[
            commit_data["branch"]
        ]  # we will rebuild version info if this fails
        node = CommitNode(commit_data["branch"], commit_id)
        node.commit_message = commit_data["commit_message"]
        commit_time = commit_data["commit_time"]
        node.commit_time = (
            None if commit_time is None else datetime.fromtimestamp(commit_time)
        )
        node.commit_user_name = commit_data["commit_user_name"]
        node.is_checkpoint = commit_data.get("is_checkpoint", False)
        node.total_samples_processed = commit_data.get("total_samples_processed", 0)
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


def integrity_check(dataset):
    try:
        rev_tensor_names = {v: k for k, v in dataset.meta.tensor_names.items()}
        for k, t in dataset._tensors(include_disabled=False).items():
            n1 = t.meta.length
            engine = t.chunk_engine
            n2 = engine.chunk_id_encoder.num_samples
            if n1 != n2:
                raise ValueError(
                    f"Tensor meta and chunk id encoder have different number of samples ({n1} and {n2} respectively) for tensor {k}."
                )
            num_sequences = getattr(engine.sequence_encoder, "num_samples", None)
            for l, info in t.meta.links.items():
                l = rev_tensor_names[l]
                l = relpath(l, dataset.group_index)
                if num_sequences is not None and not info["flatten_sequence"]:
                    n2 = num_sequences
                else:
                    n2 = n1
                n3 = dataset[l].meta.length
                if n2 != n3:
                    raise ValueError(
                        f"Tensor {k} and its linked tensor {l} have different number of samples ({n2} and {n3} respectively)."
                    )
            engine.tile_encoder

            engine.creds_encoder
    except Exception as e:
        raise DatasetCorruptError(
            f"The HEAD node of the branch {dataset.branch} of this dataset is in a corrupted state and is likely not recoverable.",
            "Please run `ds.reset()` to revert the uncommitted changes in order to continue making updates on this branch.",
        ) from e


def commit(
    dataset,
    message: Optional[str] = None,
    hash: Optional[str] = None,
    flush_version_control_info: bool = True,
    reload_meta: bool = True,
    is_checkpoint: bool = False,
    total_samples_processed: int = 0,
) -> None:
    """Modifies the version state to reflect the commit and also copies required data to the new commit directory."""
    storage = dataset.storage
    version_state = dataset.version_state
    storage.check_readonly()
    integrity_check(dataset)
    # if not the head node, checkout to an auto branch that is newly created
    auto_checkout(dataset, flush_version_control_info=False)
    stored_commit_node: CommitNode = version_state["commit_node"]
    stored_commit_id = version_state["commit_id"]
    if hash:
        if hash in version_state["commit_node_map"]:
            raise CommitError(f"Commit {hash} already exists")
    else:
        hash = generate_hash()
    version_state["commit_id"] = hash
    new_node = CommitNode(version_state["branch"], hash)
    stored_commit_node.add_successor(new_node, dataset.username, message)
    stored_commit_node.is_checkpoint = is_checkpoint
    stored_commit_node.total_samples_processed = total_samples_processed
    version_state["commit_node"] = new_node
    version_state["branch_commit_map"][version_state["branch"]] = version_state[
        "commit_id"
    ]
    version_state["commit_node_map"][hash] = new_node
    copy_metas(stored_commit_id, hash, storage)
    create_commit_chunk_maps(stored_commit_id, hash, storage)
    discard_old_metas(stored_commit_id, storage, version_state["full_tensors"])
    if reload_meta:
        load_meta(dataset)

    commit_time = stored_commit_node.commit_time
    commit_message = stored_commit_node.commit_message
    author = stored_commit_node.commit_user_name
    if flush_version_control_info:
        save_version_info(version_state, storage)
        save_commit_info(stored_commit_node, storage)
        save_commit_info(new_node, storage)
    else:
        stored_commit_node._info_updated = True
        new_node._info_updated = True
    dataset._send_commit_event(
        commit_message=commit_message, commit_time=commit_time, author=author
    )
    dataset_committed(dataset)


def checkout(
    dataset,
    address: str,
    create: bool = False,
    hash: Optional[str] = None,
    flush_version_control_info=True,
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
        auto_commit(
            dataset,
            f"auto commit before checkout to {address}",
            flush_version_control_info=False,
        )
        if hash:
            if hash in version_state["commit_node_map"]:
                raise CommitError(f"Commit {hash} already exists")
            new_commit_id = hash
        else:
            new_commit_id = generate_hash()
        new_node = CommitNode(address, new_commit_id)
        stored_commit_node = version_state["commit_node"]
        stored_commit_node.add_child(new_node)
        version_state["commit_id"] = new_commit_id
        version_state["commit_node"] = new_node
        version_state["branch"] = address
        version_state["commit_node_map"][new_commit_id] = new_node
        version_state["branch_commit_map"][address] = new_commit_id
        if flush_version_control_info:
            save_version_info(version_state, storage)
            save_commit_info(new_node, storage)
            save_commit_info(stored_commit_node, storage)
        else:
            stored_commit_node._info_updated = True
            new_node._info_updated = True
        copy_metas(original_commit_id, new_commit_id, storage)
        create_commit_chunk_maps(original_commit_id, new_commit_id, storage)
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

    try:
        load_meta(dataset)
    except Exception as e:
        checkout(dataset, original_commit_id)
        raise CheckoutError(
            f"Unable to checkout to '{address}', failed to load meta data."
        ) from e


def _squash_main(
    dataset,
) -> None:
    """
    Combines all commits in the main branch into a single commit.
    """
    storage = dataset.storage
    storage.check_readonly()

    version_state = dataset.version_state

    if len(dataset.branches) > 1:
        raise VersionControlError(
            f"Cannot squash commits if there are multiple branches"
        )
    if len(dataset.get_views()) > 0:
        raise VersionControlError(f"Cannot squash commits if there are views present")

    try:
        base_storage = get_base_storage(storage)
        versioncontrol_lock = PersistentLock(
            base_storage, get_version_control_info_lock_key()
        )
        versioncontrol_lock.acquire()  # Blocking

        dataset_lock = lock.lock_dataset(dataset, dataset.branches[0])

        for tensor in dataset._tensors(
            include_hidden=True, include_disabled=True
        ).values():
            chunk_engine = tensor.chunk_engine
            for chunk_id in [row[0] for row in chunk_engine.chunk_id_encoder._encoded]:
                chunk = chunk_engine.get_chunk_from_chunk_id(chunk_id)
                if chunk.key.startswith("versions"):
                    base_storage[
                        "/".join(
                            [
                                tensor.key,
                                "chunks",
                                ChunkIdEncoder.name_from_id(chunk_id),
                            ]
                        )
                    ] = chunk.tobytes()

            for key_fn in [
                get_tensor_info_key,
                get_tensor_meta_key,
                get_creds_encoder_key,
                get_chunk_id_encoder_key,
                get_pad_encoder_key,
                get_sequence_encoder_key,
                get_tensor_tile_encoder_key,
            ]:
                try:
                    data_bytes = storage.get_bytes(
                        key_fn(chunk_engine.key, dataset.pending_commit_id)
                    )
                except KeyError:
                    continue

                base_storage[key_fn(chunk_engine.key, FIRST_COMMIT_ID)] = data_bytes

        commits_to_delete = [
            commit_id
            for commit_id in version_state["commit_node_map"].keys()
            if commit_id != FIRST_COMMIT_ID
        ]

        dataset.version_state["commit_node_map"] = {
            FIRST_COMMIT_ID: dataset.version_state["commit_node_map"][FIRST_COMMIT_ID],
        }
        dataset.version_state["commit_node_map"][FIRST_COMMIT_ID].children = []
        dataset.version_state["commit_node_map"][FIRST_COMMIT_ID].commit_message = None
        dataset.version_state["commit_node_map"][FIRST_COMMIT_ID].commit_time = None
        dataset.version_state["commit_node_map"][
            FIRST_COMMIT_ID
        ].commit_user_name = None

        dataset.version_state["branch_commit_map"]["main"] = FIRST_COMMIT_ID
        dataset.version_state["commit_id"] = FIRST_COMMIT_ID
        dataset.version_state["commit_node"] = dataset.version_state["commit_node_map"][
            FIRST_COMMIT_ID
        ]

        base_storage[get_version_control_info_key()] = json.dumps(
            _version_info_to_json(
                {
                    "commit_node_map": version_state["commit_node_map"],
                    "branch_commit_map": version_state["branch_commit_map"],
                }
            )
        ).encode("utf-8")

        for commit_to_delete in commits_to_delete:
            delete_version_from_storage(dataset.storage, commit_to_delete)

        dataset._reload_version_state()

        dataset.commit("Squashed commits")

    finally:
        versioncontrol_lock.release()
        dataset_lock and dataset_lock.release()
    #
    # dataset._send_branch_deletion_event(branch_name)


def delete_branch(
    dataset,
    branch_name: str,
) -> None:
    """
    Deletes the branch and cleans up any unneeded data.
    Branches can only be deleted if there are no sub-branches and if it has never been merged into another branch.
    """

    storage = dataset.storage
    storage.check_readonly()

    # storage = dataset.storage
    version_state = dataset.version_state
    if version_state["branch"] == branch_name:
        raise VersionControlError(
            f"Cannot delete the currently checked out branch: {branch_name}"
        )

    if branch_name == "main":
        raise VersionControlError("Cannot delete the main branch")

    if branch_name not in version_state["branch_commit_map"].keys():
        raise VersionControlError(f"Branch {branch_name} does not exist")

    storage = get_base_storage(storage)
    versioncontrol_lock = PersistentLock(storage, get_version_control_info_lock_key())
    versioncontrol_lock.acquire()  # Blocking

    dataset_lock = lock.lock_dataset(
        dataset, version=dataset.version_state["branch_commit_map"][branch_name]
    )

    try:
        all_branch_commits = _find_branch_commits(branch_name, version_state)

        # Check that nothing points to any of the commits to delete
        for commit_id, commit_node in version_state["commit_node_map"].items():
            if commit_id in all_branch_commits:
                continue

            if commit_node.parent in all_branch_commits:
                raise VersionControlError(
                    f"Cannot delete branch {branch_name} because it has been previously merged"
                )

            for tensor in dataset.tensors:
                chunk_map_key = get_tensor_commit_chunk_map_key(tensor, commit_id)

                try:
                    found_map = dataset.storage.get_deeplake_object(
                        chunk_map_key, CommitChunkMap
                    )
                    if (
                        len(
                            [
                                1
                                for val in found_map.chunks.values()
                                if "commit_id" in val.keys()
                                and val["commit_id"] in all_branch_commits
                            ]
                        )
                        > 0
                    ):
                        raise VersionControlError(
                            f"Cannot delete branch {branch_name} because it has been previously merged into {commit_node.branch}"
                        )
                except KeyError:
                    pass  # no chunk map for this commit
                except FileNotFoundError:
                    pass  # no chunk map for this commit

        _delete_branch_and_commits(branch_name, all_branch_commits, dataset, storage)

    finally:
        versioncontrol_lock.release()
        dataset_lock and dataset_lock.release()

    dataset._send_branch_deletion_event(branch_name)


def _delete_branch_and_commits(
    branch_name: str, all_branch_commits: List[str], dataset, storage
) -> None:
    """
    Physically deletes the given branch and list of commits from the version_control_info.json and versions directories.
    Any validation on the information should have been performed before this method is called
    """
    version_state = dataset.version_state

    version_state["branch_commit_map"].pop(branch_name)
    for commit_id, commit_node in list(version_state["commit_node_map"].items()):
        if commit_id in all_branch_commits:
            version_state["commit_node_map"].pop(commit_id)
            continue

        commit_node.children = [
            child
            for child in commit_node.children
            if child.commit_id not in all_branch_commits
        ]
    for commit_id in all_branch_commits:
        delete_version_from_storage(dataset.storage, commit_id)

    storage[get_version_control_info_key()] = json.dumps(
        _version_info_to_json(
            {
                "commit_node_map": version_state["commit_node_map"],
                "branch_commit_map": version_state["branch_commit_map"],
            }
        )
    ).encode("utf-8")


def _find_branch_commits(branch_name: str, version_state: dict) -> List[str]:
    """
    Returns a list of all commits used by the given branch
    """
    all_branch_commits = []
    branch_commit = version_state["branch_commit_map"][branch_name]
    branch_commit_node = version_state["commit_node_map"][branch_commit]
    while branch_commit_node.branch == branch_name:
        all_branch_commits.append(branch_commit_node.commit_id)
        if (
            len(
                [
                    child
                    for child in branch_commit_node.children
                    if child.commit_id not in all_branch_commits
                ]
            )
            > 0
        ):
            raise VersionControlError(
                f"Cannot delete branch {branch_name} because it has sub-branches"
            )
        branch_commit_node = branch_commit_node.parent
    return all_branch_commits


def copy_metas(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
) -> None:
    """Copies meta data from one commit to another."""
    initial_autoflush = storage.autoflush
    storage.autoflush = False

    src_dataset_meta = _get_dataset_meta_at_commit(storage, src_commit_id)

    dest_dataset_meta_key = get_dataset_meta_key(dest_commit_id)
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

    tensor_list = src_dataset_meta.tensors

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


def create_commit_chunk_maps(
    src_commit_id: str,
    dest_commit_id: str,
    storage: LRUCache,
) -> None:
    """Creates commit chunk sets for all tensors in new commit."""
    tensor_list = _get_dataset_meta_at_commit(storage, src_commit_id).tensors
    for tensor in tensor_list:
        key = get_tensor_commit_chunk_map_key(tensor, dest_commit_id)
        storage[key] = CommitChunkMap()


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


def reset_and_checkout(ds, address, err, verbose=True):
    storage = ds.storage
    version_state = ds.version_state

    parent_commit_id, reset_commit_id = get_parent_and_reset_commit_ids(
        version_state, address
    )
    if parent_commit_id is False:
        # non-head node corrupted
        raise err
    if parent_commit_id is None:
        # no commits in the dataset
        storage.clear()
        ds._populate_meta()
        load_meta(ds)
        return

    ds.checkout(parent_commit_id)
    new_commit_id = replace_head(storage, version_state, reset_commit_id)
    ds.checkout(new_commit_id)

    current_node = version_state["commit_node_map"][ds.commit_id]
    if verbose:
        logger.info(f"HEAD reset. Current version:\n{current_node}")
    return ds.commit_id


def _merge_commit_node_maps(map1, map2):
    merged_map = {}

    commit_ids = [FIRST_COMMIT_ID]
    while commit_ids:
        commit_id = commit_ids.pop()
        if commit_id in map1 and commit_id in map2:
            node1 = map1[commit_id]
            node2 = map2[commit_id]
            merged_node = CommitNode(node1.branch, node2.commit_id)

            for attr in (
                "commit_message",
                "commit_user_name",
                "commit_time",
                "is_checkpoint",
                "total_samples_processed",
            ):
                setattr(merged_node, attr, getattr(node1, attr) or getattr(node2, attr))

            if node1.parent:
                assert node1.parent.commit_id == node2.parent.commit_id
                parent_id = node1.parent.commit_id
            else:
                parent_id = None

            commit_ids.extend(
                set(
                    [node.commit_id for node in node1.children]
                    + [node.commit_id for node in node2.children]
                )
            )
        else:
            if commit_id in map1:
                orig_node = map1[commit_id]
            else:
                orig_node = map2[commit_id]
            merged_node = orig_node.copy()

            if orig_node.parent:
                parent_id = orig_node.parent.commit_id
            else:
                parent_id = None

            commit_ids.extend([node.commit_id for node in orig_node.children])

        if parent_id:
            parent_node = merged_map[parent_id]
            parent_node.add_child(merged_node)
        merged_map[commit_id] = merged_node
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


def save_commit_info(commit_node: CommitNode, storage: LRUCache) -> None:
    """Saves the commit info to the storage."""
    storage = get_base_storage(storage)
    key = get_commit_info_key(commit_node.commit_id)
    storage[key] = json.dumps(commit_node.to_json()).encode("utf-8")
    commit_node._info_updated = False


def load_commit_info(commit_id: str, storage: LRUCache) -> Dict:
    """Loads the commit info from the storage."""
    storage = get_base_storage(storage)
    key = get_commit_info_key(commit_id)
    commit_info = json.loads(storage[key].decode("utf-8"))
    return commit_info


def save_version_info(version_state: Dict[str, Any], storage: LRUCache) -> None:
    """Saves the current version info to the storage."""
    storage = get_base_storage(storage)
    lock = Lock(storage, get_version_control_info_lock_key(), duration=10)
    lock.acquire()  # Blocking
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


def get_parent_and_reset_commit_ids(version_info, address):
    """Returns parent commit id and commit id which will be reset. Returns (False, False) if address is a non-HEAD commit id"""
    if address in version_info["branch_commit_map"]:
        commit_id = version_info["branch_commit_map"][address]
    elif address in version_info["commit_node_map"]:
        commit_id = address
    commit_node = version_info["commit_node_map"][commit_id]
    if not commit_node.is_head_node:
        return False, False
    parent_node = commit_node.parent
    if parent_node is None:
        previous_commit_id = None
    else:
        previous_commit_id = parent_node.commit_id
    return previous_commit_id, commit_id


def _create_new_head(
    storage: LRUCache, version_state, branch, parent_commit_id, new_commit_id
):
    # populate new commit folder
    copy_metas(parent_commit_id, new_commit_id, storage)
    create_commit_chunk_maps(parent_commit_id, new_commit_id, storage)

    # create new node
    parent_node: CommitNode = version_state["commit_node_map"][parent_commit_id]
    new_node = CommitNode(branch, new_commit_id)
    new_node.parent = parent_node
    version_state["branch_commit_map"][branch] = new_commit_id
    version_state["commit_node_map"][new_commit_id] = new_node

    return new_node


def _replace_head(storage: LRUCache, version_state, commit_id, new_head):
    parent_node = new_head.parent
    del version_state["commit_node_map"][commit_id]
    for i, child in enumerate(parent_node.children):
        if child.commit_id == commit_id:
            parent_node.children[i] = new_head
            break

    save_version_info(version_state, storage)


def delete_version_from_storage(storage: LRUCache, commit_id: str):
    deletion_folder = "/".join(("versions", commit_id))
    storage.clear(prefix=deletion_folder)
    storage.flush()


def replace_head(storage: LRUCache, version_state: Dict, reset_commit_id: str):
    """Replace HEAD of current branch with new HEAD"""
    branch = version_state["commit_node_map"][reset_commit_id].branch
    parent_commit_id = version_state["commit_id"]
    new_commit_id = generate_hash()

    new_node = _create_new_head(
        storage, version_state, branch, parent_commit_id, new_commit_id
    )

    _replace_head(storage, version_state, reset_commit_id, new_node)

    delete_version_from_storage(storage, reset_commit_id)

    return new_node.commit_id


def _replace_missing_with_head(missing_id: str, commits: Dict, branch_commit_map: Dict):
    new_commit_id = generate_hash()
    branch = None
    parent_commit_id = None
    for commit_id, commit_info in commits.items():
        if missing_id in commit_info["children"]:
            commit_info["children"].remove(missing_id)
            commit_info["children"].append(new_commit_id)
            branch = commit_info["branch"]
            parent_commit_id = commit_id
            break

    commit_info = {
        "branch": branch,
        "children": [],
        "parent": parent_commit_id,
        "commit_message": None,
        "commit_time": None,
        "commit_user_name": None,
    }
    commits[new_commit_id] = commit_info
    branch_commit_map[branch] = new_commit_id

    return branch, parent_commit_id, new_commit_id


def rebuild_version_info(storage: LRUCache):
    """Rebuilds version info from commit info."""
    branch_commit_map: Dict[str, str] = {}
    commits: Dict[str, Dict] = {}

    # don't do anything if first commit info is missing
    try:
        commit_info = load_commit_info(FIRST_COMMIT_ID, storage)
    except Exception:
        return

    stack = [FIRST_COMMIT_ID]

    new_heads = []

    while stack:
        commit_id = stack.pop()

        try:
            commit_info = load_commit_info(commit_id, storage)
        except KeyError:
            if commit_id != FIRST_COMMIT_ID:
                new_head = _replace_missing_with_head(
                    commit_id, commits, branch_commit_map
                )
                new_heads.append(new_head)
                continue
            raise
        commits[commit_id] = commit_info
        if commit_info["commit_time"] is None:
            branch_commit_map[commit_info["branch"]] = commit_id
        stack += commit_info["children"]

    if not commits:
        return

    base_storage = get_base_storage(storage)
    lock = Lock(storage, get_version_control_info_lock_key(), duration=10)
    lock.acquire()  # Blocking
    try:
        del storage[get_version_control_info_key()]
    except KeyError:
        pass
    key = get_version_control_info_key()
    version_info = {"commits": commits, "branches": branch_commit_map}
    base_storage[key] = json.dumps(version_info).encode("utf-8")
    lock.release()

    version_info = _version_info_from_json(version_info)

    for new_head in new_heads:
        _create_new_head(storage, version_info, *new_head)

    return version_info


def auto_checkout(dataset, flush_version_control_info: bool = True) -> bool:
    """Automatically checks out if current node is not the head node of the branch. This may happen either during commit/setitem/append/extend/create_tensor/delete_tensor/info updates."""
    version_state = dataset.version_state
    if not version_state["commit_node"].is_head_node:
        current_branch = version_state["branch"]
        auto_branch = f"auto_{generate_hash()}"
        logger.info(
            f"Automatically checking out to branch '{auto_branch}' as not currently at the head node of branch '{current_branch}'."
        )
        checkout(
            dataset,
            auto_branch,
            True,
            flush_version_control_info=flush_version_control_info,
        )
        return True
    return False


def auto_commit(dataset, message: str, flush_version_control_info: bool = True) -> None:
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
    commit(
        dataset,
        message,
        flush_version_control_info=flush_version_control_info,
        reload_meta=False,
    )
    checkout(dataset, original_commit_id)


def current_commit_has_change(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    return (
        version_state["commit_id"] == FIRST_COMMIT_ID
        or current_commit_has_data(version_state, storage)
        or current_commit_has_info_modified(version_state, storage)
    )


def current_commit_has_data(version_state: Dict[str, Any], storage: LRUCache) -> bool:
    """Checks if the current commit has any data present in it or not."""
    commit_id = version_state["commit_id"]
    try:
        dataset_diff_key = get_dataset_diff_key(commit_id)
        dataset_diff = storage.get_deeplake_object(dataset_diff_key, DatasetDiff)
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
        dataset_diff = storage.get_deeplake_object(dataset_diff_key, DatasetDiff)
        if dataset_diff.info_updated:
            return True
    except KeyError:
        pass

    for tensor in version_state["full_tensors"].keys():
        try:
            tensor_diff_key = get_tensor_commit_diff_key(tensor, commit_id)
            tensor_diff = storage.get_deeplake_object(tensor_diff_key, CommitDiff)
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


def _get_dataset_meta_at_commit(storage, commit_id):
    """Get dataset meta at commit."""
    meta_key = get_dataset_meta_key(commit_id)
    meta = storage.get_deeplake_object(meta_key, DatasetMeta)
    if not meta.tensor_names:  # backward compatibility
        meta.tensor_names = {key: key for key in meta.tensors}
    storage.register_deeplake_object(meta_key, meta)
    return meta


def load_meta(dataset: "deeplake.core.dataset.Dataset"):
    """Loads the meta info for the version state."""
    from deeplake.core.tensor import Tensor

    version_state = dataset.version_state
    storage: LRUCache = dataset.storage
    storage.clear_deeplake_objects()
    dataset._info = None
    dataset._ds_diff = None
    meta = _get_dataset_meta_at_commit(storage, version_state["commit_id"])

    ffw_dataset_meta(meta)
    version_state["meta"] = meta

    _tensors = version_state["full_tensors"]
    _tensors.clear()
    _tensor_names = version_state["tensor_names"]
    _tensor_names.clear()
    _tensor_names.update(meta.tensor_names)

    for tensor_key in _tensor_names.values():
        if tensor_key.startswith("__temp"):
            dataset._temp_tensors.append(tensor_key)
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
    return inp.tobytes() if isinstance(inp, DeepLakeMemoryObject) else inp
