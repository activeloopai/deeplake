import hub
import numpy as np
from typing import Dict, List

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.meta.encode.tile import TileEncoder
from hub.core.storage.provider import StorageProvider
from hub.core.version_control.commit_chunk_set import CommitChunkSet
from hub.core.version_control.commit_diff import CommitDiff
from hub.util.keys import (
    get_tensor_commit_chunk_set_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_id_encoder_key,
    get_chunk_id_encoder_key,
    get_tensor_tile_encoder_key,
)
import posixpath


def merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]],
    target_ds: hub.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges tensor metas from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)
        tensor_meta = None if overwrite else target_ds[rel_path].meta
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            if tensor_meta is None:
                tensor_meta = current_meta
            else:
                combine_metas(tensor_meta, current_meta)
        meta_key = get_tensor_meta_key(tensor, commit_id)
        storage[meta_key] = tensor_meta.tobytes()  # type: ignore


def combine_metas(ds_tensor_meta: TensorMeta, worker_tensor_meta: TensorMeta) -> None:
    """Combines the dataset's tensor meta with a single worker's tensor meta."""
    # if tensor meta is empty, copy attributes from current_meta
    if len(ds_tensor_meta.max_shape) == 0 or ds_tensor_meta.dtype is None:
        ds_tensor_meta.set_dtype_str(worker_tensor_meta.dtype)
        ds_tensor_meta.update_length(worker_tensor_meta.length)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)
    # len of min_shape will be 0 if 0 outputs from worker
    elif len(worker_tensor_meta.min_shape) != 0:
        assert ds_tensor_meta.dtype == worker_tensor_meta.dtype
        # TODO we can support this once we have ragged tensor support
        assert len(ds_tensor_meta.max_shape) == len(worker_tensor_meta.max_shape)
        assert len(ds_tensor_meta.min_shape) == len(worker_tensor_meta.min_shape)
        ds_tensor_meta.update_length(worker_tensor_meta.length)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)


def merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]],
    target_ds: hub.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges chunk_id_encoders from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)
        chunk_id_encoder = (
            None if overwrite else target_ds[rel_path].chunk_engine.chunk_id_encoder
        )
        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            if chunk_id_encoder is None:
                chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            else:
                combine_chunk_id_encoders(chunk_id_encoder, current_chunk_id_encoder)

        chunk_id_key = get_chunk_id_encoder_key(tensor, commit_id)
        storage[chunk_id_key] = chunk_id_encoder.tobytes()  # type: ignore


def combine_chunk_id_encoders(
    ds_chunk_id_encoder: ChunkIdEncoder,
    worker_chunk_id_encoder: ChunkIdEncoder,
) -> None:
    """Combines the dataset's chunk_id_encoder with a single worker's chunk_id_encoder."""
    encoded_ids = worker_chunk_id_encoder._encoded
    if not encoded_ids.flags.writeable:
        encoded_ids = encoded_ids.copy()
    if encoded_ids.size != 0:
        offset = ds_chunk_id_encoder.num_samples
        for encoded_id in encoded_ids:
            encoded_id[1] += offset
            if ds_chunk_id_encoder._encoded.size == 0:
                ds_chunk_id_encoder._encoded = np.reshape(encoded_id, (-1, 2))
            else:
                ds_chunk_id_encoder._encoded = np.vstack(
                    [ds_chunk_id_encoder._encoded, encoded_id]
                )


def merge_all_tile_encoders(
    all_workers_tile_encoders: List[Dict[str, TileEncoder]],
    all_num_samples: List[Dict[str, int]],
    target_ds: hub.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)
        chunk_engine = target_ds[rel_path].chunk_engine
        offset = 0 if overwrite else chunk_engine.num_samples
        tile_encoder = None if overwrite else chunk_engine.tile_encoder
        for i, current_worker_tile_encoder in enumerate(all_workers_tile_encoders):
            current_tile_encoder = current_worker_tile_encoder[tensor]
            if tile_encoder is None:
                tile_encoder = current_tile_encoder
            else:
                combine_tile_encoders(tile_encoder, current_tile_encoder, offset)
            offset += all_num_samples[i][tensor]
        tile_key = get_tensor_tile_encoder_key(tensor, commit_id)
        storage[tile_key] = tile_encoder.tobytes()  # type: ignore
    target_ds.flush()


def merge_all_meta_info(
    target_ds,
    storage,
    generated_tensors,
    overwrite,
    all_commit_diffs,
    all_tile_encoders,
    all_num_samples,
    all_tensor_metas,
    all_chunk_id_encoders,
    all_chunk_commit_sets,
):
    merge_all_commit_diffs(
        all_commit_diffs, target_ds, storage, overwrite, generated_tensors
    )
    merge_all_tile_encoders(
        all_tile_encoders,
        all_num_samples,
        target_ds,
        storage,
        overwrite,
        generated_tensors,
    )
    merge_all_tensor_metas(
        all_tensor_metas, target_ds, storage, overwrite, generated_tensors
    )
    merge_all_chunk_id_encoders(
        all_chunk_id_encoders, target_ds, storage, overwrite, generated_tensors
    )
    if target_ds.commit_id is not None:
        merge_all_commit_chunk_sets(
            all_chunk_commit_sets, target_ds, storage, overwrite, generated_tensors
        )


def combine_tile_encoders(
    ds_tile_encoder: TileEncoder, worker_tile_encoder: TileEncoder, offset: int
) -> None:
    """Combines the dataset's tile_encoder with a single worker's tile_encoder."""

    if len(worker_tile_encoder.entries) != 0:
        for sample_index in worker_tile_encoder.entries.keys():
            new_sample_index = int(sample_index) + offset

            if new_sample_index in ds_tile_encoder.entries:
                raise ValueError(
                    f"Sample index {new_sample_index} already exists inside `ds_tile_encoder`. Keys={ds_tile_encoder.entries}"
                )

            ds_tile_encoder.entries[new_sample_index] = worker_tile_encoder.entries[
                sample_index
            ]


def merge_all_commit_chunk_sets(
    all_workers_commit_chunk_sets: List[Dict[str, CommitChunkSet]],
    target_ds: hub.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_chunk_sets from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)
        commit_chunk_set = (
            None if overwrite else target_ds[rel_path].chunk_engine.commit_chunk_set
        )
        for current_worker_commit_chunk_set in all_workers_commit_chunk_sets:
            current_commit_chunk_set = current_worker_commit_chunk_set[tensor]
            if commit_chunk_set is None:
                commit_chunk_set = current_commit_chunk_set
            else:
                combine_commit_chunk_sets(commit_chunk_set, current_commit_chunk_set)

        commit_chunk_key = get_tensor_commit_chunk_set_key(tensor, commit_id)
        storage[commit_chunk_key] = commit_chunk_set.tobytes()  # type: ignore


def combine_commit_chunk_sets(
    ds_commit_chunk_set: CommitChunkSet,
    worker_commit_chunk_set: CommitChunkSet,
) -> None:
    """Combines the dataset's commit_chunk_set with a single worker's commit_chunk_set."""
    ds_commit_chunk_set.chunks.update(worker_commit_chunk_set.chunks)


def merge_all_commit_diffs(
    all_workers_commit_diffs: List[Dict[str, CommitDiff]],
    target_ds: hub.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_diffs from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)  # type: ignore
        commit_diff = None if overwrite else target_ds[rel_path].chunk_engine.commit_diff  # type: ignore
        for current_worker_commit_diffs in all_workers_commit_diffs:
            current_commit_diff = current_worker_commit_diffs[tensor]
            if commit_diff is None:
                commit_diff = current_commit_diff
                commit_diff.transform_data()
            else:
                combine_commit_diffs(commit_diff, current_commit_diff)

        commit_chunk_key = get_tensor_commit_diff_key(tensor, commit_id)
        storage[commit_chunk_key] = commit_diff.tobytes()  # type: ignore


def combine_commit_diffs(
    ds_commit_diff: CommitDiff, worker_commit_diff: CommitDiff
) -> None:
    """Combines the dataset's commit_diff with a single worker's commit_diff."""
    ds_commit_diff.add_data(worker_commit_diff.num_samples_added)
