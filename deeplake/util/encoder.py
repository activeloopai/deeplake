import deeplake
import numpy as np
from typing import Dict, List
from deeplake.core.meta.encode.creds import CredsEncoder
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.meta.encode.chunk_id import ChunkIdEncoder
from deeplake.core.meta.encode.tile import TileEncoder
from deeplake.core.meta.encode.sequence import SequenceEncoder
from deeplake.core.meta.encode.pad import PadEncoder
from deeplake.core.storage.provider import StorageProvider
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.util.keys import (
    get_creds_encoder_key,
    get_sequence_encoder_key,
    get_pad_encoder_key,
    get_tensor_commit_chunk_map_key,
    get_tensor_commit_diff_key,
    get_tensor_meta_key,
    get_chunk_id_encoder_key,
    get_chunk_id_encoder_key,
    get_tensor_tile_encoder_key,
)
import posixpath

from deeplake.util.path import relpath


def merge_all_meta_info(
    target_ds, storage, generated_tensors, overwrite, all_num_samples, result
):
    merge_all_commit_diffs(
        result["commit_diffs"], target_ds, storage, overwrite, generated_tensors
    )
    merge_all_tile_encoders(
        result["tile_encoders"],
        all_num_samples,
        target_ds,
        storage,
        overwrite,
        generated_tensors,
    )
    merge_all_tensor_metas(
        result["tensor_metas"], target_ds, storage, overwrite, generated_tensors
    )
    merge_all_chunk_id_encoders(
        result["chunk_id_encoders"], target_ds, storage, overwrite, generated_tensors
    )
    merge_all_creds_encoders(
        result["creds_encoders"], target_ds, storage, overwrite, generated_tensors
    )
    merge_all_sequence_encoders(
        result["sequence_encoders"], target_ds, storage, overwrite, generated_tensors
    )
    merge_all_pad_encoders(
        result["pad_encoders"], target_ds, storage, overwrite, generated_tensors
    )
    if target_ds.commit_id is not None:
        merge_all_commit_chunk_maps(
            result["commit_chunk_maps"],
            target_ds,
            storage,
            overwrite,
            generated_tensors,
        )


def merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges tensor metas from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
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
    ds_tensor_meta.update_length(worker_tensor_meta.length)
    if len(ds_tensor_meta.max_shape) == 0 or ds_tensor_meta.dtype is None:
        ds_tensor_meta.set_dtype_str(worker_tensor_meta.dtype)
        if not ds_tensor_meta.htype and worker_tensor_meta.htype:
            ds_tensor_meta.set_htype(worker_tensor_meta.htype)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)
    # len of min_shape will be 0 if 0 outputs from worker
    elif len(worker_tensor_meta.min_shape) != 0:
        assert (
            ds_tensor_meta.dtype == worker_tensor_meta.dtype
            or worker_tensor_meta.dtype is None
        )
        assert (
            ds_tensor_meta.htype == worker_tensor_meta.htype
            or worker_tensor_meta.htype is None
        )
        # TODO we can support this once we have ragged tensor support
        assert len(ds_tensor_meta.max_shape) == len(worker_tensor_meta.max_shape)
        assert len(ds_tensor_meta.min_shape) == len(worker_tensor_meta.min_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.max_shape)
        ds_tensor_meta.update_shape_interval(worker_tensor_meta.min_shape)


def merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges chunk_id_encoders from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
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
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
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


def merge_all_commit_chunk_maps(
    all_workers_commit_chunk_maps: List[Dict[str, CommitChunkMap]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_chunk_maps from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
        commit_chunk_map = (
            None if overwrite else target_ds[rel_path].chunk_engine.commit_chunk_map
        )
        for current_worker_commit_chunk_map in all_workers_commit_chunk_maps:
            current_commit_chunk_map = current_worker_commit_chunk_map[tensor]
            if commit_chunk_map is None:
                commit_chunk_map = current_commit_chunk_map
            else:
                combine_commit_chunk_maps(commit_chunk_map, current_commit_chunk_map)

        commit_chunk_key = get_tensor_commit_chunk_map_key(tensor, commit_id)
        storage[commit_chunk_key] = commit_chunk_map.tobytes()  # type: ignore


def combine_commit_chunk_maps(
    ds_commit_chunk_map: CommitChunkMap,
    worker_commit_chunk_map: CommitChunkMap,
) -> None:
    """Combines the dataset's commit_chunk_map with a single worker's commit_chunk_map."""
    ds_commit_chunk_map.chunks.update(worker_commit_chunk_map.chunks)


def merge_all_commit_diffs(
    all_workers_commit_diffs: List[Dict[str, CommitDiff]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    """Merges commit_diffs from all workers into a single one and stores it in target_ds."""
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)  # type: ignore
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


def merge_all_creds_encoders(
    all_workers_creds_encoders: List[Dict[str, CredsEncoder]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
        actual_tensor = target_ds[rel_path]
        if not actual_tensor.is_link:
            continue

        creds_encoder = None if overwrite else actual_tensor.chunk_engine.creds_encoder
        for current_worker_creds_encoder in all_workers_creds_encoders:
            current_creds_encoder = current_worker_creds_encoder[tensor]
            if creds_encoder is None:
                creds_encoder = current_creds_encoder
            else:
                combine_creds_encoders(creds_encoder, current_creds_encoder)

        creds_key = get_creds_encoder_key(tensor, commit_id)
        storage[creds_key] = creds_encoder.tobytes()  # type: ignore


def combine_creds_encoders(
    ds_creds_encoder: CredsEncoder, worker_creds_encoder: CredsEncoder
) -> None:
    """Combines the dataset's creds_encoder with a single worker's creds_encoder."""
    arr = worker_creds_encoder.array
    num_entries = len(arr)
    last_index = -1
    for i in range(num_entries):
        next_last_index = arr[i][1]
        num_samples = next_last_index - last_index
        ds_creds_encoder.register_samples((arr[i][0],), num_samples)
        last_index = next_last_index


def merge_all_sequence_encoders(
    all_workers_sequence_encoders: List[Dict[str, SequenceEncoder]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
        actual_tensor = target_ds[rel_path]
        if not actual_tensor.is_sequence:
            continue
        sequence_encoder = (
            None if overwrite else actual_tensor.chunk_engine.sequence_encoder
        )
        for current_worker_sequence_encoder in all_workers_sequence_encoders:
            current_sequence_encoder = current_worker_sequence_encoder[tensor]
            if sequence_encoder is None:
                sequence_encoder = current_sequence_encoder
            else:
                combine_sequence_encoders(sequence_encoder, current_sequence_encoder)

        sequence_key = get_sequence_encoder_key(tensor, commit_id)
        storage[sequence_key] = sequence_encoder.tobytes()  # type: ignore


def combine_sequence_encoders(
    ds_sequence_encoder: SequenceEncoder, worker_sequence_encoder: SequenceEncoder
) -> None:
    """Combines the dataset's sequence_encoder with a single worker's sequence_encoder."""
    arr = worker_sequence_encoder.array
    last_index = -1
    for i in range(len(arr)):
        next_last_index = arr[i][2]
        ds_sequence_encoder.register_samples(arr[i][0], next_last_index - last_index)
        last_index = next_last_index


def combine_pad_encoders(
    ds_pad_encoder: PadEncoder, worker_pad_encoder: PadEncoder
) -> PadEncoder:
    enc = PadEncoder()
    idx = None
    arr1 = ds_pad_encoder.array
    arr2 = worker_pad_encoder.array
    if not arr1.size or not arr2.size:
        return enc
    for i in range(int(max(arr1.max(), arr2.max())) + 1):
        if ds_pad_encoder.is_padded(i) and worker_pad_encoder.is_padded(i):
            if idx is None:
                idx = i
        else:
            if idx is not None:
                enc.add_padding(idx, i - idx)
                idx = None
    return enc


def merge_all_pad_encoders(
    all_workers_pad_encoders: List[Dict[str, PadEncoder]],
    target_ds: deeplake.Dataset,
    storage: StorageProvider,
    overwrite: bool,
    tensors: List[str],
) -> None:
    commit_id = target_ds.version_state["commit_id"]
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
        actual_tensor = target_ds[rel_path]
        pad_encoder = None if overwrite else actual_tensor.chunk_engine.pad_encoder
        for current_worker_pad_encoder in all_workers_pad_encoders:
            current_pad_encoder = current_worker_pad_encoder[tensor]
            if pad_encoder is None:
                pad_encoder = current_pad_encoder
            else:
                pad_encoder = combine_pad_encoders(pad_encoder, current_pad_encoder)

        pad_key = get_pad_encoder_key(tensor, commit_id)
        storage[pad_key] = pad_encoder.tobytes()  # type: ignore
