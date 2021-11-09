import hub
import numpy as np
from typing import Dict, List

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.core.meta.encode.tile import TileEncoder
from hub.util.keys import (
    get_tensor_meta_key,
    get_chunk_id_encoder_key,
    get_tensor_tile_encoder_key,
)
import posixpath


def merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]],
    ds_out: hub.Dataset,
) -> None:
    """Merges tensor metas from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    commit_id = ds_out.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, ds_out.group_index)  # type: ignore
        tensor_meta = ds_out[rel_path].meta  # type: ignore
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            combine_metas(tensor_meta, current_meta)
        meta_key = get_tensor_meta_key(tensor, commit_id)
        ds_out[rel_path].chunk_engine.cache[meta_key] = tensor_meta  # type: ignore
    ds_out.flush()


def combine_metas(ds_tensor_meta: TensorMeta, worker_tensor_meta: TensorMeta) -> None:
    """Combines the dataset's tensor meta with a single worker's tensor meta."""
    # if tensor meta is empty, copy attributes from current_meta
    if len(ds_tensor_meta.max_shape) == 0 or ds_tensor_meta.dtype is None:
        ds_tensor_meta.dtype = worker_tensor_meta.dtype
        ds_tensor_meta.length += worker_tensor_meta.length
        ds_tensor_meta.max_shape = worker_tensor_meta.max_shape
        ds_tensor_meta.min_shape = worker_tensor_meta.min_shape

    # len of min_shape will be 0 if 0 outputs from worker
    elif len(worker_tensor_meta.min_shape) != 0:
        assert ds_tensor_meta.dtype == worker_tensor_meta.dtype
        # TODO we can support this once we have ragged tensor support
        assert len(ds_tensor_meta.max_shape) == len(worker_tensor_meta.max_shape)
        assert len(ds_tensor_meta.min_shape) == len(worker_tensor_meta.min_shape)
        ds_tensor_meta.length += worker_tensor_meta.length
        ds_tensor_meta.update_shape_interval(tuple(worker_tensor_meta.max_shape))
        ds_tensor_meta.update_shape_interval(tuple(worker_tensor_meta.min_shape))


def merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]],
    ds_out: hub.Dataset,
) -> None:
    """Merges chunk_id_encoders from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    commit_id = ds_out.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, ds_out.group_index)  # type: ignore
        chunk_id_encoder = ds_out[rel_path].chunk_engine.chunk_id_encoder  # type: ignore
        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            combine_chunk_id_encoders(chunk_id_encoder, current_chunk_id_encoder)

        chunk_id_key = get_chunk_id_encoder_key(tensor, commit_id)
        ds_out[rel_path].chunk_engine.cache[chunk_id_key] = chunk_id_encoder  # type: ignore
    ds_out.flush()


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
    ds_out: hub.core.dataset.Dataset,
) -> None:
    tensors = list(ds_out.meta.tensors)
    commit_id = ds_out.version_state["commit_id"]
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, ds_out.group_index)  # type: ignore
        chunk_engine = ds_out[rel_path].chunk_engine
        offset = chunk_engine.num_samples
        tile_encoder = chunk_engine.tile_encoder
        for i, current_worker_tile_encoder in enumerate(all_workers_tile_encoders):
            current_tile_encoder = current_worker_tile_encoder[tensor]
            combine_tile_encoders(tile_encoder, current_tile_encoder, offset)
            offset += all_num_samples[i][tensor]
        tile_key = get_tensor_tile_encoder_key(tensor, commit_id)
        chunk_engine.cache[tile_key] = tile_encoder
    ds_out.flush()


def combine_tile_encoders(
    ds_tile_encoder: TileEncoder, worker_tile_encoder: TileEncoder, offset: int
) -> None:
    """Combines the dataset's tile_encoder with a single worker's tile_encoder."""

    if len(worker_tile_encoder.entries) != 0:
        for sample_index in worker_tile_encoder.entries.keys():
            new_sample_index = int(sample_index) + offset

            if new_sample_index in ds_tile_encoder.entries:
                raise ValueError(
                    f"Sample index {new_sample_index} already exists inside `ds_tile_encoder`. Keys={str(ds_tile_encoder.keys())}"
                )

            ds_tile_encoder.entries[
                str(new_sample_index)
            ] = worker_tile_encoder.entries[sample_index]
