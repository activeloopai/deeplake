import hub
import numpy as np
from typing import Dict, List

from hub.core.meta.tensor_meta import TensorMeta
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.util.keys import get_tensor_meta_key, get_chunk_id_encoder_key


def merge_all_tensor_metas(
    all_workers_tensor_metas: List[Dict[str, TensorMeta]],
    ds_out: hub.core.dataset.Dataset,
) -> None:
    """Merges tensor metas from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        tensor_meta = ds_out[tensor].meta
        for current_worker_metas in all_workers_tensor_metas:
            current_meta = current_worker_metas[tensor]
            combine_metas(tensor_meta, current_meta)
        meta_key = get_tensor_meta_key(tensor)
        ds_out[tensor].chunk_engine.cache[meta_key] = tensor_meta
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
        ds_tensor_meta._update_shape_interval(tuple(worker_tensor_meta.max_shape))
        ds_tensor_meta._update_shape_interval(tuple(worker_tensor_meta.min_shape))


def merge_all_chunk_id_encoders(
    all_workers_chunk_id_encoders: List[Dict[str, ChunkIdEncoder]],
    ds_out: hub.core.dataset.Dataset,
) -> None:
    """Merges chunk_id_encoders from all workers into a single one and stores it in ds_out."""
    tensors = list(ds_out.meta.tensors)
    for tensor in tensors:
        chunk_id_encoder = ds_out[tensor].chunk_engine.chunk_id_encoder
        for current_worker_chunk_id_encoders in all_workers_chunk_id_encoders:
            current_chunk_id_encoder = current_worker_chunk_id_encoders[tensor]
            combine_chunk_id_encoders(chunk_id_encoder, current_chunk_id_encoder)

        chunk_id_key = get_chunk_id_encoder_key(tensor)
        ds_out[tensor].chunk_engine.cache[chunk_id_key] = chunk_id_encoder
    ds_out.flush()


def combine_chunk_id_encoders(
    ds_chunk_id_encoder: ChunkIdEncoder,
    worker_chunk_id_encoder: ChunkIdEncoder,
) -> None:
    """Combines the dataset's chunk_id_encoder with a single worker's chunk_id_encoder."""
    encoded_ids = worker_chunk_id_encoder._encoded
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
