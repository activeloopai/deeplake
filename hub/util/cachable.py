import hub
import posixpath


def reset_cachables(target_ds: hub.Dataset) -> None:
    tensors = list(target_ds.meta.tensors)
    for tensor in tensors:
        rel_path = posixpath.relpath(tensor, target_ds.group_index)
        chunk_engine = target_ds[rel_path].chunk_engine
        chunk_engine._tensor_meta = None
        chunk_engine._chunk_id_encoder = None
        chunk_engine._tile_encoder = None
        chunk_engine._commit_chunk_set = None
        chunk_engine._commit_diff = None
