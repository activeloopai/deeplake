from posixpath import relpath
from typing import List

from hub import Dataset

def reset_cachables(target_ds: Dataset, tensors: List[str]) -> None:
    for tensor in tensors:
        rel_path = relpath(tensor, target_ds.group_index)
        chunk_engine = target_ds[rel_path].chunk_engine
        chunk_engine._tensor_meta = None
        chunk_engine._chunk_id_encoder = None
        chunk_engine._tile_encoder = None
        chunk_engine._commit_chunk_set = None
        chunk_engine._commit_diff = None
        chunk_engine.cached_data = None
        chunk_engine.cachables_in_dirty_keys = False
        chunk_engine.cache_range = range(0)
