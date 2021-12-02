from typing import List, Set
import hub
from hub.core.meta.encode.chunk_id import ChunkIdEncoder
from hub.util.keys import get_chunk_key


def get_chunk_paths(dataset: hub.Dataset, tensors: List[str]) -> Set[str]:
    """Returns the paths to the chunks present in the current commit of the dataset"""
    commit_id = dataset.commit_id
    chunk_paths = set()
    for tensor in tensors:
        chunk_engine = dataset[tensor].chunk_engine
        commit_chunk_set = chunk_engine.commit_chunk_set
        if commit_chunk_set is None:
            enc = chunk_engine.chunk_id_encoder
            ids = enc._encoded[:, 0]
            chunks = {ChunkIdEncoder.name_from_id(id) for id in ids}
        else:
            chunks = commit_chunk_set.chunks
        key = chunk_engine.key
        cur_chunk_paths = {
            get_chunk_key(key, chunk_name, commit_id) for chunk_name in chunks
        }
        chunk_paths.update(cur_chunk_paths)
    return chunk_paths
