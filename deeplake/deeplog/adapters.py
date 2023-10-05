from typing import Tuple

from deeplake.constants import FIRST_COMMIT_ID
from deeplake.deeplog import DeepLog, DeepLogSnapshot


def get_tensor_metadata(
    key: str, deeplog: DeepLog, branch_id: str, branch_version: int
):
    from deeplake.core.meta import TensorMeta

    snapshot = DeepLogSnapshot(branch_id, branch_version, deeplog)
    create_tensor = {tensor.id: tensor for tensor in snapshot.tensors()}[key]

    meta = TensorMeta()
    meta.name = create_tensor.name
    meta.htype = create_tensor.htype
    meta.dtype = create_tensor.dtype
    meta.typestr = create_tensor.typestr
    meta.min_shape = create_tensor.min_shape
    meta.max_shape = create_tensor.max_shape
    meta.length = create_tensor.length
    meta.sample_compression = create_tensor.sample_compression
    meta.chunk_compression = create_tensor.chunk_compression
    meta.max_chunk_size = create_tensor.max_chunk_size
    meta.tiling_threshold = create_tensor.tiling_threshold
    meta.hidden = create_tensor.hidden
    meta.links = create_tensor.links
    meta.is_sequence = create_tensor.is_sequence
    meta.is_link = create_tensor.is_link
    meta.verify = create_tensor.verify

    return meta


def parse_commit_id(commit_id: str) -> Tuple[str, int]:
    branch_id, _, branch_version = commit_id.partition("-")

    if branch_id == FIRST_COMMIT_ID:
        branch_id = ""
    return branch_id, int(branch_version)


def to_commit_id(branch_id: str, branch_version: int) -> str:
    if branch_id == "":
        branch_id = FIRST_COMMIT_ID
    return branch_id + "-" + str(branch_version)
