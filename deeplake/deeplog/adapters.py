from typing import Tuple

from deeplake.constants import FIRST_COMMIT_ID

from deeplake.core.meta import TensorMeta
from deeplake.deeplog import DeepLog


def get_tensor_metadata(
    deeplog: DeepLog, branch_id: str, branch_version: int
) -> TensorMeta:
    create_tensor = [
        tensor for tensor in deeplog.tensors(branch_id, branch_version).data()
    ][0]
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
    branch_id, _, branch_version = commit_id.rpartition("-")

    if branch_id == "":
        branch_id = FIRST_COMMIT_ID
    return branch_id, int(branch_version)


def to_commit_id(branch_id: str, branch_version: int) -> str:
    if branch_id == "":
        branch_id = FIRST_COMMIT_ID
    return branch_id + "-" + str(branch_version)
