from typing import NamedTuple, Dict, List, Optional, Any


class VectorStoreSummaryResponse(NamedTuple):
    status_code: int
    summary: str
    length: int
    tensors: List[
        Dict[str, Any]
    ]  # Same format as `tensor_params` in `init_vectorstore`


class VectorStoreInitResponse(NamedTuple):
    status_code: int
    path: str
    summary: str
    length: int
    tensors: List[Dict[str, Any]]
    exists: bool


class VectorStoreSearchResponse(NamedTuple):
    status_code: int
    length: int
    data: Dict[str, List[Any]]
    error: Optional[str] = None


class VectorStoreAddResponse(NamedTuple):
    status_code: int
    ids: Optional[List[str]] = None
    error: Optional[str] = None


class VectorStoreDeleteResponse(NamedTuple):
    status_code: int
    error: Optional[str] = None


class VectorStoreUpdateResponse(NamedTuple):
    status_code: int
    error: Optional[str] = None
