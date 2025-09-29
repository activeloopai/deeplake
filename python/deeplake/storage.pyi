"""
Storage readers and writers - s3, gcs, azure, filesystem, etc.
"""

from __future__ import annotations
import datetime
import deeplake._deeplake.core
import typing
from deeplake import Future

__all__ = ["Reader", "Writer", "ResourceMeta", "concurrency", "set_concurrency"]

class Reader:
    def __init__(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> None:
        ...

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def get(
        self, path: str, start_bytes: int | None = None, end_bytes: int | None = None
    ) -> deeplake._deeplake.core.MemoryBuffer: ...
    def get_async(self, path: str, start_bytes: int | None = None, end_bytes: int | None = None) -> Future[deeplake._deeplake.core.MemoryBuffer]:
        ...
    def length(self, path: str) -> int: ...
    def list(self, path: str = '') -> list[ResourceMeta]: ...
    def subdir(self, subpath: str) -> Reader: ...
    @property
    def original_path(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def token(self) -> str: ...

class Writer:
    def __init__(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> None:
        ...

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def get_reader(self) -> Reader:
        ...
    def remove(self, path: str) -> None: ...
    def remove_directory(self, prefix: str = "") -> None: ...
    def set(self, path: str, content: bytes) -> ResourceMeta: ...
    def set_async(self, path: str, content: bytes) -> Future[ResourceMeta]: ...
    def subdir(self, subpath: str) -> Writer: ...
    @property
    def original_path(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def token(self) -> str: ...

class ResourceMeta:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: ResourceMeta) -> bool:
        ...
    def __lt__(self, other: ResourceMeta) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def etag(self) -> str:
        ...
    @property
    def last_modified(self) -> datetime.datetime:
        ...
    @property
    def path(self) -> str:
        ...
    @property
    def size(self) -> int:
        ...

def concurrency() -> int:
    """
    Returns the number of threads of storage readers and writers.

    <!-- test-context
    ```python
    import deeplake
    ```
    -->

    Examples:
        ```python
        deeplake.storage.concurrency()
        ```
    """
    ...

def set_concurrency(num_threads: int) -> None:
    """
    Sets the number of threads of storage readers and writers.

    <!-- test-context
    ```python
    import deeplake
    ```
    -->

    Examples:
        ```python
        deeplake.storage.set_concurrency(64)
        ```
    """

