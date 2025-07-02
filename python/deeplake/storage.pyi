"""
Storage readers and writers - s3, gcs, azure, filesystem, etc.
"""

from __future__ import annotations
import datetime
import deeplake._deeplake.core
import typing

__all__ = ["Reader", "Writer", "ResourceMeta", "concurrency", "set_concurrency"]

class Reader:
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def get(
        self, path: str, start_bytes: int | None = None, end_bytes: int | None = None
    ) -> deeplake._deeplake.core.MemoryBuffer: ...
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
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def remove(self, path: str) -> None: ...
    def remove_directory(self, prefix: str = "") -> None: ...
    def set(self, path: str, content: bytes) -> None: ...
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

