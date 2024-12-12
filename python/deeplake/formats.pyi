from __future__ import annotations

__all__ = ["Chunk", "DataFormat"]

class DataFormat:
    """
    Base class for all datafile formats.
    """

    def __str__(self) -> str: ...

def Chunk(
    sample_compression: str | None = None, chunk_compression: str | None = None
) -> DataFormat:
    """
    Configures a "chunk" datafile format

    Parameters:
          sample_compression (str, optional): How to compress individual values within the datafile
          chunk_compression (str, optional): How to compress the datafile as a whole
    """
    ...
