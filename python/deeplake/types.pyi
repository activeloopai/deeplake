from __future__ import annotations

import typing

__all__ = ["Array",
    "Audio",
    "BM25",
    "Binary",
    "BinaryMask",
    "Bool",
    "BoundingBox",
    "Bytes",
    "ClassLabel",
    "Clustered",
    "ClusteredQuantized",
    "DataType",
    "Dict",
    "Embedding",
    "EmbeddingIndex",
    "EmbeddingIndexType",
    "EmbeddingsMatrixIndex",
    "EmbeddingsMatrixIndexType",
    "Exact",
    "Float16",
    "Float32",
    "Float64",
    "Image",
    "Index",
    "IndexType",
    "Int16",
    "Int32",
    "Int64",
    "Int8",
    "Inverted",
    "JsonIndex",
    "Link",
    "Medical",
    "Mesh",
    "NumericIndex",
    "Point",
    "Polygon",
    "PooledQuantized",
    "QuantizationType",
    "SegmentMask",
    "Sequence",
    "Struct",
    "Text",
    "TextIndex",
    "Type",
    "TypeKind",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt8",
    "Video"]

class EmbeddingIndexType:
    """
    Represents embedding index type.
    """
    @typing.overload
    def __init__(self, type: IndexType) -> None:
        ...
    @typing.overload
    def __init__(self, quantization: QuantizationType) -> None:
        ...
    def __init__(self, type: IndexType | QuantizationType) -> None:
        ...

class QuantizationType:
    """
    Enumeration of available quantization types for embeddings.

    Members:
        Binary:
            Stores a binary quantized representation of the original embedding in the index
            rather than a full copy of the embedding. This slightly decreases accuracy of
            searches, while significantly improving query time.

    <!--
    ```python
    import deeplake
    import numpy as np
    from deeplake import types
    ds = deeplake.create("tmp://")
    ```
    -->
    """

    Binary: typing.ClassVar[QuantizationType]
    __members__: typing.ClassVar[dict[str, QuantizationType]]

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the quantization type.
        """
        ...

    @property
    def value(self) -> int:
        """
        Returns:
            int: The integer value of the quantization type.
        """
        ...

Binary: QuantizationType
"""
Binary quantization type for embeddings.

This slightly decreases accuracy of searches while significantly improving query time
by storing a binary quantized representation instead of the full embedding.
"""

class IndexType:
    """
    Enumeration of available text/numeric/JSON/embeddings/embeddings matrix indexing types.

    Attributes:
        Inverted: An index that supports keyword lookup. Can be used with ``CONTAINS(column, 'wanted_value')``.
        BM25: A BM25-based index of text data. Can be used with ``BM25_SIMILARITY(column, 'search text')`` in a TQL ``ORDER BY`` clause.
        Exact: An exact match index for text data.
        PooledQuantized: A pooled quantized index for 2D embeddings matrices. Can be used with ``MAXSIM(column, query_embeddings)`` for ColBERT-style maximum similarity search.
        Clustered: Clusters embeddings in the index to speed up search. This is the default index type for embeddings.
        ClusteredQuantized: Stores a binary quantized representation of the original embedding in the index rather than a full copy of the embedding. This slightly decreases accuracy of searches, while significantly improving query time.
    """

    BM25: typing.ClassVar[IndexType]  # value = <IndexType.BM25: 2>
    Inverted: typing.ClassVar[IndexType]  # value = <IndexType.Inverted: 1>
    Exact: typing.ClassVar[IndexType]  # value = <IndexType.Exact: 3>
    PooledQuantized: typing.ClassVar[IndexType]  # value = <IndexType.PooledQuantized: 4>
    Clustered: typing.ClassVar[IndexType]  # value = <IndexType.Clustered: 5>
    ClusteredQuantized: typing.ClassVar[IndexType]  # value = <IndexType.ClusteredQuantized: 6>
    __members__: typing.ClassVar[dict[str, IndexType]]  # value = {'Inverted': <IndexType.Inverted: 1>, 'BM25': <IndexType.BM25: 2>, 'Exact': <IndexType.Exact: 3>, 'PooledQuantized': <IndexType.PooledQuantized: 4>, 'Clustered': <IndexType.Clustered: 5>, 'ClusteredQuantized': <IndexType.ClusteredQuantized: 6>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        """
        Returns:
            int: The integer value of the text index type.
        """
        ...


class NumericIndex:
    """
    Represents a numeric column index type.

    Used to create indexes on numeric columns for faster query performance.
    Supports inverted indexing for CONTAINS operations.
    """
    __hash__: typing.ClassVar[None] = None
    def __init__(self, type: IndexType) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __eq__(self, other: NumericIndex) -> bool:
        ...

class TextIndex:
    """
    Represents a text column index type.

    Used to create indexes on text columns for faster query performance.
    Supports inverted indexing (CONTAINS), BM25 similarity search, and exact matching.
    """
    __hash__: typing.ClassVar[None] = None
    def __init__(self, type: IndexType) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __eq__(self, other: TextIndex) -> bool:
        ...

class EmbeddingsMatrixIndexType:
    """
    Represents a 2D embeddings matrix index type.

    Used for ColBERT-style maximum similarity search on 2D embedding matrices.
    Supports pooled quantized indexing for efficient MAXSIM queries.
    """
    def __init__(self) -> None:
        ...

class JsonIndex:
    """
    Represents a Dict column index type.

    Used to create indexes on Dict columns for faster query performance.
    Supports inverted indexing for CONTAINS operations on JSON fields.
    """
    __hash__: typing.ClassVar[None] = None
    def __init__(self, type: IndexType) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __eq__(self, other: JsonIndex) -> bool:
        ...

class Index:
    """
    Represents all available index types in the deeplake.
    This is a polymorphic wrapper that can hold any specific index type.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: Index) -> bool:
        ...
    def __init__(self, index_type: TextIndex | EmbeddingIndexType | EmbeddingsMatrixIndexType | JsonIndex | NumericIndex) -> None:
        ...
    def __ne__(self, other: Index) -> bool:
        ...
    def __str__(self) -> str:
        ...

class DataType:
    """
    The base class all specific types extend from.

    This class provides the foundation for all data types in the deeplake.
    """

    def __eq__(self, other: DataType) -> bool: ...
    def __ne__(self, other: DataType) -> bool: ...
    def __str__(self) -> str: ...

class Type:
    """
    Base class for all complex data types in the deeplake.

    This class extends DataType to provide additional functionality for complex types
    like images, embeddings, and sequences.
    """

    def __str__(self) -> str: ...
    def __eq__(self, other: Type) -> bool: ...
    def __ne__(self, other: Type) -> bool: ...
    @property
    def data_type(self) -> DataType:
        """
        Returns:
            DataType: The underlying data type of this type.
        """
        ...

    @property
    def id(self) -> str:
        """
        Returns:
            str: The id (name) of the data type.
        """
        ...

    @property
    def is_sequence(self) -> bool:
        """
        Returns:
            bool: True if this type is a sequence, False otherwise.
        """
        ...

    @property
    def is_link(self) -> bool:
        """
        Returns:
            bool: True if this type is a link, False otherwise.
        """
        ...

    @property
    def is_image(self) -> bool:
        """
        Returns:
            bool: True if this type is an image, False otherwise.
        """
        ...

    @property
    def is_segment_mask(self) -> bool:
        """
        Returns:
            bool: True if this type is a segment mask, False otherwise.
        """
        ...

    @property
    def kind(self) -> TypeKind:
        """
        Returns:
            TypeKind: The kind of this type.
        """
        ...

    @property
    def shape(self) -> list[int] | None:
        """
        Returns:
            list[int] | None: The shape of the data type if applicable, otherwise None.
        """
        ...

class TypeKind:
    """
    Enumeration of all available type kinds in the deeplake.

    Members:
        Generic: Generic data type
        Text: Text data type
        Dict: Dictionary data type
        Embedding: Embedding data type
        Sequence: Sequence data type
        Image: Image data type
        BoundingBox: Bounding box data type
        BinaryMask: Binary mask data type
        SegmentMask: Segmentation mask data type
        Polygon: Polygon data type
        ClassLabel: Class label data type
        Point: Point data type
        Medical: Medical data type
        Mesh: Mesh data type
        Audio: Audio data type
        Video: Video data type
        Link: Link data type
    """
    Audio: typing.ClassVar[TypeKind]
    BinaryMask: typing.ClassVar[TypeKind]
    BoundingBox: typing.ClassVar[TypeKind]
    ClassLabel: typing.ClassVar[TypeKind]
    Dict: typing.ClassVar[TypeKind]
    Embedding: typing.ClassVar[TypeKind]
    Generic: typing.ClassVar[TypeKind]
    Image: typing.ClassVar[TypeKind]
    Link: typing.ClassVar[TypeKind]
    Polygon: typing.ClassVar[TypeKind]
    Point: typing.ClassVar[TypeKind]
    SegmentMask: typing.ClassVar[TypeKind]
    Sequence: typing.ClassVar[TypeKind]
    Text: typing.ClassVar[TypeKind]
    Medical: typing.ClassVar[TypeKind]
    Mesh: typing.ClassVar[TypeKind]
    Video: typing.ClassVar[TypeKind]
    __members__: typing.ClassVar[dict[str, TypeKind]]

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the type kind.
        """
        ...

    @property
    def value(self) -> int:
        """
        Returns:
            int: The integer value of the type kind.
        """
        ...

@typing.overload
def Array(dtype: DataType | str, dimensions: int) -> DataType: ...
@typing.overload
def Array(dtype: DataType | str, shape: list[int]) -> DataType: ...
@typing.overload
def Array(dtype: DataType | str) -> DataType: ...
def Array(
    dtype: DataType | str, dimensions: int | None, shape: list[int] | None
) -> DataType:
    """
    Creates a generic array of data.

    Parameters:
        dtype: DataType | str
            The datatype of values in the array
        dimensions: int | None
            The number of dimensions/axes in the array. Unlike specifying ``shape``,
            there is no constraint on the size of each dimension.
        shape: list[int] | None
            Constrain the size of each dimension in the array

    Returns:
        DataType: A new array data type with the specified parameters.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a three-dimensional array, where each dimension can have any number of elements:
        ```python
        ds.add_column("col1", types.Array("int32", dimensions=3))
        ```

        Create a three-dimensional array, where each dimension has a known size:
        ```python
        ds.add_column("col2", types.Array(types.Float32(), shape=[50, 30, 768]))
        ```
    """
    ...

def Audio(dtype: DataType | str = "uint8", sample_compression: str = "mp3") -> Type:
    """
    Creates an audio data type.

    Parameters:
        dtype: DataType | str
            The datatype of the audio samples. Defaults to "uint8".
        sample_compression: str
            The compression format for the audio samples wav or mp3. Defaults to "mp3".

    Returns:
        Type: A new audio data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create an audio column with default settings:
        ```python
        ds.add_column("col1", types.Audio())
        ```

        Create an audio column with specific sample compression:
        ```python
        ds.add_column("col2", types.Audio(sample_compression="wav"))
        ```
    """
    ...

def Bool() -> DataType:
    """
    Creates a boolean value type.

    Returns:
        DataType: A new boolean data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create columns with boolean type:
        ```python
        ds.add_column("col1", types.Bool)
        ds.add_column("col2", "bool")
        ```
    """
    ...


def Bytes() -> DataType:
    """
    Creates a byte array value type. This is useful for storing raw binary data.

    Returns:
        DataType: A new byte array data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create columns with byte array type:
        ```python
        ds.add_column("col1", types.Bytes)
        ds.add_column("col2", "bytes")
        ```

        Append raw binary data to a byte array column:
        ```python
        ds.append([{"col1": b"hello", "col2": b"world"}])
        ```
    """
    ...

def Text(index_type: str | IndexType | TextIndex | None = None, chunk_compression: str | None = 'lz4') -> Type:
    """
    Creates a text data type of arbitrary length.

    Parameters:
        index_type: str | IndexType | TextIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`
            - :class:`deeplake.types.BM25`
            - :class:`deeplake.types.Exact`

            Default is ``None`` meaning "do not index"

        chunk_compression: str | None
            defines the compression algorithm for on-disk storage of text data.
            supported values are 'lz4', 'zstd', and 'null' (no compression).

            Default is ``lz4``

    Returns:
        Type: A new text data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create text columns with different configurations:
        ```python
        ds.add_column("col1", types.Text)
        ds.add_column("col2", "text")
        ds.add_column("col3", str)
        ds.add_column("col4", types.Text(index_type=types.Inverted))
        ds.add_column("col5", types.Text(index_type=types.BM25))
        ```
    """
    ...

BM25: IndexType.BM25
"""
A BM25-based index of text data.

This index can be used with ``BM25_SIMILARITY(column, 'search text')`` in a TQL ``ORDER BY`` clause.

See Also:
    `BM25 Algorithm <https://en.wikipedia.org/wiki/Okapi_BM25>`_
"""

Inverted: IndexType.Inverted
"""
A text index that supports keyword lookup.

This index can be used with ``CONTAINS(column, 'wanted_value')``.
"""

Exact: IndexType.Exact
"""
A text index that supports whole text lookup.

This index can be used with ``EQUALS(column, 'wanted_value')``.
"""

PooledQuantized: IndexType.PooledQuantized
"""
A pooled quantized index for 2D embeddings matrices.

This index enables fast maximum similarity (MaxSim) search for ColBERT-style late interaction models.
Use this index with 2D float32 or float16 arrays representing token embeddings.

This index can be used with ``MAXSIM(column, query_embeddings)`` in TQL queries.

See Also:
    `ColBERT: Efficient and Effective Passage Search <https://arxiv.org/abs/2004.12832>`_
"""

Clustered: IndexType.Clustered
"""
Clustered index for embedding columns.

Clusters embeddings in the index to speed up similarity search. This is the default index type
for embedding columns.
"""

ClusteredQuantized: IndexType.ClusteredQuantized
"""
Clustered quantized index for embedding columns.

Stores a binary quantized representation of the original embedding in the index rather than
a full copy of the embedding. This slightly decreases accuracy of searches while significantly
improving query time and reducing storage requirements.
"""

def Dict(index_type: str | IndexType | JsonIndex | None = None) -> Type:
    """
    Creates a type that supports storing arbitrary key/value pairs in each row.

    Parameters:
        index_type: str | IndexType | JsonIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        Type: A new dictionary data type.

    See Also:
        :func:`deeplake.types.Struct` for a type that supports defining allowed keys.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create and use a dictionary column:
        ```python
        ds.add_column("col1", types.Dict)
        ds.append([{"col1": {"a": 1, "b": 2}}])
        ds.append([{"col1": {"b": 3, "c": 4}}])
        ```
    """
    ...

def Embedding(
    size: int | None = None,
    dtype: DataType | str = "float32",
    index_type: EmbeddingIndexType | QuantizationType | None = None,
) -> Type:
    """
    Creates a single-dimensional embedding of a given length.

    Parameters:
        size: int | None
            The size of the embedding
        dtype: DataType | str
            The datatype of the embedding. Defaults to float32
        index_type: EmbeddingIndexType | QuantizationType | None
            How to compress the embeddings in the index. Default uses no compression,
            but can be set to :class:`deeplake.types.QuantizationType.Binary`

    Returns:
        Type: A new embedding data type.

    See Also:
        :func:`deeplake.types.Array` for a multidimensional array.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create embedding columns:
        ```python
        ds.add_column("col1", types.Embedding(768))
        ds.add_column("col2", types.Embedding(768, index_type=types.EmbeddingIndex(types.ClusteredQuantized)))
        ```
    """
    ...

def EmbeddingIndex(type: IndexType | QuantizationType | None = None) -> EmbeddingIndexType:
    """
    Creates an embedding index.

    Parameters:
        type: IndexType | QuantizationType | None = None
            The index type for embeddings. Can be:

            - :class:`deeplake.types.IndexType.Clustered` - Default clustered index
            - :class:`deeplake.types.IndexType.ClusteredQuantized` - Quantized clustered index
            - :class:`deeplake.types.QuantizationType.Binary` - Binary quantization (maps to ClusteredQuantized)

    Returns:
        Type: EmbeddingIndexType.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create embedding columns with different index types:
        ```python
        # Using IndexType enum
        ds.add_column("col1", types.Embedding(768, index_type=types.EmbeddingIndex(types.IndexType.ClusteredQuantized)))

        # Using QuantizationType for backward compatibility
        ds.add_column("col2", types.Embedding(768, index_type=types.EmbeddingIndex(types.QuantizationType.Binary)))
        ```
    """
    ...

def EmbeddingsMatrixIndex() -> EmbeddingsMatrixIndexType:
    """
    Creates an embeddings matrix index.
    """
    ...

def Float16(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 16-bit (half) float value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 16-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 16-bit float type:
        ```python
        ds.add_column("col", types.Float16)
        ds.add_column("idx_col", deeplake.types.Float16(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Float16(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Float16("Inverted"))
        ```
    """
    ...


def Float32(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 32-bit float value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 32-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 32-bit float type:
        ```python
        ds.add_column("col", types.Float32)
        ds.add_column("idx_col", deeplake.types.Float32(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Float32(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Float32("Inverted"))
        ```
    """
    ...


def Float64(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 64-bit float value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 64-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 64-bit float type:
        ```python
        ds.add_column("col", types.Float64)
        ds.add_column("idx_col", deeplake.types.Float64(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Float64(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Float64("Inverted"))
        ```
    """
    ...


def Int16(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 16-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 16-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 16-bit integer type:
        ```python
        ds.add_column("col", types.Int16)
        ds.add_column("idx_col", deeplake.types.Int16(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Int16(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Int16("Inverted"))
        ```
    """
    ...


def Int32(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 32-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 32-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 32-bit integer type:
        ```python
        ds.add_column("col", types.Int32)
        ds.add_column("idx_col", deeplake.types.Int32(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Int32(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Int32("Inverted"))
        ```
    """
    ...


def Int64(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates a 64-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 64-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 64-bit integer type:
        ```python
        ds.add_column("col", types.Int64)
        ds.add_column("idx_col", deeplake.types.Int64(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Int64(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Int64("Inverted"))
        ```
    """
    ...


def Int8(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates an 8-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new 8-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 8-bit integer type:
        ```python
        ds.add_column("col", types.Int8)
        ds.add_column("idx_col", deeplake.types.Int8(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.Int8(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.Int8("Inverted"))
        ```
    """
    ...

def Sequence(nested_type: DataType | str | Type) -> Type:
    """
    Creates a sequence type that represents an ordered list of other data types.

    A sequence maintains the order of its values, making it suitable for time-series
    data like videos (sequences of images).

    Parameters:
        nested_type: DataType | str | Type
            The data type of the values in the sequence. Can be any data type,
            not just primitive types.

    Returns:
        Type: A new sequence data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a sequence of images:
        ```python
        ds.add_column("col1", types.Sequence(types.Image(sample_compression="jpg")))
        ```
    """

def Image(dtype: DataType | str = "uint8", sample_compression: str = "png") -> Type:
    """
    An image of a given format. The value returned will be a multidimensional array of values rather than the raw image bytes.

    **Available sample_compressions:**

    - png (default)
    - jpg / jpeg

    Parameters:
        dtype: The data type of the array elements to return
        sample_compression: The on-disk compression/format of the image

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.Image)
        ds.add_column("col2", types.Image(sample_compression="jpg"))
        ```
    """
    ...

def Link(type: DataType | Type) -> Type:
    """
    A link to an external resource. The value returned will be a reference to the external resource rather than the raw data.

    Link only supports the Bytes DataType and the Image, SegmentMask, Medical, and Audio Types.

    Parameters:
        type: The type of the linked data. Must be the Bytes DataType or one of the following Types: Image, SegmentMask, Medical, or Audio.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.Link(types.Image()))
        ```
    """
    ...

def Polygon() -> Type:
    """
    Polygon datatype for storing polygons with ability to visualize them.
    <!-- test-context
    ```python
    import deeplake
    from deeplake import types
    import numpy as np
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", deeplake.types.Polygon())
        poly1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        poly2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ds.append({"col1": [[poly1, poly2], [poly1, poly2]]})
        print(ds[0]["col1"])
        # Output: [[[1. 2.]
        #          [3. 4.]
        #          [5. 6.]]

        #         [[1. 2.]
        #          [3. 4.]
        #          [5. 6.]]]
        print(ds[1]["col1"])
        # Output: [[[1. 2.]
        #          [3. 4.]
        #          [5. 6.]]
        #         [[1. 2.]
        #          [3. 4.]
        #          [5. 6.]]]

        ```
    """
    ...

def Point(dimensions: int = 2) -> Type:
    """
    Point datatype for storing points with ability to visualize them.

    Parameters:
        dimensions: The dimension of the point. For example, 2 for 2D points, 3 for 3D points, etc.: defaults to "2"

    <!-- test-context
    ```python
    import deeplake
    from deeplake import types

    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.Point())
        ds.append([{"col1": [[1.0, 2.0], [0.0, 1.0]]}])
        ```
    """
    ...

def ClassLabel(dtype: DataType | str) -> Type:
    """
    Stores categorical labels as numerical values with a mapping to class names.

    ClassLabel is designed for classification tasks where you want to store labels
    as efficient numerical indices while maintaining human-readable class names.
    The class names are stored in the column's metadata under the key "class_names",
    and the actual data contains numerical indices pointing to these class names.

    Parameters:
        dtype: DataType | str
            The datatype for storing the numerical class indices.
            Common choices are "uint8", "uint16", "uint32" or their DataType equivalents.
            Choose based on the number of classes you have.

    How it works:
        1. Define a column with ClassLabel type
        2. Set the "class_names" in the column's metadata as a list of strings
        3. Store numerical indices (0, 1, 2, ...) that map to the class names
        4. When reading, you can use the metadata to convert indices back to class names

    <!--
    ```python
    import deeplake
    from deeplake import types
    import numpy as np
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Basic usage with class labels:
        ```python
        # Create a column for object categories
        ds.add_column("categories", types.ClassLabel(types.Array("uint32", 1)))

        # Define the class names in metadata
        ds["categories"].metadata["class_names"] = ["person", "car", "dog", "cat"]

        # Store numerical indices corresponding to class names
        # 0 = "person", 1 = "car", 2 = "dog", 3 = "cat"
        ds.append({
            "categories": [np.array([0, 1], dtype="uint32")]  # person and car
        })
        ds.append({
            "categories": [np.array([2, 3], dtype="uint32")]  # dog and cat
        })

        # Access the numerical values
        print(ds[0]["categories"])  # Output: [0 1]

        # Get the class names from metadata
        class_names = ds["categories"].metadata["class_names"]
        indices = ds[0]["categories"]
        labels = [class_names[i] for i in indices]
        print(labels)  # Output: ['person', 'car']
        ```

        Advanced usage from COCO ingestion pattern:
        ```python
        # This example shows the pattern used in COCO dataset ingestion
        # where you have multiple annotation groups

        # Create dataset
        ds = deeplake.create("tmp://")

        # Add category columns with ClassLabel type
        ds.add_column("categories", types.ClassLabel(types.Array("uint32", 1)))
        ds.add_column("super_categories", types.ClassLabel(types.Array("uint32", 1)))

        # Set class names from COCO categories
        ds["categories"].metadata["class_names"] = [
            "person", "bicycle", "car", "motorcycle", "airplane"
        ]
        ds["super_categories"].metadata["class_names"] = [
            "person", "vehicle", "animal"
        ]

        # Ingest data with numerical indices
        # Categories: [0, 2, 1] maps to ["person", "car", "bicycle"]
        # Super categories: [0, 1, 1] maps to ["person", "vehicle", "vehicle"]
        ds.append({
            "categories": [np.array([0, 2, 1], dtype="uint32")],
            "super_categories": [np.array([0, 1, 1], dtype="uint32")]
        })
        ```

        Using different data types for different numbers of classes:
        ```python
        # For datasets with fewer than 256 classes, use uint8
        ds.add_column("small_set", types.ClassLabel(types.Array("uint8", 1)))
        ds["small_set"].metadata["class_names"] = ["class_a", "class_b"]

        # For datasets with more classes, use uint16 or uint32
        ds.add_column("large_set", types.ClassLabel(types.Array("uint32", 1)))
        ds["large_set"].metadata["class_names"] = [f"class_{i}" for i in range(1000)]
        ```
    """
    ...
def BoundingBox(
    dtype: DataType | str = "float32",
    format: str | None = None,
    bbox_type: str | None = None,
) -> Type:
    """
    Stores an array of values specifying the bounding boxes of an image.

    Parameters:
        dtype: The datatype of values (default float32)
        format: The bounding box format. Possible values: `ccwh`, `ltwh`, `ltrb`, `unknown`
        bbox_type: The pixel type. Possible values: `pixel`, `fractional`

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.BoundingBox())
        ds.add_column("col2", types.BoundingBox(format="ltwh"))
        ```
    """
    ...

def BinaryMask(
    sample_compression: str | None = None, chunk_compression: str | None = None
) -> Type:
    """
    In binary mask, pixel value is a boolean for whether there is/is-not an object of a class present.

    NOTE: Since binary masks often contain large amounts of data, it is recommended to compress them using lz4.

    Parameters:
        sample_compression:
            How to compress each row's value.
            supported values are ``lz4``, ``zstd``, and ``null`` (no compression).
        chunk_compression:
            Defines the compression algorithm for on-disk storage of mask data.
            supported values are ``lz4``, ``zstd``, and ``null`` (no compression).

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.BinaryMask(sample_compression="lz4"))
        ds.append([{"col1": np.zeros((512, 512, 5), dtype="bool")}])
        ```
    """
    ...

def SegmentMask(
    dtype: DataType | str = "uint8",
    sample_compression: str | None = None,
    chunk_compression: str | None = None,
) -> Type:
    """
    Segmentation masks are 2D representations of class labels where a numerical class value is encoded in an array of same shape as the image.

    NOTE: Since segmentation masks often contain large amounts of data, it is recommended to compress them using lz4.

    Parameters:
        sample_compression:
            How to compress each row's value.
            supported values are ``lz4``, ``zstd``, and ``null`` (no compression).
        chunk_compression:
            Defines the compression algorithm for on-disk storage of mask data.
            supported values are ``lz4``, ``zstd``, ``png``, ``nii``, ``nii.gz``, and ``null`` (no compression).

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.SegmentMask(sample_compression="lz4"))
        ds.append([{"col1": np.zeros((512, 512, 3))}])
        ```
    """
    ...

def Video(compression: str = "mp4") -> Type:
    """
    Video datatype for storing videos.

    Parameters:
        compression: The compression format. Only H264 codec is supported at the moment.

    <!--
    ```python
    import deeplake
    from io import BytesIO
    from deeplake import types
    from inspect import Signature, Parameter
    from functools import wraps
    ds = deeplake.create("tmp://")

    def __open(*args, **kwargs):
        return BytesIO(b"")

    # Extract the original open signature
    original_signature = Signature(
        parameters=[
            Parameter("file", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("mode", Parameter.POSITIONAL_OR_KEYWORD, default="r"),
            Parameter("buffering", Parameter.POSITIONAL_OR_KEYWORD, default=-1),
            Parameter("encoding", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("errors", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("newline", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("closefd", Parameter.POSITIONAL_OR_KEYWORD, default=True),
            Parameter("opener", Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
    )

    @wraps(__open)
    def new_open(*args, **kwargs):
        return __open(*args, **kwargs)

    new_open.__signature__ = original_signature

    open = new_open
    ```
    -->

    Examples:
        ```python
        ds.add_column("video", types.Video(compression="mp4"))

        with open("path/to/video.mp4", "rb") as f:
            bytes_data = f.read()
            ds.append([{"video": bytes_data}])
        ```
    """

def Medical(compression: str) -> Type:
    """
    Medical datatype for storing medical images.

    **Available compressions:**

    - nii
    - nii.gz
    - dcm

    <!-- test-context
    ```python
    import deeplake
    from io import BytesIO
    from deeplake import types
    from inspect import Signature, Parameter
    from functools import wraps

    ds = deeplake.create("tmp://")

    def __open(*args, **kwargs):
        return BytesIO(b"")

    # Extract the original open signature
    original_signature = Signature(
        parameters=[
            Parameter("file", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("mode", Parameter.POSITIONAL_OR_KEYWORD, default="r"),
            Parameter("buffering", Parameter.POSITIONAL_OR_KEYWORD, default=-1),
            Parameter("encoding", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("errors", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("newline", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("closefd", Parameter.POSITIONAL_OR_KEYWORD, default=True),
            Parameter("opener", Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
    )

    @wraps(__open)
    def new_open(*args, **kwargs):
        return __open(*args, **kwargs)

    new_open.__signature__ = original_signature

    open = new_open

    ```
    -->

    Parameters:
        compression: How to compress each row's value. Possible values: ``dcm``, ``nii``, ``nii.gz``
    Examples:
        ```python
        ds.add_column("col1", types.Medical(compression="dcm"))

        with open("path/to/dicom/file.dcm", "rb") as f:
            bytes_data = f.read()
            ds.append([{"col1": bytes_data}])
        ```
    """
    ...

def Mesh() -> Type:
    """
    Mesh datatype for storing 3D meshes.

    **Available compressions:**

    - ply
    - stl

    <!-- test-context
    ```python
    import deeplake
    from io import BytesIO
    from deeplake import types
    from inspect import Signature, Parameter
    from functools import wraps
    ds = deeplake.create("tmp://")
    def __open(*args, **kwargs):
        return BytesIO(b"ply")
    # Extract the original open signature
    original_signature = Signature(
        parameters=[
            Parameter("file", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("mode", Parameter.POSITIONAL_OR_KEYWORD, default="r"),
            Parameter("buffering", Parameter.POSITIONAL_OR_KEYWORD, default=-1),
            Parameter("encoding", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("errors", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("newline", Parameter.POSITIONAL_OR_KEYWORD, default=None),
            Parameter("closefd", Parameter.POSITIONAL_OR_KEYWORD, default=True),
            Parameter("opener", Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
    )
    @wraps(__open)
    def new_open(*args, **kwargs):
        return __open(*args, **kwargs)
    new_open.__signature__ = original_signature
    open = new_open
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.Mesh())
        with open("path/to/mesh/file.stl", "rb") as f:
            bytes_data = f.read()
            ds.append([{"col1": bytes_data}])
        ```
    """
    ...

def Struct(fields: dict[str, DataType | str | Type]) -> Type:
    """
    Defines a custom datatype with specified keys.

    See [deeplake.types.Dict][] for a type that supports different key/value pairs per value.

    Parameters:
        fields: A dict where the key is the name of the field, and the value is the datatype definition for it

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.Struct({
           "field1": types.Int16(),
           "field2": "text",
        }))

        ds.append([{"col1": {"field1": 3, "field2": "a"}}])
        print(ds[0]["col1"]["field1"]) # Output: 3
        ```
    """
    ...


def UInt16(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates an unsigned 16-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new unsigned 16-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col", types.UInt16)
        ds.add_column("idx_col", deeplake.types.UInt16(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.UInt16(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.UInt16("Inverted"))
        ```
    """
    ...


def UInt32(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates an unsigned 32-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new unsigned 32-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col", types.UInt32)
        ds.add_column("idx_col", deeplake.types.UInt32(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.UInt32(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.UInt32("Inverted"))
        ```
    """
    ...


def UInt64(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates an unsigned 64-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new unsigned 64-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.UInt64)
        ds.add_column("idx_col", deeplake.types.UInt64(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.UInt64(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.UInt64("Inverted"))
        ```
    """

def UInt8(index_type: str | IndexType | NumericIndex | None = None) -> DataType | Type:
    """
    Creates an unsigned 8-bit integer value type.

    Parameters:
        index_type: str | IndexType | NumericIndex | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`

            Default is ``None`` meaning "do not index"

    Returns:
        DataType | Type: A new unsigned 8-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col", types.UInt8)
        ds.add_column("idx_col", deeplake.types.UInt8(deeplake.types.NumericIndex(deeplake.types.Inverted)))
        ds.add_column("idx_col_1", deeplake.types.UInt8(deeplake.types.Inverted))
        ds.add_column("idx_col_2", deeplake.types.UInt8("Inverted"))
        ```
    """
    ...
    
