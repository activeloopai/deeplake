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
    "EmbeddingIndexEnumType",
    "EmbeddingIndexType",
    "EmbeddingsMatrixIndex",
    "EmbeddingsMatrixIndexType",
    "Exact",
    "Float16",
    "Float32",
    "Float64",
    "Image",
    "IndexType",
    "Int16",
    "Int32",
    "Int64",
    "Int8",
    "Inverted",
    "Link",
    "Medical",
    "NumericIndex",
    "NumericIndexEnumType",
    "NumericIndexType",
    "Point",
    "Polygon",
    "QuantizationType",
    "SegmentMask",
    "Sequence",
    "Struct",
    "Text",
    "TextIndex",
    "TextIndexEnumType",
    "TextIndexType",
    "Type",
    "TypeKind",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt8",
    "Video"]

class EmbeddingIndexEnumType:
    """
    Enumeration of available index types for embedding.

    Members:

      Clustered:
            Clusters embeddings in the index to speed up search. This is the default index type.

      ClusteredQuantized:
            Stores a binary quantized representation of the original embedding in the index
            rather than a full copy of the embedding. This slightly decreases accuracy of
            searches, while significantly improving query time.
    """
    Clustered: typing.ClassVar[EmbeddingIndexEnumType]
    ClusteredQuantized: typing.ClassVar[EmbeddingIndexEnumType]
    __members__: typing.ClassVar[dict[str, EmbeddingIndexEnumType]]
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
        ...


class EmbeddingIndexType:
    """
    Represents embedding index type.
    """
    def __init__(self, type: EmbeddingIndexEnumType) -> None:
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

class TextIndexEnumType:
    """
    Enumeration of available text indexing types.

    Members:
        Inverted:
            A text index that supports keyword lookup. Can be used with ``CONTAINS(column, 'wanted_value')``.
        BM25:
            A BM25-based index of text data. Can be used with ``BM25_SIMILARITY(column, 'search text')``
            in a TQL ``ORDER BY`` clause.
    """

    BM25: typing.ClassVar[TextIndexType]  # value = <deeplake._deeplake.types.TextIndexType object>
    Inverted: typing.ClassVar[TextIndexType]  # value = <deeplake._deeplake.types.TextIndexType object>
    Exact: typing.ClassVar[TextIndexType]  # value = <deeplake._deeplake.types.TextIndexType object>
    __members__: typing.ClassVar[dict[str, TextIndexEnumType]]  # value = {'Inverted': <TextIndexEnumType.Inverted: 1>, 'BM25': <TextIndexEnumType.BM25: 2>, 'Exact': <TextIndexEnumType.Exact: 3>}
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


class NumericIndexEnumType:
    """
    Enumeration of available numeric indexing types.

    Members:
        Inverted:
            A numeric index that supports keyword lookup. Can be used with ``CONTAINS(column, 'wanted_value')``.
    """

    Inverted: typing.ClassVar[NumericIndexType]

    __members__: typing.ClassVar[dict[str, NumericIndexEnumType]]
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
            int: The integer value of the numeric index type.
        """
        ...


class NumericIndexType:
    def __init__(self, type: NumericIndexEnumType) -> None:
        ...

class TextIndexType:
    def __init__(self, type: TextIndexEnumType) -> None:
        ...

class EmbeddingsMatrixIndexType:
    def __init__(self) -> None:
        ...

class IndexType:
    """
    Represents all available index types in the deeplake.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: IndexType) -> bool:
        ...
    def __init__(self, index_type: TextIndexType | EmbeddingIndexType | EmbeddingsMatrixIndexType | NumericIndexType) -> None:
        ...
    def __ne__(self, other: IndexType) -> bool:
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

def Text(index_type: str | TextIndexEnumType | TextIndexType | None = None) -> Type:
    """
    Creates a text data type of arbitrary length.

    Parameters:
        index_type: str | TextIndexEnumType | TextIndexType | None
            How to index the data in the column for faster searching.
            Options are:

            - :class:`deeplake.types.Inverted`
            - :class:`deeplake.types.BM25`

            Default is ``None`` meaning "do not index"

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

def TextIndex(type: TextIndexEnumType) -> TextIndexType:
    """
    Creates a text index.

    Parameters:
        type: TextIndexEnumType

    Returns:
        Type: Text index type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create text columns with different text index types:
        ```python
        ds.add_column("col1", types.Text)
        ds.add_column("col2", types.Text)
        ds["col1"].create_index(types.TextIndex(types.Inverted))
        ds["col2"].create_index(types.TextIndex(types.BM25))
        ```
    """
    ...

BM25: TextIndexEnumType.BM25
"""
A BM25-based index of text data.

This index can be used with ``BM25_SIMILARITY(column, 'search text')`` in a TQL ``ORDER BY`` clause.

See Also:
    `BM25 Algorithm <https://en.wikipedia.org/wiki/Okapi_BM25>`_
"""

Inverted: TextIndexEnumType.Inverted
"""
A text index that supports keyword lookup.

This index can be used with ``CONTAINS(column, 'wanted_value')``.
"""

Exact: TextIndexEnumType.Exact
"""
A text index that supports whole text lookup.

This index can be used with ``EQUALS(column, 'wanted_value')``.
"""

def Dict() -> Type:
    """
    Creates a type that supports storing arbitrary key/value pairs in each row.

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
        quantization: QuantizationType | None
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

def EmbeddingIndex(type: EmbeddingIndexEnumType | None = None) -> EmbeddingIndexType:
    """
    Creates an embedding index.

    Parameters:
        type: EmbeddingIndexEnumType | None = None
            The enumeration type of embedding index

    Returns:
        Type: EmbeddingIndexType.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create embedding columns:
        ```python
        ds.add_column("col", types.Embedding(768, index_type=types.EmbeddingIndex(types.ClusteredQuantized)))
        ```
    """
    ...

def EmbeddingsMatrixIndex() -> EmbeddingsMatrixIndexType:
    """
    Creates an embeddings matrix index.
    """
    ...

def Float16() -> DataType:
    """
    Creates a 16-bit (half) float value type.

    Returns:
        DataType: A new 16-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 16-bit float type:
        ```python
        ds.add_column("col1", types.Float16)
        ```
    """
    ...

def Float32() -> DataType:
    """
    Creates a 32-bit float value type.

    Returns:
        DataType: A new 32-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 32-bit float type:
        ```python
        ds.add_column("col1", types.Float32)
        ```
    """
    ...

def Float64() -> DataType:
    """
    Creates a 64-bit float value type.

    Returns:
        DataType: A new 64-bit float data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 64-bit float type:
        ```python
        ds.add_column("col1", types.Float64)
        ```
    """
    ...

def Int16() -> DataType:
    """
    Creates a 16-bit integer value type.

    Returns:
        DataType: A new 16-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 16-bit integer type:
        ```python
        ds.add_column("col1", types.Int16)
        ```
    """
    ...

def Int32() -> DataType:
    """
    Creates a 32-bit integer value type.

    Returns:
        DataType: A new 32-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 32-bit integer type:
        ```python
        ds.add_column("col1", types.Int32)
        ```
    """
    ...

def Int64() -> DataType:
    """
    Creates a 64-bit integer value type.

    Returns:
        DataType: A new 64-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 64-bit integer type:
        ```python
        ds.add_column("col1", types.Int64)
        ```
    """
    ...

def Int8() -> DataType:
    """
    Creates an 8-bit integer value type.

    Returns:
        DataType: A new 8-bit integer data type.

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        Create a column with 8-bit integer type:
        ```python
        ds.add_column("col1", types.Int8)
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

    **Available formats:**

    - png (default)
    - apng
    - jpg / jpeg
    - tiff / tif
    - jpeg2000 / jp2
    - bmp
    - nii
    - nii.gz
    - dcm

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

def Link(type: Type) -> Type:
    """
    A link to an external resource. The value returned will be a reference to the external resource rather than the raw data.

    Parameters:
        type: The type of the linked data

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

def ClassLabel(dtype: DataType | str) -> Type: ...
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
        sample_compression: How to compress each row's value. Possible values: lz4, null (default: null)
        chunk_compression: How to compress all the values stored in a single file. Possible values: lz4, null (default: null)

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
        sample_compression: How to compress each row's value. Possible values: lz4, null (default: null)
        chunk_compression: How to compress all the values stored in a single file. Possible values: lz4, null (default: null)

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
        compression: How to compress each row's value. Possible values: dcm, nii, nii.gz
    Examples:
        ```python
        ds.add_column("col1", types.Medical(compression="dcm"))

        with open("path/to/dicom/file.dcm", "rb") as f:
            bytes_data = f.read()
            ds.append([{"col1": bytes_data}])
        ```
    """
    ...


def NumericIndex(type: NumericIndexEnumType) -> NumericIndexType:
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

def UInt16() -> DataType:
    """
    An unsigned 16-bit integer value

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.UInt16)
        ```
    """
    ...

def UInt32() -> DataType:
    """
    An unsigned 32-bit integer value

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.UInt16)
        ```
    """
    ...

def UInt64() -> DataType:
    """
    An unsigned 64-bit integer value

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.UInt64)
        ```
    """
    ...

def UInt8() -> DataType:
    """
    An unsigned 8-bit integer value

    <!--
    ```python
    ds = deeplake.create("tmp://")
    ```
    -->

    Examples:
        ```python
        ds.add_column("col1", types.UInt16)
        ```
    """
    ...
