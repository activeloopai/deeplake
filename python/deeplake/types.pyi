from __future__ import annotations

import typing

__all__ = [
    "Array",
    "BM25",
    "Bool",
    "DataType",
    "Dict",
    "Embedding",
    "Float32",
    "Float64",
    "Type",
    "Int16",
    "Int32",
    "Int64",
    "Int8",
    "Inverted",
    "Sequence",
    "Image",
    "Struct",
    "Text",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt8",
    "BoundingBox",
    "BinaryMask",
    "SegmentMask",
    "TypeKind",
    "TextIndexType",
    "QuantizationType",
    "Binary",
]


class QuantizationType:
    Binary: typing.ClassVar[QuantizationType]
    """
    Stores a binary quantized representation of the original embedding in the index rather than the a full copy of the embedding.
    
    This slightly decreases accuracy of searches, while significantly improving query time.   
    """

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
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

Binary: QuantizationType

class TextIndexType:
    """
    Members:

      Inverted

      BM25
    """

    BM25: typing.ClassVar[TextIndexType]
    Inverted: typing.ClassVar[TextIndexType]
    __members__: typing.ClassVar[dict[str, TextIndexType]]

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
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class DataType:
    """
    The base class all specific types extend from.
    """

    def __eq__(self, other: DataType) -> bool:
        ...

    def __ne__(self, other: DataType) -> bool:
        ...

    def __repr__(self) -> str: ...


class Type:
    """ """

    def __repr__(self) -> str: ...

    def __eq__(self, other: Type) -> bool:
        ...

    def __ne__(self, other: Type) -> bool:
        ...

    @property(readonly=True)
    def data_type(self) -> DataType: ...

    @property(readonly=True)
    # Temporary workaround. Need to remove `deeplake._deeplake` from the return type.
    def default_format(self) -> deeplake._deeplake.formats.DataFormat: ...

    @property
    def id(self) -> str:
        """
        The id (name) of the data type
        """
        ...

    @property
    def is_sequence(self) -> bool:
        ...

    @property
    def kind(self) -> TypeKind:
        ...

    @property
    def shape(self) -> list[int] | None:
        """
        The shape of the data type if applicable. Otherwise none
        """
        ...


class TypeKind:
    """
    Members:
    
      Generic
    
      Text
    
      Dict
    
      Embedding
    
      Sequence
    
      Image
    
      BoundingBox
    
      BinaryMask
    
      SegmentMask
    """
    BinaryMask: typing.ClassVar[TypeKind]
    BoundingBox: typing.ClassVar[TypeKind]
    Dict: typing.ClassVar[TypeKind]
    Embedding: typing.ClassVar[TypeKind]
    Generic: typing.ClassVar[TypeKind]
    Image: typing.ClassVar[TypeKind]
    SegmentMask: typing.ClassVar[TypeKind]
    Sequence: typing.ClassVar[TypeKind]
    Text: typing.ClassVar[TypeKind]
    __members__: typing.ClassVar[dict[str, TypeKind]]
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


@typing.overload
def Array(dtype: DataType | str, dimensions: int) -> DataType: ...


@typing.overload
def Array(dtype: DataType | str, shape: list[int]) -> DataType: ...


def Array(dtype: DataType | str, dimensions: int, shape: list[int]) -> DataType:
    """
    A generic array of data.

    Parameters:
        dtype: The datatype of values in the array
        dimensions: The number of dimensions/axies in the array. Unlike specifying `shape`, there is no constraint on the size of each dimension.
        shape: Constrain the size of each dimension in the array

    Examples:
        >>> # Create a three-dimensional array, where each dimension can have any number of elements
        >>> ds.add_column("col1", types.Array("int32", dimensions=3))
        >>>
        >>> # Create a three-dimensional array, where each dimension has a known size
        >>> ds.add_column("col2", types.Array(types.Float32(), shape=[50, 30, 768]))
    """
    ...


def Bool() -> DataType:
    """
    A boolean value

    Examples:
        >>> ds.add_column("col1", types.Bool)
        >>> ds.add_column("col2", "bool")
    """
    ...


def Text(index_type: str | TextIndexType | None = None) -> Type:
    """
    Text data of arbitrary length.

    Options for index_type are:

    - [deeplake.types.Inverted][]
    - [deeplake.types.BM25][]

    Parameters:
        index_type: How to index the data in the column for faster searching. Default is `None` meaning "do not index"

    Examples:
        >>> ds.add_column("col1", types.Text)
        >>> ds.add_column("col2", "text")
        >>> ds.add_column("col3", str)
        >>> ds.add_column("col4", types.Text(index_type=types.Inverted))
        >>> ds.add_column("col4", types.Text(index_type=types.BM25))
    """
    ...


BM25: TextIndexType
"""
A [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) based index of text data.

This index can be used with `BM25_SIMILARITY(column, 'search text')` in a TQL `ORDER BY` clause.
"""

Inverted: TextIndexType
"""
A text index that supports keyword lookup.

This index can be used with `CONTAINS(column, 'wanted_value')`.
"""

def Dict() -> Type:
    """
    Supports storing arbitrary key/value pairs in each row.

    See [deeplake.types.Struct][] for a type that supports defining allowed keys.

    Examples:
        >>> ds.add_column("col1", types.Dict)
        >>>
        >>> ds.append([{"col1", {"a": 1, "b": 2}}])
        >>> ds.append([{"col1", {"b": 3, "c": 4}}])
    """

    ...


def Embedding(size: int, dtype: DataType | str = "float32", quantization: QuantizationType | None = None ) -> Type:
    """
    A single-dimensional embedding of a given length. See [deeplake.types.Array][] for a multidimensional array.

    Parameters:
        size: The size of the embedding
        dtype: The datatype of the embedding. Defaults to float32
        quantization: How to compress the embeddings in the index. Default uses no compression, but can be set to [deeplake.types.QuantizationType.Binary][]

    Examples:
         >>> ds.add_column("col1", types.Embedding(768))
         >>> ds.add_column("col2", types.Embedding(768, quantization=types.QuantizationType.Binary))
    """
    ...


def Float32() -> DataType:
    """
    A 32-bit float value

    Examples:
         >>> ds.add_column("col1", types.Float)
    """
    ...


def Float64() -> DataType:
    """
    A 64-bit float value

    Examples:
         >>> ds.add_column("col1", types.Float64)
    """
    ...


def Int16() -> DataType:
    """
    A 16-bit integer value

    Examples:
         >>> ds.add_column("col1", types.Int16)
    """
    ...


def Int32() -> DataType:
    """
    A 32-bit integer value

    Examples:
         >>> ds.add_column("col1", types.Int32)
    """
    ...


def Int64() -> DataType:
    """
    A 64-bit integer value

    Examples:
         >>> ds.add_column("col1", types.Int64)
    """
    ...


def Int8() -> DataType:
    """
    An 8-bit integer value

    Examples:
         >>> ds.add_column("col1", types.Int8)
    """
    ...


def Sequence(nested_type: DataType | str | Type) -> Type:
    """
    A sequence is a list of other data types, where there is a order to the values in the list.

    For example, a video can be stored as a sequence of images to better capture the time-based ordering of the images rather than simply storing them as an Array

    Parameters:
        nested_type: The data type of the values in the sequence. Can be any data type, not just primitive types.

    Examples:
         >>> ds.add_column("col1", types.Sequence(types.Image(sample_compression="jpeg")))
    """
    ...


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

    Examples:
        >>> ds.add_column("col1", types.Sequence(types.Image))
        >>> ds.add_column("col1", types.Sequence(types.Image(sample_compression="jpg")))
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
        format: The bounding box format. Possible values: `ccwh`, `tlwh`, `tlbr`, `unknown`
        bbox_type: The pixel type. Possible values: `pixel`, `fractional`

    Examples:
        >>> ds.add_column("col1", types.BoundingBox())
        >>> ds.add_column("col2", types.BoundingBox(format="tlwh"))
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

    Examples:
        >>> ds.add_column("col1", types.BinaryMask(sample_compression="lz4"))
        >>> ds.append(np.zeros((512, 512, 5), dtype="bool"))
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

    Examples:
        >>>  ds.add_column("col1", types.SegmentMask(sample_compression="lz4"))
        >>>  ds.append("col1", np.zeros((512, 512)))
    """
    ...


def Struct(fields: dict[str, DataType | str]) -> DataType:
    """
    Defines a custom datatype with specified keys.

    See [deeplake.types.Dict][] for a type that supports different key/value pairs per value.

    Parameters:
        fields: A dict where the key is the name of the field, and the value is the datatype definition for it

    Examples:
        >>> ds.add_column("col1", types.Struct({
        >>>    "field1": types.Int16(),
        >>>    "field2": types.Text(),
        >>> }))
        >>>
        >>> ds.append([{"col1": {"field1": 3, "field2": "a"}}])
        >>> print(ds[0]["col1"]["field1"])


    """
    ...


def UInt16() -> DataType:
    """
    An unsigned 16-bit integer value

    Examples:
         >>> ds.add_column("col1", types.UInt16)
    """
    ...


def UInt32() -> DataType:
    """
    An unsigned 32-bit integer value

    Examples:
         >>> ds.add_column("col1", types.UInt16)
    """
    ...


def UInt64() -> DataType:
    """
    An unsigned 64-bit integer value

    Examples:
         >>> ds.add_column("col1", types.UInt64)
    """
    ...


def UInt8() -> DataType:
    """
    An unsigned 8-bit integer value

    Examples:
         >>> ds.add_column("col1", types.UInt16)
    """
    ...
