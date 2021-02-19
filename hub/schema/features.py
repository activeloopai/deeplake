"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple, Dict, Iterable
import hub

Shape = Tuple[int, ...]


class FlatTensor:
    """ Tensor metadata after applying flatten function """

    def __init__(self, path: str, shape: Shape, dtype, max_shape: Shape, chunks: Shape):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.max_shape = max_shape
        self.chunks = chunks


class HubSchema:
    """ Base class for all datatypes"""

    def _flatten(self) -> Iterable[FlatTensor]:
        """ Flattens dtype into list of tensors that will need to be stored seperately """
        raise NotImplementedError()


class Primitive(HubSchema):
    """Class for handling primitive datatypes.
    All numpy primitive data types like int32, float64, etc... should be wrapped around this class.
    """

    def __init__(self, dtype, chunks=None, compressor="lz4"):
        self._dtype = hub.dtype(dtype)
        self.chunks = _normalize_chunks(chunks)
        self.shape = self.max_shape = ()
        self.dtype = self._dtype
        self.compressor = compressor

    def _flatten(self):
        yield FlatTensor("", (), self._dtype, (), self.chunks)

    def __str__(self):
        return "'" + str(self.dtype) + "'"

    def __repr__(self):
        return self.__str__()


class SchemaDict(HubSchema):
    """Class for dict branching of a datatype.
    SchemaDict dtype contains str -> dtype associations.
    This way you can describe complex datatypes.
    """

    def __init__(self, dict_):
        self.dict_: Dict[str, HubSchema] = {
            key: featurify(value) for key, value in dict_.items()
        }

    def _flatten(self):
        for key, value in self.dict_.items():
            for item in value._flatten():
                yield FlatTensor(
                    f"/{key}{item.path}",
                    item.shape,
                    item.dtype,
                    item.max_shape,
                    item.chunks,
                )

    def __str__(self):
        out = "SchemaDict("
        out += str(self.dict_)
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()


def featurify(schema) -> HubSchema:
    """This functions converts naked primitive datatypes and ditcs into Primitives and SchemaDicts.
    That way every node in dtype tree is a SchemaConnector type object.
    """
    if isinstance(schema, dict):
        return SchemaDict(schema)
    elif isinstance(schema, HubSchema):
        return schema
    else:
        return Primitive(schema)


def _normalize_chunks(chunks):
    chunks = (chunks,) if isinstance(chunks, int) else chunks
    chunks = tuple(chunks) if chunks else None
    return chunks


class Tensor(HubSchema):
    """Tensor type in schema.
    Has np-array like structure contains any type of elements (Primitive and non-Primitive).
    Tensors can't be visualized at app.activeloop.ai.
    """

    def __init__(
        self,
        shape: Shape = (None,),
        dtype="float64",
        max_shape: Shape = None,
        chunks=None,
        compressor="lz4",
    ):
        """
        Parameters
        ----------
        shape : Tuple[int]
            Shape of tensor, can contains None(s) meaning the shape can be dynamic
            Dynamic shape means it can change during editing the dataset
        dtype : SchemaConnector or str
            dtype of each element in Tensor. Can be Primitive and non-Primitive type
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        if shape is None:
            raise TypeError("shape cannot be None")
        if isinstance(shape, Iterable) and None in shape and max_shape is None:
            raise ValueError(
                "while specifying shape containing None dimensions, max_shape argument needs to be provided"
            )
        if not isinstance(shape, (tuple, int, list)):
            raise TypeError(f"shape of {type(shape)} is not supported")

        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        for dim in shape:
            if not isinstance(dim, int) and dim is not None:
                raise TypeError(f"shape can't have {type(dim)} in its dimension")
        max_shape = shape if max_shape is None else max_shape
        if not isinstance(max_shape, (tuple, int, list)):
            raise TypeError(f"max_shape of {type(max_shape)} is not supported")
        max_shape = (max_shape,) if isinstance(max_shape, int) else tuple(max_shape)
        for dim in max_shape:
            if dim is None:
                raise TypeError("max_shape can't have None in it's dimension")
            elif not isinstance(dim, int):
                raise TypeError(f"max_shape can't have {type(dim)} in its dimension")

        if len(shape) != len(max_shape):
            raise ValueError(
                f"shape {shape} and max_shape {max_shape} have different lengths"
            )
        for dim, max_dim in zip(shape, max_shape):
            if dim is not None and dim != max_dim:
                raise ValueError(f"shape and max_shape mismatch, {dim} != {max_dim}")

        chunks = _normalize_chunks(chunks)

        # TODO add errors if shape and max_shape have wrong values
        self.shape = tuple(shape)
        self.dtype = featurify(dtype)
        self.max_shape = max_shape
        self.chunks = chunks
        self.compressor = compressor

    def _flatten(self):
        for item in self.dtype._flatten():
            yield FlatTensor(
                item.path,
                self.shape + item.shape,
                item.dtype,
                self.max_shape + item.max_shape,
                self.chunks or item.chunks,
            )

    def __str__(self):
        out = "Tensor(shape=" + str(self.shape) + ", dtype=" + str(self.dtype)
        out = (
            out + ", max_shape=" + str(self.max_shape)
            if self.max_shape != self.shape
            else out
        )
        out = out + ", chunks=" + str(self.chunks) if self.chunks is not None else out
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()


def flatten(dtype, root=""):
    """ Flattens nested dictionary and returns tuple (dtype, path) """
    dtype = featurify(dtype)
    if isinstance(dtype, SchemaDict):
        for key, value in dtype.dict_.items():
            yield from flatten(value, root + "/" + key)
    else:
        yield (dtype, root)
