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


class HubFeature:
    """ Base class for all datatypes"""

    def _flatten(self) -> Iterable[FlatTensor]:
        """ Flattens dtype into list of tensors that will need to be stored seperately """
        raise NotImplementedError()


def featurify(feature) -> HubFeature:
    """This functions converts naked primitive datatypes and ditcs into Primitives and FeatureDicts
    That way every node in dtype tree is a FeatureConnector type object
    """
    if isinstance(feature, dict):
        return FeatureDict(feature)
    elif isinstance(feature, HubFeature):
        return feature
    else:
        return Primitive(feature)


class Primitive(HubFeature):
    """Class for handling primitive datatypes
    All numpy primitive data types like int32, float64, etc... should be wrapped around this class
    """

    def __init__(self, dtype, chunks=None, compressor="lz4"):
        self._dtype = hub.dtype(dtype)
        self.chunks = _normalize_chunks(chunks)
        self.shape = self.max_shape = ()
        self.dtype = self._dtype
        self.compressor = compressor

    def _flatten(self):
        yield FlatTensor("", (), self._dtype, (), self.chunks)


class FeatureDict(HubFeature):
    """Class for dict branching of a datatype
    FeatureDict dtype contains str -> dtype associations
    This way you can describe complex datatypes
    """

    def __init__(self, dict_):
        self.dict_: Dict[str, HubFeature] = {
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


def _normalize_chunks(chunks):
    chunks = (chunks,) if isinstance(chunks, int) else chunks
    chunks = tuple(chunks) if chunks else None
    return chunks


class Tensor(HubFeature):
    """Tensor type in features
    Has np-array like structure contains any type of elements (Primitive and non-Primitive)
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
        dtype : FeatureConnector or str
            dtype of each element in Tensor. Can be Primitive and non-Primitive type
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        chunks = _normalize_chunks(chunks)
        max_shape = max_shape or shape
        if len(shape) != len(max_shape):
            raise ValueError(
                f"Length of shape ({len(shape)}) and max_shape ({len(max_shape)}) does not match"
            )
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


def flatten(dtype, root=""):
    """ Flattens nested dictionary and returns tuple (dtype, path) """
    dtype = featurify(dtype)
    if isinstance(dtype, FeatureDict):
        for key, value in dtype.dict_.items():
            yield from flatten(value, root + "/" + key)
    else:
        yield (dtype, root)
