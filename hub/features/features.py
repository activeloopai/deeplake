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


class FeatureConnector:
    """ Base class for all datatypes"""

    def _flatten(self) -> Iterable[FlatTensor]:
        """ Flattens dtype into list of tensors that will need to be stored seperately """
        raise NotImplementedError()


def featurify(feature) -> FeatureConnector:
    """This functions converts naked primitive datatypes and ditcs into Primitives and FeatureDicts
    That way every node in dtype tree is a FeatureConnector type object
    """
    if isinstance(feature, dict):
        return FeatureDict(feature)
    elif isinstance(feature, FeatureConnector):
        return feature
    else:
        return Primitive(feature)


class Primitive(FeatureConnector):
    """Class for handling primitive datatypes
    All numpy primitive data types like int32, float64, etc... should be wrapped around this class
    """

    def __init__(self, dtype, chunks=True):
        self._dtype = hub.dtype(dtype)
        self.chunks = chunks

    def _flatten(self):
        yield FlatTensor("", (), self._dtype, (), self.chunks)


class FeatureDict(FeatureConnector):
    """Class for dict branching of a datatype
    FeatureDict dtype contains str -> dtype associations
    This way you can describe complex datatypes
    """

    def __init__(self, dict_):
        self.dict_: Dict[str, FeatureConnector] = {
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


class Tensor(FeatureConnector):
    """Tensor type in features
    Has np-array like structure contains any type of elements (Primitive and non-Primitive)
    """

    def __init__(self, shape: Shape, dtype, max_shape: Shape = None, chunks=True):
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
            Describes how to split tensor dimentions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimentions (first dimention)
            If default value is chosen, automatically detects how to split into chunks
        """
        self.shape = tuple(shape)
        self.dtype = featurify(dtype)
        self.max_shape = tuple(max_shape or shape)
        self.chunks = chunks

    def _flatten(self):
        for item in self.dtype._flatten():
            yield FlatTensor(
                item.path,
                self.shape + item.shape,
                item.dtype,
                self.max_shape + item.max_shape,
                # FIXME get chunks=None and write line below for that case
                self.chunks if self.chunks is not True else item.chunks,
            )
