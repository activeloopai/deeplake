from typing import Tuple, Dict, Iterable

import hub

Shape = Tuple[int, ...]


class FlatTensor:
    def __init__(self, path: str, shape: Shape, dtype, max_shape: Shape, chunks: Shape):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.max_shape = max_shape
        self.chunks = chunks


class HubFeature:
    def _flatten(self) -> Iterable[FlatTensor]:
        raise NotImplementedError()


def featurify(feature) -> HubFeature:
    if isinstance(feature, dict):
        return FeatureDict(feature)
    elif isinstance(feature, HubFeature):
        return feature
    else:
        return Primitive(feature)


class Primitive(HubFeature):
    def __init__(self, dtype, chunks=True, compressor="lz4"):
        self._dtype = hub.dtype(dtype)
        self.chunks = chunks
        self.shape = self.max_shape = ()
        self.dtype = self._dtype
        self.compressor = compressor

    def _flatten(self):
        yield FlatTensor("", (), self._dtype, (), self.chunks)


class FeatureDict(HubFeature):
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


class Tensor(HubFeature):
    def __init__(
        self,
        shape: Shape = (None,),
        dtype="float64",
        max_shape: Shape = None,
        chunks=None,
        compressor="lz4",
    ):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        chunks = (chunks,) if isinstance(chunks, int) else tuple(shape)
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
