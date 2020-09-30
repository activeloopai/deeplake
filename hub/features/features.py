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


class FeatureConnector:
    def _flatten(self) -> Iterable[FlatTensor]:
        raise NotImplementedError()


def featurify(feature) -> FeatureConnector:
    if isinstance(feature, dict):
        return FeatureDict(feature)
    elif isinstance(feature, FeatureConnector):
        return feature
    else:
        return Primitive(feature)


class Primitive(FeatureConnector):
    def __init__(self, dtype, chunks=True):
        self._dtype = hub.dtype(dtype)
        self.chunks = chunks

    def _flatten(self):
        yield FlatTensor("", (), self._dtype, (), self.chunks)


class FeatureDict(FeatureConnector):
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
    def __init__(self, shape: Shape, dtype, max_shape: Shape = None, chunks=True):
        self.shape = shape
        self.dtype = featurify(dtype)
        self.max_shape = max_shape or shape
        self.chunks = chunks

    def _flatten(self):
        for item in self.dtype._flatten():
            yield FlatTensor(
                item.path,
                self.shape + item.shape,
                item.dtype,
                self.max_shape + item.max_shape,
                item.chunks,
            )
