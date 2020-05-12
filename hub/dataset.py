import json
from typing import Tuple, Dict
from .storage import Base as Storage
from .array import Array, Props

try:
    from hub.integrations.pytorch import TorchIterableDataset
except:
    pass

try:
    from hub.integrations.tensorflow import HubTensorflowDataset
except Exception as ex:
    pass


class DatasetProps:
    paths: Dict[str, str] = None


class Dataset:
    def __init__(self, path: str, storage: Storage):
        self._path = path
        self._storage = storage
        self._props = DatasetProps()
        self._props.__dict__ = json.loads(storage.get(path + "/info.json"))
        self._components = self._setup(self.paths)

    @property
    def paths(self) -> Dict[str, str]:
        return self._props.paths

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._common_shape(self.shapes)

    @property
    def chunk(self) -> Tuple[int, ...]:
        return self._common_shape(self.chunks)

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self._get_property("shape")

    @property
    def chunks(self) -> Dict[str, Tuple[int, ...]]:
        return self._get_property("chunk")

    @property
    def dtype(self) -> Dict[str, str]:
        return self._get_property("dtype")

    @property
    def compress(self) -> Dict[str, str]:
        return self._get_property("compress")

    @property
    def compresslevel(self) -> Dict[str, float]:
        return self._get_property("compresslevel")

    def _get_property(self, name: str) -> Dict:
        return {k: getattr(comp, name) for k, comp in self._components.items()}

    def _common_shape(self, shapes: Tuple[int, ...]) -> Tuple[int, ...]:
        shapes = [shapes[k] for k in shapes]
        shapes = sorted(shapes, key=lambda x: len(x))
        min_shape = shapes[0]
        common_shape = []
        for dim in range(len(min_shape)):
            for shp in shapes:
                if min_shape[dim] != shp[dim]:
                    return common_shape
            common_shape.append(min_shape[dim])
        return common_shape

    def _setup(self, components: Dict[str, str]) -> Dict[str, Array]:
        datas = {}
        if components is None:
            return datas

        for key, path in components.items():
            datas[key] = Array(path, self._storage)

        return datas

    def to_pytorch(self, transform=None):
        try:
            return TorchIterableDataset(self, transform=transform)
        except Exception as ex:
            raise ex  # Exception('PyTorch is not installed')

    def to_tensorflow(self):
        try:
            return HubTensorflowDataset(self)
        except Exception as ex:
            raise ex  # Exception('TensorFlow is not intalled')

    def __getitem__(self, slices):
        if not isinstance(slices, list) and not isinstance(slices, tuple):
            slices = [slices]

        if isinstance(slices[0], str):
            if len(slices) == 1:
                return self._components[slices[0]]
            else:
                return self._components[slices[0]][slices[1:]]
        else:
            if len(slices) <= len(self.shape):
                datas = {key: value[slices] for key, value in self._components.items()}
                # return list(map(lambda x: x[slices], datas))
                return datas
            else:
                raise Exception(
                    "Slices ({}) could not much to multiple arrays".format(slices)
                )

    def __setitem__(self, slices, item):
        if isinstance(slices[0], str):
            if len(slices) == 1:
                return self._components[slices[0]]
            else:
                self._components[slices[0]][slices[1:]] = item
        else:
            if len(slices) < len(self.chunk_shape) and len(item) == len(
                self._components
            ):
                datas = [self._components[k] for k in self._components]

                def assign(xy):
                    xy[0][slices] = xy[1]

            return list(map(assign, zip(datas, item)))

    def __len__(self):
        return self.shape[0]

    def items(self):
        return list(self._components.items())

    def __iter__(self):
        for i in range(0, len(self)):
            yield self[i]
