import hub
from pathos.threading import ThreadPool
import numpy as np
from hub.log import logger
from .bbox import Bbox, chunknames, shade, Vec, generate_chunks
from hub.backend.storage import Storage, S3, FS
from hub.exceptions import IncompatibleBroadcasting, IncompatibleTypes, IncompatibleShapes, NotFound
from hub.utils import StoreControlClient
import json
from hub.marray.array import HubArray
from .array import MetaObject

try:
    from hub.integrations import TorchDataset
except:
    pass


class Dataset(MetaObject):
    def __init__(self, objdict=None, key='', storage=None):
        super().__init__(key=key, storage=storage, create=objdict)
        if objdict is None:
            self.initialize(self.key)
            self.datas = self.setup(self.info['keys'])
        else:
            self.datas = self.setup(objdict)
            self.initialize(self.key)

        parallel = max(len(self.datas), 1)
        self.pool = ThreadPool(nodes=parallel)
        self.info['dclass'] = self.dclass = 'dataset'

    @property
    def shapes(self):
        return self.get_property('shape')

    @property
    def dtype(self):
        return self.get_property('dtype')

    @property
    def chunk_shapes(self):
        return self.get_property('chunk_shape')

    @property
    def protocol(self):
        return self.get_property('protocol')

    @property
    def storages(self):
        return self.get_property('storage')

    @property
    def protocol(self):
        return self.get_property('protocol')

    @property
    def order(self):
        return self.get_property('order')

    @property
    def keys(self):
        return self.get_property('key')

    def get_property(self, name):
        return {k: getattr(self.datas[k], name)
                for k in self.datas}

    @property
    def shape(self):
        return self.common_shape(self.shapes)

    @property
    def chunk_shape(self):
        return self.common_shape(self.chunk_shapes)

    def common_shape(self, shapes):
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

    def setup(self, objdict):
        self.datas = {}
        if objdict is None:
            return self.datas

        for k in objdict:
            if isinstance(objdict[k], str):
                self.datas[k] = HubArray(key=objdict[k])
            elif isinstance(objdict[k], HubArray):
                self.datas[k] = objdict[k]
            else:
                raise Exception(
                    'Input to the dataset is unknown: {}:{}'.format(k, objdict[k]))

        self.info['keys'] = self.keys
        self.info['chunk_shapes'] = self.chunk_shapes
        self.info['order'] = self.order
        self.info['shapes'] = self.shapes
        self.info['shape'] = self.shape
        self.info['chunk_shape'] = self.chunk_shape

        return self.datas

    def to_pytorch(self, transforms=None):
        try:
            return TorchDataset(
                self,
                transforms=transforms
            )
        except:
            Exception('PyTorch is not installed')

    def to_tensorflow(self):
        raise NotImplemented


    def __getitem__(self, slices):
        if not isinstance(slices, list) and not isinstance(slices, tuple):
            slices = [slices]

        if isinstance(slices[0], str):
            if len(slices) == 1:
                return self.datas[slices[0]]
            else:
                return self.datas[slices[0]][slices[1:]]
        else:
            if len(slices) <= len(self.shape):
                datas = [self.datas[k] for k in self.datas]
                return self.pool.map(lambda x: x[slices], datas)
            else:
                raise Exception(
                    'Slices ({}) could not much to multiple arrays'.format(slices))

    def __setitem__(self, slices, item):
        if isinstance(slices[0], str):
            if len(slices) == 1:
                return self.datas[slices[0]]
            else:
                self.datas[slices[0]][slices[1:]] = item
        else:
            if len(slices) < len(self.chunk_shape) and len(item) == len(self.datas):
                datas = [self.datas[k] for k in self.datas]

                def assign(xy):
                    xy[0][slices] = xy[1]
            return self.pool.map(assign, zip(datas, item))
