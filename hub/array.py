from typing import *
import sys, os, json, uuid, traceback, random, time
import numpy as np

# from pathos.threading import ThreadPool

from multiprocessing.pool import ThreadPool

from .storage import Base as Storage
from . import codec
from .bbox import Bbox, chunknames, shade, Vec, generate_chunks
from .exceptions import (
    IncompatibleBroadcasting,
    IncompatibleTypes,
    IncompatibleShapes,
    NotFound,
)

_hub_thread_pool = None


class Props:
    shape: Tuple[int, ...] = None
    chunk_shape: Tuple[int, ...] = None
    dtype: str = None
    compress: str = None
    compresslevel: float = 0.5
    # dsplit: Optional[Union[int, List[int]]] = None
    darray: str = None

    @property
    def chunk(self) -> Tuple[int, ...]:
        return self.chunk_shape

    @chunk.setter
    def chunk(self, value: Tuple[int, ...]):
        self.chunk_shape = value

    def __init__(self, dict: dict = None):
        if dict is not None:
            self.__dict__ = dict


class Array:
    def __init__(self, path: str, storage: Storage, threaded=False):
        self._path = path
        self._storage = storage
        self._props = Props(json.loads(storage.get(os.path.join(path, "info.json"))))
        self._codec = codec.from_name(self.compress, self.compresslevel)
        self._dcodec = codec.Default()
        global _hub_thread_pool
        if _hub_thread_pool is None and threaded:
            print("Thread Pool Created")
            _hub_thread_pool = ThreadPool(32)
        self._map = _hub_thread_pool.map if threaded else map

        self._darray = None
        if self._props.darray:
            self._darray = Array(os.path.join(path, self._props.darray), storage)
        # assert isinstance(self._props.dsplit, int)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._props.shape

    @property
    def darray(self) -> "Array":
        return self._darray

    # @property
    # def dynamic_shape(self, slices: Union[Slice[int, ...], Tuple[int, ...], List[int, ...]]):
    #     pass

    @property
    def chunk(self) -> Tuple[int, ...]:
        return self._props.chunk

    @property
    def dtype(self) -> str:
        return self._props.dtype

    @property
    def compress(self) -> str:
        return self._props.compress

    @property
    def compresslevel(self) -> float:
        return self._props.compresslevel

    # @property
    # def _dshape_path(self):
    #     return os.path.join(self._path, 'dshape.json')

    # def _get_dshape(self) -> np.ndarray:
    #     data = self._storage.get(self._dshape_path)
    #     return self._dcodec.decode(data)

    # def _set_dshape(self, arr: np.ndarray):
    #     data = self._dcodec.encode(arr)
    #     self._storage.put(self._dshape_path, data)

    # def get_shape(self, slices: Iterable[slice]):
    #     slices = tuple(slices)

    # # Iterable slices
    # def set_shape(self, slices: Iterable[slice], shape: Iterable[int]):
    #     slices = tuple(slices)
    #     shape = tuple(shape)
    #     assert len(slices) == self._props.dsplit
    #     assert len(shape) == len(self._props.shape) - self._props.dsplit

    #     arr = self._get_dshape()
    #     arr[slices] = shape
    #     self._set_dshape(arr)

    def __getitem__(self, slices: Tuple[slice]):
        cloudpaths, requested_bbox = self._generate_cloudpaths(slices)
        tensor = self._download(cloudpaths, requested_bbox)
        tensor = self._squeeze(slices, tensor)
        return tensor

    def __setitem__(self, slices: Tuple[slice], content: np.ndarray):
        cloudpaths, requested_bbox = self._generate_cloudpaths(slices)
        self._upload(cloudpaths, requested_bbox, content)

    def _generate_cloudpaths(self, slices):
        # Slices -> Bbox
        if isinstance(slices, int):
            slices = (slices,)
        elif isinstance(slices, slice):
            slices = (slices,)
        slices = tuple(slices)

        _shape = list(self.shape)
        if self._darray is not None:
            s = len(self._darray.shape) - 1
            arr = self._darray[slices[:s]]
            res = np.amax(arr, axis=tuple(range(0, len(arr.shape) - 1)))
            assert len(res.shape) == 1
            assert len(_shape[s:]) == res.shape[0]
            _shape[s:] = res
            _shape = tuple(_shape)

        slices = Bbox(Vec.zeros(_shape), _shape).reify_slices(slices, bounded=True)
        requested_bbox = Bbox.from_slices(slices)

        # Make sure chunks fit
        full_bbox = requested_bbox.expand_to_chunk_size(
            self.chunk, offset=Vec.zeros(self.shape)
        )

        # Clamb the border
        full_bbox = Bbox.clamp(full_bbox, Bbox(Vec.zeros(self.shape), self.shape))

        # Generate chunknames
        cloudpaths = list(
            chunknames(
                full_bbox,
                self.shape,
                self._path,
                self.chunk,
                protocol="none",  # self.protocol
            )
        )

        return cloudpaths, requested_bbox

    def _download_chunk(self, cloudpath):

        chunk = self._storage.get_or_none(cloudpath)
        if chunk:
            chunk = self._codec.decode(chunk)
        else:
            chunk = np.zeros(shape=self.chunk, dtype=self.dtype)

        bbox = Bbox.from_filename(cloudpath)
        return chunk, bbox

    def _download(self, cloudpaths, requested_bbox):
        # Download chunks
        chunks_bboxs = list(self._map(self._download_chunk, cloudpaths))

        # Combine Chunks
        renderbuffer = np.zeros(shape=requested_bbox.to_shape(), dtype=self.dtype)

        def process(chunk_bbox):
            chunk, bbox = chunk_bbox
            shade(renderbuffer, requested_bbox, chunk, bbox)

        list(self._map(process, chunks_bboxs))

        return renderbuffer

    def _squeeze(self, slices, tensor):
        squeeze_dims = []

        if isinstance(slices, list) and len(slices) == 1:
            slices = slices[0]

        if not isinstance(slices, list) and not isinstance(slices, tuple):
            slices = [slices]

        for dim in range(len(slices)):
            if isinstance(slices[dim], int):
                squeeze_dims.append(dim)

        if len(squeeze_dims) >= 1:
            tensor = tensor.squeeze(axis=(*squeeze_dims,))

        if len(tensor.shape) == 0:
            tensor = tensor.item()

        return tensor

    def _upload_chunk(self, cloudpath_chunk):
        cloudpath, chunk = cloudpath_chunk
        chunk = self._codec.encode(chunk)
        chunk = self._storage.put(cloudpath, chunk)

    def _chunkify(self, cloudpaths, requested_bbox, item):
        chunks = []
        for path in cloudpaths:
            cloudchunk = Bbox.from_filename(path)
            intersection = Bbox.intersection(cloudchunk, requested_bbox)
            chunk_slices = (intersection - cloudchunk.minpt).to_slices()
            item_slices = (intersection - requested_bbox.minpt).to_slices()

            chunk = None
            if np.any(np.array(intersection.to_shape()) != np.array(self.chunk)):
                chunk, _ = self._download_chunk(path)
            else:
                chunk = np.zeros(shape=self.chunk, dtype=self.dtype)

            chunk.setflags(write=1)
            chunk[chunk_slices] = item[item_slices]
            chunks.append(chunk)

        return zip(cloudpaths, chunks)

    def _upload(self, cloudpaths, requested_bbox, item):
        try:
            item = np.broadcast_to(item, requested_bbox.to_shape())
        except ValueError as err:
            raise IncompatibleBroadcasting(err)

        try:
            item = item.astype(self.dtype)
        except Exception as err:
            raise IncompatibleTypes(err)

        cloudpaths_chunks = self._chunkify(cloudpaths, requested_bbox, item)
        list(self._map(self._upload_chunk, list(cloudpaths_chunks)))
