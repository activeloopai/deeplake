import numpy
import json
from typing import Tuple
# from pathos.threading import ThreadPool

from multiprocessing.pool import ThreadPool

from .storage import Base as Storage
from . import codec
from .marray.bbox import Bbox, chunknames, shade, Vec, generate_chunks
from .exceptions import IncompatibleBroadcasting, IncompatibleTypes, IncompatibleShapes, NotFound

_hub_thread_pool = None

class Props():
    shape: Tuple[int, ...] = None
    chunk_shape: Tuple[int, ...] = None
    dtype: str = None
    compress: str = None
    compresslevel: float = 0.5

    @property
    def chunk(self) -> Tuple[int, ...]:
        return self.chunk_shape

    @chunk.setter
    def chunk(self, value: Tuple[int, ...]):
        self.chunk_shape = value

class Array():
    def __init__(self, path: str, storage: Storage, threaded=True):
        self._path = path
        self._storage = storage
        self._props = Props()
        self._props.__dict__ = json.loads(storage.get(path + "/info.json"))
        self._codec = codec.from_name(self.compress, self.compresslevel)
        global _hub_thread_pool
        if _hub_thread_pool is None:
            print('Thread Pool Created')
            _hub_thread_pool = ThreadPool(32)
        self._map = _hub_thread_pool.map if threaded else map
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._props.shape

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

    def __getitem__(self, slices: Tuple[slice]):
        cloudpaths, requested_bbox = self._generate_cloudpaths(slices)
        tensor = self._download(cloudpaths, requested_bbox)
        tensor = self._squeeze(slices, tensor)
        return tensor

    def __setitem__(self, slices: Tuple[slice], content: numpy.ndarray):
        cloudpaths, requested_bbox = self._generate_cloudpaths(slices)
        self._upload(cloudpaths, requested_bbox, content)

    def _generate_cloudpaths(self, slices):
        # Slices -> Bbox
        slices = Bbox(Vec.zeros(self.shape), self.shape).reify_slices(slices, bounded=True)
        requested_bbox = Bbox.from_slices(slices)

        # Make sure chunks fit
        full_bbox = requested_bbox.expand_to_chunk_size(
            self.chunk, offset=Vec.zeros(self.shape)
        )

        # Clamb the border
        full_bbox = Bbox.clamp(full_bbox, Bbox(
            Vec.zeros(self.shape), self.shape))

        # Generate chunknames
        cloudpaths = list(chunknames(
            full_bbox, self.shape,
            self._path, self.chunk,
            protocol = 'none' # self.protocol
        ))

        return cloudpaths, requested_bbox

    def _download_chunk(self, cloudpath):
        
        chunk = self._storage.get_or_none(cloudpath)
        if chunk:
            chunk = self._codec.decode(chunk)
        else:
            chunk = numpy.zeros(shape=self.chunk, dtype=self.dtype)
                  
        bbox = Bbox.from_filename(cloudpath)
        return chunk, bbox

    def _download(self, cloudpaths, requested_bbox):
        # Download chunks
        chunks_bboxs = list(self._map(self._download_chunk, cloudpaths))

        # Combine Chunks
        renderbuffer = numpy.zeros(shape=requested_bbox.to_shape(), dtype=self.dtype)

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
            tensor = tensor.squeeze(axis=(*squeeze_dims, ))

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
            chunk_slices = (intersection-cloudchunk.minpt).to_slices()
            item_slices = (intersection-requested_bbox.minpt).to_slices()

            chunk = None
            if numpy.any(numpy.array(intersection.to_shape()) != numpy.array(self.chunk)):
                chunk, _ = self._download_chunk(path)
            else:
                chunk = numpy.zeros(shape=self.chunk, dtype=self.dtype)

            chunk.setflags(write=1)
            chunk[chunk_slices] = item[item_slices]
            chunks.append(chunk)

        return zip(cloudpaths, chunks)

    def _upload(self, cloudpaths, requested_bbox, item):
        try:
            item = numpy.broadcast_to(item, requested_bbox.to_shape())
        except ValueError as err:
            raise IncompatibleBroadcasting(err)

        try:
            item = item.astype(self.dtype)
        except Exception as err:
            raise IncompatibleTypes(err)

        cloudpaths_chunks = self._chunkify(cloudpaths, requested_bbox, item)
        list(self._map(self._upload_chunk, list(cloudpaths_chunks)))
