from pathos.threading import ThreadPool
import numpy as np
from hub.log import logger
from .bbox import Bbox, chunknames, shade, Vec, generate_chunks
from hub.exceptions import IncompatibleBroadcasting, IncompatibleTypes, IncompatibleShapes, NotFound
from .meta import MetaObject
import os
import io
import zlib
import gzip
import pickle
import hub.backend.storage
from PIL import Image
from typing import Tuple, Optional
from hub import codec

class HubArray(MetaObject):
    def __init__(self, 
        shape: Optional[Tuple[int, ...]] = None, 
        chunk_shape: Optional[Tuple[int, ...]] = None, 
        dtype: Optional[str] = None, 
        key=None, 
        protocol='s3', 
        parallel=True,  
        public=False, 
        storage: hub.backend.storage = None, 
        compression: Optional[str]='zlib', 
        compression_level: float = 0.5):

        super().__init__(key=key, storage=storage, create=shape)
        
        if compression is str:
            compression = compression.lower()

        self.info['shape'] = shape
        self.info['chunk_shape'] = chunk_shape
        self.info['dtype'] = dtype
        self.info['key'] = key
        self.info['protocol'] = protocol
        self.info['dclass'] = self.dclass = 'array'
        self.info['compress'] = compression
        self.info['compresslevel'] = compression_level

        parallel = 25 if parallel else 1

        self.pool = ThreadPool(nodes=1)
        self.initialize(self.key)
        self._codec = codec.create(self.compression, self.compression_level)

        # Make sure the objectmultidimentional was properly initialized
        assert self.shape
        assert self.chunk_shape
        assert self.dtype
        assert self.compression

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.info['shape']
    
    @property
    def chunk(self) -> Tuple[int, ...]:
        return self.info['chunk_shape']

    @property
    def chunk_shape(self) -> Tuple[int, ...]:
        return self.info['chunk_shape']

    @property
    def dtype(self) -> str:
        return self.info['dtype']
    
    @property
    def protocol(self) -> str:
        return self.info['protocol']

    @property
    def compress(self) -> Optional[str]:
        return self.info['compress']

    @property
    def compression(self) -> Optional[str]:
        return self.info['compress']

    @property
    def compresslevel(self) -> float:
        return self.info['compresslevel']

    @property
    def compression_level(self) -> float:
        return self.info['compresslevel']

    def generate_cloudpaths(self, slices):
        # Slices -> Bbox
        slices = Bbox(Vec.zeros(self.shape), self.shape).reify_slices(slices)
        requested_bbox = Bbox.from_slices(slices)

        # Make sure chunks fit
        full_bbox = requested_bbox.expand_to_chunk_size(
            self.chunk_shape, offset=Vec.zeros(self.shape)
        )

        # Clamb the border
        full_bbox = Bbox.clamp(full_bbox, Bbox(
            Vec.zeros(self.shape), self.shape))

        # Generate chunknames
        cloudpaths = list(chunknames(
            full_bbox, self.shape,
            self.key, self.chunk_shape,
            protocol=self.protocol
        ))

        return cloudpaths, requested_bbox

    # read from raw file and transform to numpy array
    def decode(self, chunk: bytes) -> np.ndarray:
        return self._codec.decode(chunk)

    def download_chunk(self, cloudpath):
        chunk = self.storage.get(cloudpath)
        if chunk:
            chunk = self.decode(chunk)
        else:
            chunk = np.zeros(shape=self.chunk_shape,
                             dtype=self.dtype)
                             
        bbox = Bbox.from_filename(cloudpath)
        return chunk, bbox

    def download(self, cloudpaths, requested_bbox):
        # Download chunks
        chunks_bboxs = list(map(self.download_chunk, cloudpaths))

        # Combine Chunks
        renderbuffer = np.zeros(
            shape=requested_bbox.to_shape(), dtype=self.dtype)

        def process(chunk_bbox):
            chunk, bbox = chunk_bbox
            shade(renderbuffer, requested_bbox, chunk, bbox)
        list(map(process, chunks_bboxs))

        return renderbuffer

    def squeeze(self, slices, tensor):
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

    def __getitem__(self, slices):
        cloudpaths, requested_bbox = self.generate_cloudpaths(slices)
        tensor = self.download(cloudpaths, requested_bbox)
        tensor = self.squeeze(slices, tensor)
        return tensor

    def encode(self, chunk: np.ndarray) -> bytes:
        return self._codec.encode(chunk)

    def upload_chunk(self, cloudpath_chunk):
        cloudpath, chunk = cloudpath_chunk
        chunk = self.encode(chunk)
        chunk = self.storage.put(cloudpath, chunk)

    def chunkify(self, cloudpaths, requested_bbox, item):
        chunks = []
        for path in cloudpaths:
            cloudchunk = Bbox.from_filename(path)
            intersection = Bbox.intersection(cloudchunk, requested_bbox)
            chunk_slices = (intersection-cloudchunk.minpt).to_slices()
            item_slices = (intersection-requested_bbox.minpt).to_slices()

            chunk = None
            if np.any(np.array(intersection.to_shape()) != np.array(self.chunk_shape)):
                logger.debug('Non aligned write')
                chunk, _ = self.download_chunk(path)
            else:
                chunk = np.zeros(shape=self.chunk_shape,
                                 dtype=self.dtype)

            chunk.setflags(write=1)
            chunk[chunk_slices] = item[item_slices]
            chunks.append(chunk)

        return zip(cloudpaths, chunks)

    def upload(self, cloudpaths, requested_bbox, item):
        try:
            item = np.broadcast_to(item, requested_bbox.to_shape())
        except ValueError as err:
            raise IncompatibleBroadcasting(err)

        try:
            item = item.astype(self.dtype)
        except Exception as err:
            raise IncompatibleTypes(err)

        cloudpaths_chunks = self.chunkify(cloudpaths, requested_bbox, item)
        list(map(self.upload_chunk, list(cloudpaths_chunks)))

    def __setitem__(self, slices, item):
        cloudpaths, requested_bbox = self.generate_cloudpaths(slices)
        self.upload(cloudpaths, requested_bbox, item)
