from pathos.threading import ThreadPool
import numpy as np
from hub.log import logger
from .bbox import Bbox, chunknames, shade, Vec, generate_chunks
from .storage import Storage, S3
from hub.exceptions import IncompatibleBroadcasting, IncompatibleTypes, IncompatibleShapes, ArrayNotFound
import json

class HubArray(object):
    def __init__(self, shape=None, chunk_shape=None, dtype=None, key=None, protocol=None, parallel=True, order='F', public=False):
        self.shape = shape
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        self.key = key
        self.protocol = protocol
        self.storage = S3(self.key, public=public)
        self.order = order
        
        if parallel == False:
            parallel = 1
        if parallel == True:
            parallel = 25

        self.pool = ThreadPool(nodes=parallel)
        self.initialize(self.key)

        # Make sure the object was properly initialized
        assert self.shape
        assert self.chunk_shape
        assert self.dtype

    def initialize(self, path):
        cloudpath = "{}/info.txt".format(path)
        info = self.storage.get(cloudpath)
        if not info and not self.shape:
            name, dataset, version = self.key.split('/')[-3:]
            raise ArrayNotFound('Could not identify array with name {}/{}:{}. Please make sure the array name is correct.'.format(name, dataset, version))
        if info:
            info = json.loads(info)
            self.shape = info['shape']
            self.chunk_shape = info['chunk_shape']
            self.key = info['key']
            self.dtype = info['dtype']
            self.order = info['order']
            self.protocol = info['protocol']
        else:
            info = {
                'shape': self.shape, 
                'chunk_shape': self.chunk_shape,
                'key': self.key,
                'dtype': self.dtype,
                'order': self.order,
                'protocol': self.protocol,
            }
            info = json.dumps(info)
            self.storage.put(cloudpath, info, content_type=None)

    def generate_cloudpaths(self, slices):
        # Slices -> Bbox
        slices = Bbox(Vec.zeros(self.shape), self.shape).reify_slices(slices)
        requested_bbox = Bbox.from_slices(slices)

        # Make sure chunks fit
        full_bbox = requested_bbox.expand_to_chunk_size(
            self.chunk_shape, offset = Vec.zeros(self.shape)
        )

        # Clamb the border
        full_bbox = Bbox.clamp(full_bbox, Bbox(Vec.zeros(self.shape), self.shape))

        # Generate chunknames
        cloudpaths = list(chunknames(
            full_bbox, self.shape, 
            self.key, self.chunk_shape, 
            protocol=self.protocol
        ))
        return cloudpaths, requested_bbox 

    # read from raw file and transform to numpy array 
    def decode(self, chunk):
        return np.frombuffer(bytearray(chunk), dtype=self.dtype).reshape(self.chunk_shape, order='F')

    def download_chunk(self, cloudpath):
        chunk = self.storage.get(cloudpath)
        if chunk:
            chunk = self.decode(chunk)
        else: 
            chunk = np.zeros(shape=self.chunk_shape, dtype=self.dtype, order=self.order)
        bbox = Bbox.from_filename(cloudpath)
        return chunk, bbox

    def download(self, cloudpaths, requested_bbox):
        # Download chunks
        chunks_bboxs = self.pool.map(self.download_chunk, cloudpaths)

        # Combine Chunks
        renderbuffer = np.zeros(shape=requested_bbox.to_shape(), dtype=self.dtype, order=self.order)
        def process(chunk_bbox):
            chunk, bbox = chunk_bbox
            shade(renderbuffer, requested_bbox, chunk, bbox)
        self.pool.map(process, chunks_bboxs)

        return renderbuffer

    def squeeze(self, slices, tensor):
        squeeze_dims = [] 

        if isinstance(slices, list) and len(slices)==1:
            slices = slices[0]

        if not isinstance(slices, list) and not isinstance(slices, tuple):
            slices = [slices]
            
        for dim in range(len(slices)):
            if isinstance(slices[dim], int):
                squeeze_dims.append(dim)

        if len(squeeze_dims) >= 1:
            tensor = tensor.squeeze(axis=(*squeeze_dims, ) )

        if len(tensor.shape) == 0:
            tensor = tensor.item()

        return tensor

    def __getitem__(self, slices):
        cloudpaths, requested_bbox = self.generate_cloudpaths(slices)
        tensor = self.download(cloudpaths, requested_bbox)
        tensor = self.squeeze(slices, tensor)
        return tensor

    def encode(self, chunk):
        return chunk.tostring('F')

    def upload_chunk(self, cloudpath_chunk):
        cloudpath, chunk = cloudpath_chunk
        chunk = self.encode(chunk)
        chunk = self.storage.put(cloudpath, chunk, content_type=None)
        
    def chunkify(self, cloudpaths, requested_bbox, item):
        chunks = []
        for path in cloudpaths:
            cloudchunk = Bbox.from_filename(path)
            intersection = Bbox.intersection(cloudchunk, requested_bbox)
            chunk_slices = (intersection-cloudchunk.minpt).to_slices()
            item_slices = (intersection-requested_bbox.minpt).to_slices()

            chunk = np.zeros(shape=self.chunk_shape, dtype=self.dtype, order=self.order)
            if np.any(np.array(intersection.to_shape()) != np.array(self.chunk_shape)):
                logger.debug('Non aligned write')
                chunk, _ = self.download_chunk(path)
            else:
                chunk = np.zeros(shape=self.chunk_shape, dtype=self.dtype, order=self.order)

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
        self.pool.map(self.upload_chunk, list(cloudpaths_chunks))

    def __setitem__(self, slices, item):
        cloudpaths, requested_bbox = self.generate_cloudpaths(slices)
        self.upload(cloudpaths, requested_bbox, item)

