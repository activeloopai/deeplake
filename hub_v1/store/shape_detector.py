"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.utils import _tuple_product
from hub_v1.png_numcodec import PngCodec
import math

import numpy as np

from hub_v1.defaults import CHUNK_DEFAULT_SIZE, OBJECT_CHUNK, DEFAULT_COMPRESSOR
from hub_v1.exceptions import HubException


class ShapeDetector:
    # TODO replace asserts with appropriate errors, not clear what to do if it misses the alert
    # TODO move this to hub/store
    def __init__(
        self,
        shape,
        max_shape=None,
        chunks=None,
        dtype="float64",
        chunksize=CHUNK_DEFAULT_SIZE,
        object_chunking=OBJECT_CHUNK,
        compressor=DEFAULT_COMPRESSOR,
    ):
        self._int32max = np.iinfo(np.dtype("int32")).max

        self._dtype = dtype = np.dtype(dtype)
        self._object_chunking = object_chunking
        self._compressor = compressor

        self._chunksize = chunksize = self._get_chunksize(chunksize, compressor)
        self._shape = shape = self._get_shape(shape)
        self._max_shape = max_shape = self._get_max_shape(shape, max_shape)
        self._chunks = chunks = self._get_chunks(
            shape, max_shape, chunks, dtype, chunksize
        )

    def _get_chunksize(self, chunksize, compressor):
        if isinstance(compressor, PngCodec):
            return int(math.ceil(0.25 * chunksize))
        else:
            return chunksize

    def _get_shape(self, shape):
        assert shape is not None
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        for s in shape:
            assert s is None or isinstance(s, int)
        assert isinstance(shape[0], int)
        return shape

    def _get_max_shape(self, shape, max_shape):
        if max_shape is None:
            return tuple([s or self._int32max for s in shape])
        elif isinstance(max_shape, int):
            assert max_shape == shape[0]
            return self._get_max_shape(shape, None)
        else:
            max_shape = tuple(max_shape)
            assert len(shape) == len(max_shape)
            for (s, ms) in zip(shape, max_shape):
                if not isinstance(ms, int):
                    raise HubException("MaxShape Dimension should be int")
                if s is not None and s != ms:
                    raise HubException(
                        """Dimension in shape cannot be != max_shape dimension, 
                        if shape is not None """
                    )
                assert s == ms or s is None and isinstance(ms, int)
            return max_shape

    def _get_chunks(self, shape, max_shape, chunks, dtype, chunksize):
        if chunks is None:
            prod = _tuple_product(max_shape[1:])
            if dtype == "object":
                return (self._object_chunking,) + max_shape[1:]
            if prod <= 2 * chunksize:
                # FIXME not properly handled object type, U type, and so on.
                sz = dtype.itemsize
                chunks = int(math.ceil(chunksize / (prod * sz)))
                return (chunks,) + max_shape[1:]
            else:
                return (1,) + self._determine_chunksizes(
                    max_shape[1:], dtype, chunksize
                )
        elif isinstance(chunks, int):
            assert chunks > 0
            if chunks > 1:
                return (chunks,) + tuple(
                    [s or ms for s, ms in zip(shape[1:], max_shape[1:])]
                )
            else:
                return (1,) + self._determine_chunksizes(
                    max_shape[1:], dtype, chunksize
                )
        else:
            chunks = tuple(chunks)
            if len(chunks) == 1:
                return self._get_chunks(shape, max_shape, chunks[0], dtype, chunksize)
            else:
                if len(chunks) != len(shape):
                    raise Exception(
                        "If you want a multidimensional chunk, the number of dimensions should match with that of the shape parameter."
                        f" Number of chunk dimensions received={len(chunks)}, number of chunk dimensions required={len(shape)}"
                    )
                if chunks[0] != 1:
                    raise Exception(
                        "If you want multiple samples in the chunk, then specify only the number in the chunk, rest of the dimensions "
                        f"should be omitted. In this case use 'chunks={chunks[0]}'"
                    )
                return chunks

    def _determine_chunksizes(self, max_shape, dtype, chunksize):
        """
        Autochunking of tensors
        Chunk is determined by 16MB blocks keeping left dimensions inside a chunk
        Dimensions from left are kept until 16MB block is filled

        Parameters
        ----------
        shape: tuple
            the shape of the whole array
        dtype: type
            the type of the element (int, float)
        chunk_default_size: int (optional)
            how big the chunk size should be. Default to 16MB (CHUNK_DEFAULT_SIZE)
        """

        sz = np.dtype(dtype).itemsize
        elem_count_in_chunk = chunksize / sz

        # Get left most part which will be left static inside the chunk
        a = list(max_shape)
        a.reverse()
        left_part = max_shape
        prod = 1
        for i, dim in enumerate(a):
            prod *= dim
            if elem_count_in_chunk < prod:
                left_part = max_shape[-i:]
                break

        # If the tensor is smaller then the chunk size return
        if len(left_part) == len(max_shape):
            return tuple(left_part)

        # Get the middle chunk size of dimension
        els = int(math.ceil(elem_count_in_chunk / np.prod(np.array(left_part))))

        # Contruct the chunksize shape
        chunksize = [els] + list(left_part)
        if len(chunksize) < len(max_shape):
            chunksize = [1] * (len(max_shape) - len(chunksize)) + chunksize
        return tuple(chunksize)

    @property
    def shape(self):
        return self._shape

    @property
    def max_shape(self):
        return self._max_shape

    @property
    def chunks(self):
        return self._chunks

    @property
    def dtype(self):
        return self._dtype

    @property
    def chunksize(self):
        return self._chunksize
