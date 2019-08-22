# BSD 3-Clause License

# Copyright (c) 2017, Ignacio Tartavull, William Silversmith, and later authors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
from six.moves import range, reduce

import json
import os
import re
import sys
import math
import operator
import time
import random
import string
from itertools import product
import posixpath

import numpy as np
from hub.exceptions import OutOfBoundsError, AlignmentError


def map2(fn, a, b):
    assert len(a) == len(b), "Vector lengths do not match: {} (len {}), {} (len {})".format(
        a[:3], len(a), b[:3], len(b))

    result = np.empty(len(a))

    for i in range(len(result)):
        result[i] = fn(a[i], b[i])

    if isinstance(a, Vec) or isinstance(b, Vec):
        return Vec(*result)

    return result


def max2(a, b):
    return map2(max, a, b).astype(a.dtype)


def min2(a, b):
    return map2(min, a, b).astype(a.dtype)


def clamp(val, low, high):
    return min(max(val, low), high)


def generate_chunks(img, offset, chunk_size):
    shape = Vec(*img.shape)
    offset = Vec(*offset)

    bounds = Bbox(offset, shape + offset)

    alignment_check = bounds.round_to_chunk_size(
        chunk_size, Vec.zeros(*chunk_size))

    if not np.all(alignment_check.minpt == bounds.minpt):
        raise AlignmentError("""
        Only chunk aligned writes are supported by this function. 
  
        Got:             {}
        Volume Offset:   {} 
        Nearest Aligned: {}
      """.format(
            bounds, Vec.zeros(*chunk_size), alignment_check)
        )

    bounds = Bbox.clamp(bounds, bounds)

    img_offset = bounds.minpt - offset
    img_end = Vec.clamp(bounds.size3() + img_offset, Vec.zeros(shape), shape)

    for startpt in product(*xranges):
        startpt = startpt.clone()
        endpt = min2(startpt + chunk_size, shape)
        spt = (startpt + bounds.minpt).astype(int)
        ept = (endpt + bounds.minpt).astype(int)
        yield (startpt, endpt, spt, ept)


def shade(dest_img, dest_bbox, src_img, src_bbox):
    """
    Shade dest_img at coordinates dest_bbox using the
    image contained in src_img at coordinates src_bbox.

    The buffer will only be painted in the overlapping
    region of the content.

    Returns: void
    """
    if not Bbox.intersects(dest_bbox, src_bbox):
        return

    spt = max2(src_bbox.minpt, dest_bbox.minpt)
    ept = min2(src_bbox.maxpt, dest_bbox.maxpt)
    dbox = Bbox(spt, ept) - dest_bbox.minpt
    ZERO = Vec.zeros(dest_bbox.maxpt)
    istart = max2(spt - src_bbox.minpt, ZERO)

    # FIXME in case some here
    #iend = min2(ept - src_bbox.maxpt, ZERO) + src_img.shape
    iend = istart + (dest_bbox.maxpt - dest_bbox.minpt)
    sbox = Bbox(istart, iend)
    dest_img[dbox.to_slices()] = src_img[sbox.to_slices()]


def chunknames(bbox, volume_bbox, key, chunk_size, protocol=None):

    path = posixpath if protocol != 'file' else os.path

    xranges = []
    for minim, maxim, step_size in zip(bbox.minpt, bbox.maxpt, chunk_size):
        a = range(minim, maxim, step_size)
        xranges.append(a)

    for index in product(*xranges):
        highpt = min2(Vec(*index) + chunk_size,
                      Bbox([0 for i in volume_bbox], volume_bbox).maxpt)
        filename = ["{}-{}".format(x, max_x)
                    for x, max_x in zip(index, highpt)]
        filename = "_".join(filename)
        yield path.join(key, filename)


def check_bounds(val, low, high):
    if val > high or val < low:
        raise OutOfBoundsError(
            'Value {} cannot be outside of inclusive range {} to {}'.format(val, low, high))
    return val


if sys.version_info < (3,):
    integer_types = (int, long, np.integer)
else:
    integer_types = (int, np.integer)

floating_types = (float, np.floating)

# Formula produces machine epsilon regardless of platform architecture
MACHINE_EPSILON = (7. / 3) - (4. / 3) - 1


def floating(lst):
    return any((isinstance(x, float) for x in lst))


class Vec(np.ndarray):
    def __new__(cls, *args, **kwargs):
        dtype = kwargs['dtype'] if 'dtype' in kwargs else int
        return super(Vec, cls).__new__(cls, shape=(len(args),), buffer=np.array(args).astype(dtype), dtype=dtype)

    @classmethod
    def clamp(cls, val, minvec, maxvec):
        return Vec(*min2(max2(val, minvec), maxvec))

    @classmethod
    def zeros(cls, shape):
        if isinstance(shape, int):
            shape = range(shape)
        return Vec(*[0 for i in shape])

    def clone(self):
        return Vec(*self[:], dtype=self.dtype)

    def null(self):
        return self.length() <= 10 * np.finfo(np.float32).eps

    def dot(self, vec):
        return sum(self * vec)

    def length2(self):
        return self.dot(self)

    def length(self):
        return math.sqrt(self.dot(self))

    def rectVolume(self):
        return reduce(operator.mul, self)

    def __hash__(self):
        return int(''.join(map(str, self)))

    def __repr__(self):
        values = u",".join(list(self.astype(str)))
        return u"Vec({}, dtype={})".format(values, self.dtype)


class Bbox(object):
    __slots__ = ['minpt', 'maxpt', '_dtype']

    """Represents a three dimensional cuboid in space."""

    def __init__(self, a, b, dtype=None):
        if dtype is None:
            if floating(a) or floating(b):
                dtype = np.float32
            else:
                dtype = np.int32

        self.minpt = Vec(*[min(ai, bi) for ai, bi in zip(a, b)], dtype=dtype)
        self.maxpt = Vec(*[max(ai, bi) for ai, bi in zip(a, b)], dtype=dtype)

        self._dtype = np.dtype(dtype)

    @classmethod
    def deserialize(cls, bbx_data):
        bbx_data = json.loads(bbx_data)
        return Bbox.from_dict(bbx_data)

    def serialize(self):
        return json.dumps(self.to_dict())

    @property
    def ndim(self):
        return len(self.minpt)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def intersection(cls, bbx1, bbx2):
        result = Bbox([0] * bbx1.ndim, [0] * bbx2.ndim)

        if not Bbox.intersects(bbx1, bbx2):
            return result

        for i in range(result.ndim):
            result.minpt[i] = max(bbx1.minpt[i], bbx2.minpt[i])
            result.maxpt[i] = min(bbx1.maxpt[i], bbx2.maxpt[i])

        return result

    @classmethod
    def intersects(cls, bbx1, bbx2):
        return np.all(bbx1.minpt < bbx2.maxpt) and np.all(bbx1.maxpt > bbx2.minpt)

    @classmethod
    def near_edge(cls, bbx1, bbx2, distance=0):
        return (
            np.any(np.abs(bbx1.minpt - bbx2.minpt) <= distance)
            or np.any(np.abs(bbx1.maxpt - bbx2.maxpt) <= distance)
        )

    @classmethod
    def create(cls, obj, context=None, bounded=False):
        typ = type(obj)
        if typ is Bbox:
            obj = obj
        elif typ in (list, tuple):
            obj = Bbox.from_slices(obj, context, bounded)
        elif typ is Vec:
            obj = Bbox.from_vec(obj)
        elif typ is str:
            obj = Bbox.from_filename(obj)
        elif typ is dict:
            obj = Bbox.from_dict(obj)
        else:
            raise NotImplementedError(
                "{} is not a Bbox convertible type.".format(typ))

        if context and bounded:
            if not context.contains_bbox(obj):
                raise OutOfBoundsError(
                    "{} did not fully contain the specified bounding box {}.".format(
                        context, obj
                    ))

        return obj

    @classmethod
    def from_delta(cls, minpt, plus):
        return Bbox(minpt, Vec(*minpt) + plus)

    @classmethod
    def from_dict(cls, data):
        dtype = data['dtype'] if 'dtype' in data else np.float32
        return Bbox(data['minpt'], data['maxpt'], dtype=dtype)

    @classmethod
    def from_vec(cls, vec, dtype=int):
        return Bbox((0, 0, 0), vec, dtype=dtype)

    @classmethod
    def from_filename(cls, filename, dtype=int):
        match = os.path.basename(filename).replace(
            r'(?:\.gz)?$', '').split('_')
        match = [x.split('-') for x in match]
        mins = [int(x[0]) for x in match]
        maxs = [int(x[1]) for x in match]

        return Bbox(mins, maxs, dtype=dtype)

    @classmethod
    def from_slices(cls, slices, context=None, bounded=False):
        if context:
            slices = context.reify_slices(slices, bounded=bounded)

        return Bbox(
            [slc.start for slc in slices],
            [slc.stop for slc in slices]
        )

    @classmethod
    def from_list(cls, lst):
        """
        from_list(cls, lst)
        the lst length should be 6
        the first three values are the start, and the last 3 values are the stop 
        """
        assert len(lst) == 6
        return Bbox(lst[:3], lst[3:6])

    def to_filename(self):
        return '_'.join(
            (str(self.minpt[i]) + '-' + str(self.maxpt[i])
             for i in range(self.ndim))
        )

    def to_slices(self):
        return tuple([
            slice(int(self.minpt[i]), int(self.maxpt[i])) for i in range(self.ndim)
        ])

    def to_list(self):
        return list(self.minpt) + list(self.maxpt)

    def to_dict(self):
        return {
            'minpt': self.minpt.tolist(),
            'maxpt': self.maxpt.tolist(),
            'dtype': np.dtype(self.dtype).name,
        }

    def to_shape(self):
        return list(self.maxpt - self.minpt)

    def reify_slices(self, slices, bounded=True):
        """
        Convert free attributes of a slice object 
        (e.g. None (arr[:]) or Ellipsis (arr[..., 0]))
        into bound variables in the context of this
        bounding box.

        That is, for a ':' slice, slice.start will be set
        to the value of the respective minpt index of 
        this bounding box while slice.stop will be set 
        to the value of the respective maxpt index.

        Example:
          bbx = Bbox( (-1,-2,-3), (1,2,3) )
          bbx.reify_slices( (np._s[:],) )

          >>> [ slice(-1,1,1), slice(-2,2,1), slice(-3,3,1) ]

        Returns: [ slice, ... ]
        """
        if isinstance(slices, integer_types) or isinstance(slices, floating_types):
            slices = [slice(int(slices), int(slices)+1, 1)]
        elif type(slices) == slice:
            slices = [slices]
        elif type(slices) == Bbox:
            slices = slices.to_slices()
        elif slices == Ellipsis:
            slices = []

        slices = list(slices)

        for index, slc in enumerate(slices):
            if slc == Ellipsis:
                fill = self.ndim - len(slices) + 1
                slices = slices[:index] + \
                    (fill * [slice(None, None, None)]) + slices[index+1:]
                break

        while len(slices) < self.ndim:
            slices.append(slice(None, None, None))

        # First three slices are x,y,z, last is channel.
        # Handle only x,y,z here, channel seperately
        for index, slc in enumerate(slices):
            if isinstance(slc, integer_types) or isinstance(slc, floating_types):
                slices[index] = slice(int(slc), int(slc)+1, 1)
            elif slc == Ellipsis:
                raise ValueError(
                    "More than one Ellipsis operator used at once.")
            else:
                start = self.minpt[index] if slc.start is None else slc.start
                end = self.maxpt[index] if slc.stop is None else slc.stop
                step = 1 if slc.step is None else slc.step

                if step < 0:
                    raise ValueError(
                        'Negative step sizes are not supported. Got: {}'.format(step))

                # note: when unbounded, negative indicies do not refer to
                # the end of the volume as they can describe, e.g. a 1px
                # border on the edge of the beginning of the dataset as in
                # marching cubes.
                if bounded:
                    # if start < 0: # this is support for negative indicies
                    # start = self.maxpt[index] + start
                    check_bounds(start, self.minpt[index], self.maxpt[index])
                    # if end < 0: # this is support for negative indicies
                    #   end = self.maxpt[index] + end
                    check_bounds(end, self.minpt[index], self.maxpt[index])

                slices[index] = slice(start, end, step)

        return slices

    @classmethod
    def expand(cls, *args):
        result = args[0].clone()
        for bbx in args:
            result.minpt = min2(result.minpt, bbx.minpt)
            result.maxpt = max2(result.maxpt, bbx.maxpt)
        return result

    @classmethod
    def clamp(cls, bbx0, bbx1):
        result = bbx0.clone()
        result.minpt = Vec.clamp(bbx0.minpt, bbx1.minpt, bbx1.maxpt)
        result.maxpt = Vec.clamp(bbx0.maxpt, bbx1.minpt, bbx1.maxpt)
        return result

    def size(self):
        return Vec(*(self.maxpt - self.minpt), dtype=self.dtype)

    def size3(self):
        return Vec(*(self.maxpt[:3] - self.minpt[:3]), dtype=self.dtype)

    def subvoxel(self):
        """
        Previously, we used bbox.volume() < 1 for testing
        if a bounding box was larger than one voxel. However, 
        if two out of three size dimensions are negative, the 
        product will be positive. Therefore, we first test that 
        the maxpt is to the right of the minpt before computing 
        whether conjunctioned with volume() < 1.

        Returns: boolean
        """
        return (not self.valid()) or self.volume() < 1

    def empty(self):
        """
        Previously, we used bbox.volume() <= 0 for testing
        if a bounding box was empty. However, if two out of 
        three size dimensions are negative, the product will 
        be positive. Therefore, we first test that the maxpt 
        is to the right of the minpt before computing whether 
        the bbox is empty and account for 20x machine epsilon 
        of floating point error.

        Returns: boolean
        """
        return (not self.valid()) or (self.volume() < (20 * MACHINE_EPSILON))

    def valid(self):
        return np.all(self.minpt <= self.maxpt)

    def volume(self):
        if np.issubdtype(self.dtype, np.integer):
            return self.size3().astype(np.int64).rectVolume()
        else:
            return self.size3().astype(np.float64).rectVolume()

    def center(self):
        return (self.minpt + self.maxpt) / 2.0

    def grow(self, amt):
        assert amt >= 0
        self.minpt -= amt
        self.maxpt += amt
        return self

    def shrink(self, amt):
        assert amt >= 0
        self.minpt += amt
        self.maxpt -= amt

        if not self.valid():
            raise ValueError("Cannot shrink bbox below zero volume.")

        return self

    def expand_to_chunk_size(self, chunk_size, offset=Vec(0, 0, 0, dtype=int)):
        """
        Align a potentially non-axis aligned bbox to the grid by growing it
        to the nearest grid lines.

        Required:
          chunk_size: arraylike (x,y,z), the size of chunks in the 
                        dataset e.g. (64,64,64)
        Optional:
          offset: arraylike (x,y,z), the starting coordinate of the dataset
        """
        chunk_size = np.array(chunk_size, dtype=np.float32)
        result = self.clone()
        result = result - offset
        result.minpt = np.floor(result.minpt / chunk_size) * chunk_size
        result.maxpt = np.ceil(result.maxpt / chunk_size) * chunk_size
        return (result + offset).astype(self.dtype)

    def shrink_to_chunk_size(self, chunk_size, offset=Vec(0, 0, 0, dtype=int)):
        """
        Align a potentially non-axis aligned bbox to the grid by shrinking it
        to the nearest grid lines.

        Required:
          chunk_size: arraylike (x,y,z), the size of chunks in the 
                        dataset e.g. (64,64,64)
        Optional:
          offset: arraylike (x,y,z), the starting coordinate of the dataset
        """
        chunk_size = np.array(chunk_size, dtype=np.float32)
        result = self.clone()
        result = result - offset
        result.minpt = np.ceil(result.minpt / chunk_size) * chunk_size
        result.maxpt = np.floor(result.maxpt / chunk_size) * chunk_size

        # If we are inside a single chunk, the ends
        # can invert, which tells us we should collapse
        # to a single point.
        if np.any(result.minpt > result.maxpt):
            result.maxpt = result.minpt.clone()

        return (result + offset).astype(self.dtype)

    def round_to_chunk_size(self, chunk_size, offset=Vec(0, 0, 0, dtype=int)):
        """
        Align a potentially non-axis aligned bbox to the grid by rounding it
        to the nearest grid lines.

        Required:
          chunk_size: arraylike (x,y,z), the size of chunks in the 
                        dataset e.g. (64,64,64)
        Optional:
          offset: arraylike (x,y,z), the starting coordinate of the dataset
        """
        chunk_size = np.array(chunk_size, dtype=np.float32)
        result = self.clone()
        result = result - offset
        result.minpt = np.round(result.minpt / chunk_size) * chunk_size
        result.maxpt = np.round(result.maxpt / chunk_size) * chunk_size
        return (result + offset).astype(self.dtype)

    def contains(self, point):
        """
        Tests if a point on or within a bounding box.

        Returns: boolean
        """
        return np.all(point >= self.minpt) and np.all(point <= self.maxpt)

    def contains_bbox(self, bbox):
        return self.contains(bbox.minpt) and self.contains(bbox.maxpt)

    def clone(self):
        return Bbox(self.minpt, self.maxpt, dtype=self.dtype)

    def astype(self, typ):
        tmp = self.clone()
        tmp.minpt = tmp.minpt.astype(typ)
        tmp.maxpt = tmp.maxpt.astype(typ)
        tmp._dtype = tmp.minpt.dtype
        return tmp

    def transpose(self):
        return Bbox(self.minpt[::-1], self.maxpt[::-1])

    # note that operand can be a vector
    # or a scalar thanks to numpy
    def __sub__(self, operand):
        tmp = self.clone()

        if isinstance(operand, Bbox):
            tmp.minpt -= operand.minpt
            tmp.maxpt -= operand.maxpt
        else:
            tmp.minpt -= operand
            tmp.maxpt -= operand

        return tmp

    def __iadd__(self, operand):
        if isinstance(operand, Bbox):
            self.minpt += operand.minpt
            self.maxpt += operand.maxpt
        else:
            self.minpt += operand
            self.maxpt += operand

        return self

    def __add__(self, operand):
        tmp = self.clone()
        return tmp.__iadd__(operand)

    def __imul__(self, operand):
        self.minpt *= operand
        self.maxpt *= operand
        return self

    def __mul__(self, operand):
        tmp = self.clone()
        tmp.minpt *= operand
        tmp.maxpt *= operand
        return tmp.astype(tmp.minpt.dtype)

    def __idiv__(self, operand):
        if (
            isinstance(operand, float)
            or self.dtype in (float, np.float32, np.float64)
            or (hasattr(operand, 'dtype') and operand.dtype in (float, np.float32, np.float64))
        ):
            return self.__itruediv__(operand)
        else:
            return self.__ifloordiv__(operand)

    def __div__(self, operand):
        if (
            isinstance(operand, float)
            or self.dtype in (float, np.float32, np.float64)
            or (hasattr(operand, 'dtype') and operand.dtype in (float, np.float32, np.float64))
        ):

            return self.__truediv__(operand)
        else:
            return self.__floordiv__(operand)

    def __ifloordiv__(self, operand):
        self.minpt //= operand
        self.maxpt //= operand
        return self

    def __floordiv__(self, operand):
        tmp = self.astype(float)
        tmp.minpt //= operand
        tmp.maxpt //= operand
        return tmp.astype(int)

    def __itruediv__(self, operand):
        self.minpt /= operand
        self.maxpt /= operand
        return self

    def __truediv__(self, operand):
        tmp = self.clone()

        if isinstance(operand, int):
            operand = float(operand)

        tmp.minpt = Vec(*(tmp.minpt.astype(float) / operand), dtype=float)
        tmp.maxpt = Vec(*(tmp.maxpt.astype(float) / operand), dtype=float)
        return tmp.astype(tmp.minpt.dtype)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        return np.array_equal(self.minpt, other.minpt) and np.array_equal(self.maxpt, other.maxpt)

    def __hash__(self):
        return int(''.join(map(str, map(int, self.to_list()))))

    def __repr__(self):
        return "Bbox({},{}, dtype={})".format(list(self.minpt), list(self.maxpt), self.dtype)
