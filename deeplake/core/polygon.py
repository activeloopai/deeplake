from deeplake.util.exceptions import EmptyPolygonError
from typing import Union, List

import numpy as np
import deeplake


class Polygon:
    """Represents a polygon."""

    def __init__(self, coords: Union[np.ndarray, List[float]], dtype="float32"):
        if coords is None or len(coords) == 0:
            raise EmptyPolygonError(
                "A polygons sample can be empty or None but a polygon within a sample cannot be empty or None."
            )
        self.coords = coords
        self.dtype = dtype

    @property
    def ndim(self):
        """Dimension of the polygon."""
        return len(self.coords[0])

    def __array__(self, dtype=None) -> np.ndarray:
        """Returns a numpy array with the co-ordinates of the points in the polygon."""
        if not dtype:
            dtype = self.dtype
        if isinstance(self.coords, np.ndarray) and self.coords.dtype == dtype:
            return self.coords
        return np.array(self.coords, dtype=dtype)

    def __len__(self):
        """Returns the number of points in the polygon."""
        return len(self.coords)

    def __getitem__(self, i):
        """Returns the ``i`` th co-ordinate of the polygon."""
        return self.coords[i]

    def tobytes(self) -> bytes:
        return self.__array__().tobytes()

    @property
    def shape(self):
        """Returns the shape of the Polygon - (#co-ordinates, #dimensions)"""
        return (len(self.coords), len(self.coords[0]))


class Polygons:
    """Represents a list of polygons"""

    def __init__(self, data: Union[np.ndarray, List], dtype="float32"):
        if data is None:
            data = []
        if isinstance(data, deeplake.core.tensor.Tensor):
            data = data.numpy()
        self.data = data
        self.dtype = dtype
        self._validate()
        # Note: here ndim is the number of dimensions of the polygon's space, not the ndim of data.
        self.ndim = len(self.data[0][0]) if len(self.data) else 0
        self.shape = (len(self.data), max(map(len, self.data), default=0), self.ndim)

    def _validate(self):
        if len(self.data):
            ndim = self[0].ndim
            for p in self:
                assert p.ndim == ndim

    def __getitem__(self, i):
        """Returns a :class:`~deeplake.core.polygon.Polygon` if ``i`` is an ``int``, otherwise another :class:`~deeplake.core.polygon.Polygons` object."""
        if isinstance(i, (int, slice, list)):
            return Polygon(self.data[i], self.dtype)
        elif isinstance(i, tuple):
            if len(i) != 1:
                raise IndexError(f"Unsupported index: {i}")
            return Polygon(self.data[i[0]], self.dtype)
        else:
            raise IndexError(f"Unsupported index: {i}")

    def __len__(self):
        """Returns the number of polygons in this group."""
        return len(self.data)

    def __iter__(self):
        for c in self.data:
            yield Polygon(c, self.dtype)

    def tobytes(self) -> memoryview:
        if not len(self.data):
            return memoryview(b"")
        ndim = self.ndim
        assert ndim < 256, "Maximum number of dimensions supported is 255."  # uint8
        lengths = list(map(len, self))
        assert (
            max(lengths) < 65536
        ), "Maximum number of points per polygon is 65535."  # uint16
        num_polygons = len(self.data)
        assert (
            num_polygons < 65536
        ), "Maximum number of polygons per sample is 65535."  # uint16
        dtype = np.dtype(self.dtype)
        data_size = dtype.itemsize * self.ndim * sum(lengths)
        header_size = 2 + num_polygons * 2  # [num_polygons ui16][lengths ui16]
        buff = bytearray(header_size + data_size)
        buff[:2] = num_polygons.to_bytes(2, "little")
        offset = num_polygons * 2 + 2
        buff[2:offset] = np.array(lengths, dtype=np.uint16).tobytes()
        for polygon in self:
            bts = polygon.tobytes()
            nbts = len(bts)
            buff[offset : offset + nbts] = bts
            offset += nbts
        assert offset == len(buff)
        return memoryview(buff)

    @classmethod
    def frombuffer(cls, buff, dtype, ndim):
        if not buff:
            return cls([], dtype)
        num_polygons = int.from_bytes(buff[:2], "little")
        offset = num_polygons * 2 + 2
        lengths = np.frombuffer(buff[2:offset], dtype=np.uint16)
        points = np.frombuffer(buff[offset:], dtype=dtype).reshape(-1, ndim)
        data = []
        for l in lengths:
            data.append(points[:l])
            points = points[l:]
        return cls(data, dtype)

    def astype(self, dtype):
        """Returns the polygons in the specified dtype."""
        return Polygons(self.data, np.dtype(dtype).name)

    def copy(self):
        # No-op, called by pytorch integration
        if isinstance(self.data, np.ndarray):
            return Polygons(self.data.copy(), self.dtype)
        return Polygons(
            [p.copy() if isinstance(p, np.ndarray) else p for p in self.data],
            self.dtype,
        )

    def numpy(self) -> List[np.ndarray]:
        """Returns a list of numpy arrays corresponding to each polygon in this group."""
        return [Polygon(p, self.dtype).__array__() for p in self.data]
