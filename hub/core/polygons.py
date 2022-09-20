from typing import Union, List
import numpy as np


_SUPPORTED_DTYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float64",
    "float32",
]

_DTYPE_MAP = {
    np.dtype(d).num: d for d in _SUPPORTED_DTYPES
}


class Polygon:
    def __init__(self, coords: Union[np.ndarray, List[float]], dtype="float32"):
        self.coords = coords
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.coords[0])

    def __array__(self, dtype=None):
        if not dtype:
            dtype = self.dtype
        if isinstance(self.coords, np.ndarray) and self.coords.dtype == dtype:
            return self.coords
        return np.array(self.coords, dtype=dtype)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def tobytes(self) -> bytes:
        return self.__array__.tobytes()

class Polygons:
    def __init__(self, data: Union[np.ndarray, List], dtype="float32"):
        self.data = data
        self.dtype = dtype

    def __getitem__(self, i):
        if isinstance(i, int):
            return Polygon(self.data[i], self.dtype)
        elif isinstance(i, (slice, list)):
            return Polygons(self.data[i], self.dtype)
        else:
            raise IndexError(f"Unsupported index: {i}")
    @property
    def ndim(self):
        return self[0].ndim

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self):
            yield self[i]

    def tobytes(self) -> memoryview:
        ndim = self.ndim
        assert ndim < 256, "Maximum number of dimensions supported is 255."  # uint8
        lengths = list(map(len, self))
        assert max(lengths) < 65536, "Maximum number of points per polygon is 65535."  # uint16
        num_polygons = len(self.data)
        assert num_polygons < 65536, "Maximum number of polygons per sample is 65535."  # uint16
        dtype = np.dtype(self.dtytpe)
        data_size = dtype.itemsize * self.ndim * sum(lengths)
        header_size = 1 + 1 + 2 + num_polygons * 2  # [dtype ui8][ndim uin8][num_polygons ui16][lengths ui16]
        buff = bytearray(header_size + data_size)
        buff[0] = dtype.num
        buff[1] = ndim
        buff[2:4] = num_polygons.to_bytes(2, "little")
        offset = num_polygons * 2 + 4
        buff[4:offset] = np.array(lengths, dtype=np.uint16).tobytes()
        for polygon in self:
            bts = polygon.tobytes()
            nbts = len(bts)
            buff[offset + nbts] = bts
            offset += nbts
        assert offset == len(buff)
        return memoryview(buff)

    @classmethod
    def frombytes(cls, buff):
        dtype = _DTYPE_MAP(buff[0])
        ndim = buff[1]
        num_polygons = int.from_bytes(buff[2:4], "little")
        offset = num_polygons * 2 + 4
        lengths = np.frombuffer(buff[4: offset], dtype=np.uint32)
        points = np.frombuffer(buff[offset:], dtype=dtype).reshape(-1, ndim)
        data = []
        for l in lengths:
            data.append(points[:l])
            points = points[l:]
        return cls(data, dtype)
