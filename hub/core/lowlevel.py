import numpy as np
import ctypes
from collections import namedtuple
from typing import Tuple, Sequence, Union, Optional, List
import hub


class Pointer(object):
    __slots__ = ("address", "size", "_c_array", "_refs")

    def __init__(
        self,
        address: Optional[int] = None,
        size: Optional[int] = None,
        c_array: Optional[ctypes.Array] = None,
    ) -> None:
        self._refs: List[ctypes.Array] = []
        if c_array is None:
            if address is None or size is None:
                raise ValueError("Expected c_array or address and size args.")
            self.address = address
            self.size = size
            self._set_c_array()
        else:
            self._c_array = c_array
            self.address = ctypes.addressof(c_array)
            self.size = len(c_array)

    def _set_c_array(self) -> None:
        try:
            self._refs.append(self._c_array)
        except AttributeError:
            pass
        self._c_array = (ctypes.c_byte * self.size).from_address(self.address)

    def __add__(self, i: int) -> "Pointer":
        assert i >= 0
        assert i <= self.size
        ret = Pointer(self.address + i, self.size - i)
        ret._refs.append(self._c_array)
        return ret

    def __iadd__(self, i: int) -> "Pointer":
        assert i >= 0
        assert i <= self.size
        self.address += i
        self.size -= i
        self._set_c_array()
        return self

    def __setitem__(self, idx: int, byte: int) -> None:
        self._c_array[idx] = byte

    def __getitem__(self, idx: Union[int, slice]) -> Union[int, "Pointer"]:
        if isinstance(idx, int):
            return self._c_array[idx]
        elif isinstance(idx, slice):
            assert idx.step is None
            start = idx.start
            end = idx.stop
            n = self.size
            if start is None:
                start = 0
            elif start < 0:
                start += n
            if end is None:
                end = n
            elif end < 0:
                end += n
            assert start >= 0 and start < n
            assert end >= start and end <= n
            ret = Pointer(self.address + start, end - start)
            ret._refs.append(self._c_array)
            return ret

    @property
    def memoryview(self):
        return memoryview(self._c_array)

    @property
    def bytes(self):
        return bytes(self._c_array)

    @property
    def bytearray(self):
        return bytearray(self._c_array)

    def __len__(self):
        return self.size


def malloc(size: int) -> Pointer:
    return Pointer(c_array=(ctypes.c_byte * size)())


def memcpy(dest: Pointer, src: Pointer, count=None) -> None:
    if count is None:
        count = src.size
    ctypes.memmove(dest.address, src.address, count)


def _write_pybytes(ptr: Pointer, byts: Union[bytes, memoryview]) -> Pointer:
    memcpy(ptr, _ndarray_to_ptr(np.frombuffer(byts, dtype=np.byte)))
    return ptr + len(byts)


def _ndarray_to_ptr(arr: np.ndarray) -> Pointer:
    return Pointer(arr.__array_interface__["data"][0], arr.itemsize * arr.size)


def _pybytes_to_c_array(byts: bytes) -> Pointer:
    return Pointer(
        np.frombuffer(byts, dtype=np.byte).__array_interface__["data"][0], len(byts)
    )


def _infer_chunk_num_bytes(
    version: str,
    shape_info: np.ndarray,
    byte_positions: np.ndarray,
    data: Optional[Union[Sequence[bytes], Sequence[memoryview]]] = None,
    len_data: Optional[int] = None,
) -> int:
    # NOTE: Assumption: version string contains ascii characters only (ord(c) < 128)
    # NOTE: Assumption: len(version) < 256
    assert len(version) < 256
    assert max((map(ord, version))) < 128
    # assert shape_info.ndim == 2
    # assert byte_positions.ndim == 2
    # version_slice_size = 1 + len(version)
    # shape_info_slice_size = 4 + 4 + shape_info.nbytes
    # byte_positions_slice_size = 4 + 4 + byte_positions.nbytes
    # data_slice_size = sum(map(len, data))
    if len_data is None:
        len_data = sum(map(len, data))  # type: ignore
    return len(version) + shape_info.nbytes + byte_positions.nbytes + len_data + 17


def encode_chunk(
    version: str,
    shape_info: np.ndarray,
    byte_positions: np.ndarray,
    data: Union[Sequence[bytes], Sequence[memoryview]],
    len_data: Optional[int],
) -> memoryview:

    flatbuff = malloc(
        _infer_chunk_num_bytes(version, shape_info, byte_positions, data, len_data)
    )
    ptr = flatbuff + 0

    # write version
    ptr[0] = len(version)
    ptr += 1
    for c in version:
        ptr[0] = ord(c)
        ptr += 1

    # write shape info
    ptr = _write_pybytes(ptr, np.int32(shape_info.shape[0]).tobytes())
    ptr = _write_pybytes(ptr, np.int32(shape_info.shape[1]).tobytes())
    memcpy(ptr, _ndarray_to_ptr(shape_info))
    ptr += shape_info.nbytes

    # write byte positions
    ptr = _write_pybytes(ptr, np.int32(byte_positions.shape[0]).tobytes())
    ptr = _write_pybytes(ptr, np.int32(byte_positions.shape[1]).tobytes())
    memcpy(ptr, _ndarray_to_ptr(byte_positions))
    ptr += byte_positions.nbytes

    # write actual data
    for d in data:
        if isinstance(d, Pointer):
            d = d.memoryview
        ptr = _write_pybytes(ptr, d)

    return memoryview(flatbuff.bytes)


def decode_chunk(
    buff: Union[bytes, Pointer, memoryview]
) -> Tuple[str, np.ndarray, np.ndarray, memoryview]:
    if not isinstance(buff, Pointer):
        buff = _pybytes_to_c_array(buff)
        copy = True
    else:
        copy = False
    ptr = buff + 0

    # read version
    len_version: int = ptr[0]  # type: ignore
    version = ""
    ptr += 1
    for i in range(len_version):
        version += chr(ptr[i])  # type: ignore
    ptr += len_version

    # read shape info
    shape_info_dtype = np.dtype(hub.constants.ENCODING_DTYPE)
    shape_info_shape = np.frombuffer(ptr.memoryview[:8], dtype=np.int32)
    ptr += 8
    shape_info_data_size = int(np.prod(shape_info_shape) * shape_info_dtype.itemsize)
    shape_info = np.frombuffer(
        ptr.memoryview[:shape_info_data_size], dtype=shape_info_dtype
    ).reshape(shape_info_shape)
    if copy:
        shape_info = shape_info.copy()
    ptr += shape_info_data_size

    # read byte positions
    byte_positions_dtype = np.dtype(hub.constants.ENCODING_DTYPE)
    byte_positions_shape = np.frombuffer(ptr.memoryview[:8], dtype=np.int32)
    ptr += 8
    byte_positions_data_size = int(
        np.prod(byte_positions_shape) * byte_positions_dtype.itemsize
    )
    byte_positions = np.frombuffer(
        ptr.memoryview[:byte_positions_data_size], dtype=byte_positions_dtype
    ).reshape(byte_positions_shape)
    if copy:
        byte_positions = byte_positions.copy()
    ptr += byte_positions_data_size
    if copy:
        data = memoryview(ptr.bytes)
    else:
        data = ptr.memoryview
    return version, shape_info, byte_positions, data


def encode_chunkids(version: str, ids: Sequence[np.ndarray]) -> memoryview:
    len_version = len(version)
    flatbuff = malloc(1 + len_version + sum([x.nbytes for x in ids]))

    # Write version
    ptr = flatbuff + 0
    ptr[0] = len_version
    ptr += 1

    for i, c in enumerate(version):
        ptr[i] = ord(c)

    ptr += len_version

    for arr in ids:
        memcpy(ptr, _ndarray_to_ptr(arr))
        ptr += arr.nbytes

    return memoryview(flatbuff.bytes)


def decode_chunkids(buff: bytes) -> Tuple[str, np.ndarray]:
    ptr = _pybytes_to_c_array(buff)

    # Read version
    len_version: int = ptr[0]  # type: ignore
    ptr += 1
    version = ""
    for i in range(len_version):
        version += chr(ptr[i])  # type: ignore

    ptr += len_version

    # Read chunk ids
    ids = (
        np.frombuffer(ptr.memoryview, dtype=hub.constants.ENCODING_DTYPE)
        .reshape(-1, 2)
        .copy()
    )

    return version, ids


def test_chunk_encoding():
    version = hub.__version__
    shape_info = np.cast[hub.constants.ENCODING_DTYPE](
        np.random.randint(100, size=(17, 63))
    )
    byte_positions = np.cast[hub.constants.ENCODING_DTYPE](
        np.random.randint(100, size=(31, 79))
    )
    data = [b"1234" * 7, b"abcdefg" * 8, b"qwertyuiop" * 9]
    encoded = bytes(encode_chunk(version, shape_info, byte_positions, data))

    # from bytes
    decoded = decode_chunk(encoded)
    version2, shape_info2, byte_positions2, data2 = decoded
    assert version2 == version
    np.testing.assert_array_equal(shape_info, shape_info2)
    np.testing.assert_array_equal(byte_positions, byte_positions2)
    assert b"".join(data) == bytes(data2)

    # from pointer
    buff = Pointer(c_array=(ctypes.c_byte * len(encoded))(*encoded))
    decoded = decode_chunk(buff)
    version2, shape_info2, byte_positions2, data2 = decoded
    assert version2 == version
    np.testing.assert_array_equal(shape_info, shape_info2)
    np.testing.assert_array_equal(byte_positions, byte_positions2)
    assert b"".join(data) == bytes(data2)


def test_chunkids_encoding():
    version = hub.__version__
    shards = [
        np.cast[hub.constants.ENCODING_DTYPE](np.random.randint(100, size=(100, 2)))
    ]
    encoded = encode_chunkids(version, shards)
    decoded = decode_chunkids(encoded)
    version2, ids = decoded
    assert version2 == version
    np.testing.assert_array_equal(np.concatenate(shards), ids)


if __name__ == "__main__":
    test_chunk_encoding()
    test_chunkids_encoding()
