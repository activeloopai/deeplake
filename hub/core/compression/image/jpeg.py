import mmap
from typing import Tuple
from hub.core.compression import STRUCT_HHB, BaseCompressor, re_find_first
import re


_JPEG_SOFS = [
    b"\xff\xc0",
    b"\xff\xc2",
    b"\xff\xc1",
    b"\xff\xc3",
    b"\xff\xc5",
    b"\xff\xc6",
    b"\xff\xc7",
    b"\xff\xc9",
    b"\xff\xca",
    b"\xff\xcb",
    b"\xff\xcd",
    b"\xff\xce",
    b"\xff\xcf",
    b"\xff\xde",
    # Skip:
    b"\xff\xcc",
    b"\xff\xdc",
    b"\xff\xdd",
    b"\xff\xdf",
    # App: (0xFFE0 - 0xFFEF):
    *map(lambda x: x.to_bytes(2, "big"), range(0xFFE0, 0xFFF0)),
    # DQT:
    b"\xff\xdb",
    # COM:
    b"\xff\xfe",
    # Start of scan
    b"\xff\xda",
]

_JPEG_SKIP_MARKERS = set(_JPEG_SOFS[14:])
_JPEG_SOFS_RE = re.compile(b"|".join(_JPEG_SOFS))


_JPEG_DTYPE = "|u1"


class JPEG(BaseCompressor):
    def verify(self):
        # TODO: docstring

        if self.file is not None:
            return _verify_jpeg_file(self.file), _JPEG_DTYPE

        return _verify_jpeg_buffer(self.buffer), _JPEG_DTYPE

    def read_shape_and_dtype(self):
        # TODO: formerly _read_jpeg_shape

        if self.file is not None:
            return _read_jpeg_shape_from_file(self.file), _JPEG_DTYPE

        return _read_jpeg_shape_from_buffer(self.buffer), _JPEG_DTYPE  # type: ignore


def _verify_jpeg_buffer(buf: bytes):
    # Start of Image
    mview = memoryview(buf)
    assert buf.startswith(b"\xff\xd8")
    # Look for Start of Frame
    sof_idx = -1
    offset = 0
    while True:
        match = re_find_first(_JPEG_SOFS_RE, mview[offset:])
        if match is None:
            break
        idx = match.start(0) + offset
        marker = buf[idx : idx + 2]
        if marker == _JPEG_SOFS[-1]:
            break
        elif marker in _JPEG_SKIP_MARKERS:
            offset = idx + int.from_bytes(buf[idx + 2 : idx + 4], "big")
        else:
            sof_idx = idx
            offset = idx + 2
    if sof_idx == -1:
        raise Exception()

    length = int.from_bytes(mview[sof_idx + 2 : sof_idx + 4], "big")
    assert mview[sof_idx + length + 2 : sof_idx + length + 4] in [
        b"\xff\xc4",
        b"\xff\xdb",
        b"\xff\xdd",
    ]  # DHT, DQT, DRI
    shape = STRUCT_HHB.unpack(mview[sof_idx + 5 : sof_idx + 10])
    assert buf.find(b"\xff\xd9") != -1
    if shape[-1] in (1, None):
        shape = shape[:-1]
    return shape


def _read_jpeg_shape_from_file(f) -> Tuple[int, ...]:
    """Reads shape of a jpeg image from file without loading the whole image in memory"""
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    mv = memoryview(mm)
    try:
        # Look for Start of Frame
        sof_idx = -1
        offset = 0
        while True:
            view = mv[offset:]
            match = re_find_first(_JPEG_SOFS_RE, view)
            view.release()
            if match is None:
                break
            idx = match.start(0) + offset
            marker = mm[idx : idx + 2]
            if marker == _JPEG_SOFS[-1]:
                break
            elif marker in _JPEG_SKIP_MARKERS:
                f.seek(idx + 2)
                offset = idx + int.from_bytes(f.read(2), "big")
            else:
                sof_idx = idx
                offset = idx + 2
        if sof_idx == -1:
            raise Exception()
        f.seek(sof_idx + 5)
        shape = STRUCT_HHB.unpack(f.read(5))  # type: ignore
        if shape[-1] in (1, None):
            shape = shape[:-1]
        return shape
    finally:
        mv.release()
        mm.close()


def _read_jpeg_shape_from_buffer(buf: bytes) -> Tuple[int, ...]:
    """Gets shape of a jpeg file from its contents"""
    # Look for Start of Frame
    mv = memoryview(buf)
    sof_idx = -1
    offset = 0
    while True:
        match = re_find_first(_JPEG_SOFS_RE, mv[offset:])
        if match is None:
            break
        idx = match.start(0) + offset
        marker = buf[idx : idx + 2]
        if marker == _JPEG_SOFS[-1]:
            break
        elif marker in _JPEG_SKIP_MARKERS:
            offset = idx + int.from_bytes(buf[idx + 2 : idx + 4], "big")
        else:
            sof_idx = idx
            offset = idx + 2
    if sof_idx == -1:
        raise Exception()
    shape = STRUCT_HHB.unpack(memoryview(buf)[sof_idx + 5 : sof_idx + 10])  # type: ignore
    if shape[-1] in (1, None):
        shape = shape[:-1]
    return shape


def _verify_jpeg_file(f):
    # See: https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files#2-The-metadata-structure-in-JPEG
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mv = memoryview(mm)
    try:
        soi = f.read(2)
        # Start of Image
        assert soi == b"\xff\xd8"

        # Look for Start of Frame
        sof_idx = -1
        offset = 0
        while True:
            view = mv[offset:]
            match = re_find_first(_JPEG_SOFS_RE, view)
            view.release()
            if match is None:
                break
            idx = match.start(0) + offset
            marker = mm[idx : idx + 2]
            if marker == _JPEG_SOFS[-1]:
                break
            elif marker in _JPEG_SKIP_MARKERS:
                f.seek(idx + 2)
                offset = idx + int.from_bytes(f.read(2), "big")
            else:
                sof_idx = idx
                offset = idx + 2
        if sof_idx == -1:
            raise Exception()  # Caught by verify_compressed_file()

        f.seek(sof_idx + 2)
        length = int.from_bytes(f.read(2), "big")
        f.seek(length - 2, 1)
        definition_start = f.read(2)
        assert definition_start in [
            b"\xff\xc4",
            b"\xff\xdb",
            b"\xff\xdd",
        ]  # DHT, DQT, DRI
        f.seek(sof_idx + 5)
        shape = STRUCT_HHB.unpack(f.read(5))
        # TODO this check is too slow
        assert mm.find(b"\xff\xd9") != -1  # End of Image
        if shape[-1] in (1, None):
            shape = shape[:-1]
        return shape
    finally:
        mv.release()
        mm.close()
