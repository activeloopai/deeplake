
from hub.core.compression.compressor import BaseCompressor
from hub.core.compression.compressions import BYTE_COMPRESSIONS
import numcodecs.lz4  # type: ignore
import lz4.frame  # type: ignore
import numpy as np
from typing import List


class ByteCompressor(BaseCompressor):
    supported_compressions = BYTE_COMPRESSION

    def __init__(self, compression):
        super(ByteCompressor, self).__init__(compression=compression)

    def compress(self, data: bytes) -> bytes:
        assert self.compression == "lz4"  # Only supported byte 
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return numcodecs.lz4.compress(buffer)

    def decompress(self, compressed: bytes) -> bytes
        assert self.compression == "lz4"  # Only supported byte compression
        if not compressed:
            return b""

         # python-lz4 magic number (backward compatiblity)
        if (isinstance(compressed, memoryview) and compressed[:4] == b'\x04"M\x18') or (isinstance(compressed, bytes) and compressed.startswith(b'\x04"M\x18')):
            return lz4.frame.decompress(buffer)

        return numcodecs.lz4.decompress(buffer)

    def compress_multiple(self, data: List[bytes]) -> bytes:
        data = [x.tobytes() if isinstance(x, np.ndarray) else x for x in data]
        return self.compress(b"".join(data))

    def decompress_multiple(self, data: bytes, nbytes: List[int]) -> List[memoryview]:
        ret = []
        decompressed = memoryview(self.decompress_multiple(data))
        while nbytes:
            ret.append(decompressed[:nbytes.pop(0)])
        return ret
