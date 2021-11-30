from typing import List

class BaseCompressor(object):
    # Compressions supported by each compressor. Must be overridden.
    supported_compressions = []

    def __init__(self, compression: str):
        if compression not in self.supported_compressions:
            raise ValueError("Unsupported compression: {compression}. Supported compressions are: {self.supported_compressions}.")
        self.compression = compression

    def compress(slef, data: Any) -> Any:
        return data

    def decompress(self, compressed: Any) -> Any:
        return compressed

    def compress_multiple(self, data: List[Any]) -> Any:
        return data

    def decompress_multiple(self, compressed: Any, *kwargs) -> List[Any]:
        return list()


    def accepts(self, data: Any) -> bool:
        return True

    def verify(self, compressed: Any):
        self.decompress(compressed)
