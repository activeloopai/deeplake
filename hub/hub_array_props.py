from typing import Tuple

class HubArrayProps():
    shape: Tuple[int, ...] = None
    chunk: Tuple[int, ...] = None
    dtype: str = None
    compress: str = None
    compresslevel: float = 0.5