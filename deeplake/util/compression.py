from typing import Optional


compression_ratios = {None: 1.0, "jpeg": 0.5, "png": 0.5, "webp": 0.5, "lz4": 0.5}


def get_compression_ratio(compression: Optional[str]) -> float:
    return compression_ratios.get(compression, 0.5)
