from typing import Union

from hub.core import compression
from hub.util.exceptions import InvalidCompressor


def get_compressor(
    compressor_name: str, **kwargs
) -> Union[compression.BaseImgCodec, compression.BaseNumCodec]:
    """Get compressor object

    Example:
        compressor = get_compressor('LZ4', acceleration=2)

    Args:
        compressor_name (str): name of the required compressor
        **kwargs: Optional;
            acceleration (int): Lz4 codec argument. The larger the acceleration value,
            the faster the algorithm, but also the lesser the compression.
            level (int): Zstd codec argument. Sets the compression level (1-22).
            single_channel (bool): PngCodec and JpegCodec argument. if True,
            encoder will remove the last dimension of input if it is 1.
            Doesn't have impact on 3-channel images.
            quality (int): JpegCodec argument. The image quality on a scale from 1 (worst) to 95 (best).

    Returns:
        Compressor object.

    Raises:
        InvalidCompressor: if the compressor_name is not in supported compressors.
    """
    try:
        compressor = compression.__dict__[compressor_name]
        return compressor(**kwargs)
    except KeyError:
        raise InvalidCompressor(compression.AVAILABLE_COMPRESSORS)
