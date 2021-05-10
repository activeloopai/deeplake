import numcodecs
from hub.codec.codecs import PngCodec, Lz4, NumPy, Zstd, JpegCodec


def _get_compressor(compressor: str, **kwargs):
    """Get compressor object

    Example:
        compressor = _get_compressor('lz4', acceleration=2)

    Args:
        compressor (str): lowercase name of the required compressor

    Returns:
        Codec object providing compression

    Raises:
        ValueError: if the name of compressor isn't in ['lz4', 'zstd', 'numpy', 'png', 'jpeg']
    """
    if compressor is None:
        return None
    elif compressor.lower() == "lz4":
        acceleration = kwargs.get("acceleration", numcodecs.lz4.DEFAULT_ACCELERATION)
        return Lz4(acceleration)
    elif compressor.lower() == "zstd":
        level = kwargs.get("level", numcodecs.zstd.DEFAULT_CLEVEL)
        return Zstd(level)
    elif compressor.lower() == "numpy":
        return NumPy()
    elif compressor.lower() == "png":
        single_channel = kwargs.get("single_channel", True)
        return PngCodec(single_channel=single_channel)
    elif compressor.lower() == "jpeg":
        single_channel = kwargs.get("single_channel", True)
        return JpegCodec(single_channel=single_channel)
    else:
        raise ValueError(
            f"Wrong compressor: {compressor}, only LZ4, PNG and ZSTD are supported"
        )
