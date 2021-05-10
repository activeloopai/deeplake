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
    compressor = compressor.lower()
    if compressor == "lz4":
        acceleration = kwargs.get("acceleration", numcodecs.lz4.DEFAULT_ACCELERATION)
        return Lz4(acceleration)
    elif compressor == "zstd":
        level = kwargs.get("level", numcodecs.zstd.DEFAULT_CLEVEL)
        return Zstd(level)
    elif compressor == "numpy":
        return NumPy()
    elif compressor == "png":
        single_channel = kwargs.get("single_channel", True)
        return PngCodec(single_channel=single_channel)
    elif compressor == "jpeg":
        single_channel = kwargs.get("single_channel", True)
        quality = kwargs.get("quality", 95)
        return JpegCodec(single_channel=single_channel, quality=quality)
    else:
        raise ValueError(
            f"Wrong compressor: {compressor}, only LZ4, PNG, JPEG, NumPy and ZSTD are supported"
        )
