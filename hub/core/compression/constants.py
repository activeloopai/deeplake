import numcodecs  # type: ignore

MSGPACK = numcodecs.MsgPack()
WEBP_COMPRESSOR_NAME = "webp"
IMAGE_SHAPE_ERROR_MESSAGE = (
    "The shape length {len(arr.shape)} of the given array should "
    "be greater than the number of expected dimensions "
)
