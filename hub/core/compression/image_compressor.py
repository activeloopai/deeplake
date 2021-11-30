from hub.core.compression.compressor import BaseCompressor
from hub.core.compression.compressions import IMAGE_COMPRESSIONS
from PIL import Image
from typing import List, Tuple


class ImageCompressor(BaseCompressor):
    supported_compressions = IMAGE_COMPRESSIONS
    def __init__(self, compression: str):
        super(ImageCompressor, self).__init__(compression)

    def compress(self, data: np.ndarray) -> bytes:
        compression = self.compression
        if compression == "apng":
            raise NotImplementedError()
        shape = data.shape
        if len(shape) == 3 and shape[0] != 1 and shape[2] == 1:
            # convert (X,Y,1) grayscale to (X,Y) for pillow compatibility
            data = data.squeeze(axis=2)
        img = Image.fromarray(data)
        try:
            out = BytesIO()
            out._close = out.close  # type: ignore
            out.close = (  # type: ignore
                lambda: None
            )  # sgi save handler will try to close the stream (see https://github.com/python-pillow/Pillow/pull/5645)
            kwargs = {"sizes": [img.size]} if compression == "ico" else {}
            img.save(out, compression, **kwargs)
            out.seek(0)
            compressed_bytes = out.read()
            out._close()  # type: ignore
            return compressed_bytes
        except (TypeError, OSError) as e:
            raise SampleCompressionError(shape, compression, str(e))

    def decompress(self, compressed: bytes, shape: Tuple[int, ...] = None) -> np.ndarray:
        try:
            if not isinstance(compressed, str):
                compressed = BytesIO(compressed)  # type: ignore
            img = Image.open(compressed)  # type: ignore
            arr = np.array(img)
            if shape is not None:
                arr = arr.reshape(shape)
            return arr
        except Exception:
            raise SampleDecompressionError()

    def _get_bounding_shape(self, shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, int, int]:
        """Gets the shape of a bounding box that can contain the given the shapes tiled horizontally."""
        if len(shapes) == 0:
            return (0, 0, 0)
        channels_shape = shapes[0][2:]
        for shape in shapes:
            if shape[2:] != channels_shape:
                raise ValueError(
                    "The data can't be compressed as the number of channels doesn't match."
                )
        return (max(s[0] for s in shapes), sum(s[1] for s in shapes)) + channels_shape  # type: ignore

    def compress_multiple(self, data: List[np.ndarray]) -> bytes:
        arrays = data
        dtype = arrays[0].dtype
        for arr in arrays:
            if arr.dtype != dtype:
                raise SampleCompressionError(
                    arr.shape,
                    compression,
                    message="All arrays expected to have same dtype.",
                )
        canvas = np.zeros(self._get_bounding_shape([arr.shape for arr in arrays]), dtype=dtype)
        next_x = 0
        for arr in arrays:
            canvas[: arr.shape[0], next_x : next_x + arr.shape[1]] = arr
            next_x += arr.shape[1]
        return self.compress(canvas)

    def decompress_multiple(self, compressed: bytes, shape: List[Tuple[int, ...]]) -> List[np.ndarray]:
        canvas = self.decompress(compressed)
        arrays = []
        next_x = 0
        for shape in shapes:
            arrays.append(canvas[: shape[0], next_x : next_x + shape[1]])
            next_x += shape[1]
        return arrays

    def verify(self):
        pass  # TODO
