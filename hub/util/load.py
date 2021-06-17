import re
import numpy as np
import pathlib
import exiftool  # type: ignore
from typing import Callable, Dict, List, Union
from hub.util.exceptions import (
    HubAutoUnsupportedFileExtensionError,
    SampleCorruptedError,
    ImageReadError,
)
from PIL import Image  # type: ignore


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]
SUPPORTED_SUFFIXES: List[str] = IMAGE_SUFFIXES


class SymbolicSample:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._array = None

    @property
    def name(self):
        return self.path.split("/")[-1]

    @property
    def shape(self):
        # TODO: raise good exception if `numpy` wasn't called
        return self._array.shape

    @property
    def dtype(self):
        # TODO: raise good exception if `numpy` wasn't called
        return self._array.dtype.name

    @property
    def compression(self) -> str:
        # TODO: raise exception if `numpy` wasn't called
        return self._compression

    def raw_bytes(self):
        with open(self.path, "rb") as image_file:
            return image_file.read()

    def numpy(self) -> np.ndarray:
        # TODO: explain this
        if self._array is not None:
            return self._array

        suffix = self.path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
            img = Image.open(self.path)

            # TODO: mention in docstring that if this loads correctly, the meta is assumed to be valid
            try:
                self._array = np.array(img)
            except:
                # TODO: elaborate on why it corrupted
                raise SampleCorruptedError(self.path)

            # TODO: set meta values
            self._compression = img.format
            return self._array

        raise HubAutoUnsupportedFileExtensionError(self._suffix, SUPPORTED_SUFFIXES)

    # TODO: __str__


def load(path: Union[str, pathlib.Path]) -> SymbolicSample:
    # TODO: mention that you can do `.numpy()` on this output to make it extremely easy to use
    return SymbolicSample(path)


# def get_png_channels(color_type: int) -> int:
#     """
#     Get number of image channels from png color type.
#
#     Args:
#         color_type (int): Png color type value from metainfo
#
#     Returns:
#         Number of png image channels.
#     """
#     if color_type == 6:
#         return 4
#     elif color_type == 4:
#         return 2
#     elif color_type == 2:
#         return 3
#     return 1


# def check_image_meta(image_path: str, **kwargs):
#     """
#     Check if image metadata correesponds to actual image params.
#
#     Args:
#         image_path: Path to image to be checked
#         kwargs: meta_size, meta_channels, meta_extension and meta dtype from the image metainfo.
#
#     Raises:
#         ImageReadError: If image can't be opened by PIL.Image.open()
#         SampleCorruptedError: If any parameter from metadata doesn't match the image.
#     """
#     try:
#         image = Image.open(image_path)
#     except Exception as e:
#         raise ImageReadError(image_path, e)
#     image_arr = np.asarray(image)
#     image_dtype = image_arr.dtype
#     if image.mode == "RGB":
#         image_channels = 3
#     elif image.mode == "RGBA":
#         image_channels = 4
#     else:
#         image_channels = 1
#     image_size = image.size
#     image_extension = image.format
#     if (
#         kwargs["meta_size"] != image_size
#         or kwargs["meta_extension"] != image_extension
#         or kwargs["meta_channels"] != image_channels
#         or kwargs["meta_dtype"] != image_dtype
#     ):
#         raise SampleCorruptedError(image_path)


# def read(image_path: str, check_meta: bool = True):
#     """
#     Get image bytes and metadata.
#
#     Args:
#         image_path (str): Path to the image file.
#         check_meta (bool): If True, check if image can be read and image metadata
#             information corresponds to the actual image parameters.
#
#     Returns:
#         Dictionary containing image bytes, extension, dtype and shape.
#     """
#     # TODO: merge into `load` method as `symbolic`
#
#     with exiftool.ExifTool() as et:
#         metadata = et.get_metadata(image_path)
#     with open(image_path, "rb") as image_file:
#         image_bytes = image_file.read()
#     meta_size = tuple(map(int, re.split("x| ", metadata["Composite:ImageSize"])))
#     for meta_key, meta_value in metadata.items():
#         if meta_key.endswith("FileType"):
#             meta_extension = meta_value
#         elif "ColorComponents" in meta_key:
#             meta_channels = meta_value
#         elif "PNG:ColorType" in meta_key:
#             color_type = int(meta_value)
#             meta_channels = get_png_channels(color_type)
#         elif "BitDepth" in meta_key or "BitsPerSample" in meta_key:
#             meta_dtype = "uint" + str(meta_value)
#     if check_meta:
#         check_image_meta(
#             image_path,
#             meta_channels=meta_channels,
#             meta_dtype=meta_dtype,
#             meta_extension=meta_extension,
#             meta_size=meta_size,
#         )
#     return {
#         "bytes": image_bytes,
#         "name": image_path.split("/")[-1],
#         "dtype": meta_dtype,
#         "shape": meta_size + (meta_channels,),
#         "compression": meta_extension,
#     }
#
