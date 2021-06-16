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


def _load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    return np.array(img)


def load(path: Union[str, pathlib.Path], symbolic=False) -> Union[Callable, np.ndarray]:
    path = pathlib.Path(path)

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        if symbolic:
            # TODO: symbolic loading (for large samples)
            raise NotImplementedError("Symbolic `hub.load` not implemented.")
        return _load_image(path)

    raise HubAutoUnsupportedFileExtensionError(suffix, SUPPORTED_SUFFIXES)


def read(image_path: str, check_meta: bool = True):
    """
    Get image bytes and metadata.

    Args:
        image_path (str): Path to the image file.
        check_meta (bool): If True, check if image can be read and image metadata
            information corresponds to the actual image parameters.

    Raises:
        ImageReadError: If image can't be opened by PIL.Image.open()
        WrongMetadataError: If any parameter from metadata doesn't match the image.

    Returns:
        Dictionary containing image bytes, extension, dtype and shape.
    """
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(image_path)
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    meta_size = tuple(map(int, re.split("x| ", metadata["Composite:ImageSize"])))
    for meta_key, meta_value in metadata.items():
        if meta_key.endswith("FileType"):
            meta_extension = meta_value
        elif "ColorComponents" in meta_key:
            meta_channels = meta_value
        elif "PNG:ColorType" in meta_key:
            color_type = int(meta_value)
            if color_type == 6:
                meta_channels = 4
            elif color_type == 4:
                meta_channels = 2
            elif color_type == 2:
                meta_channels = 3
            else:
                meta_channels = 1
        elif "BitDepth" in meta_key or "BitsPerSample" in meta_key:
            meta_dtype = "uint" + str(meta_value)
    if check_meta:
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ImageReadError(image_path, e)
        image_arr = np.asarray(image)
        image_dtype = image_arr.dtype
        if image.mode == "RGB":
            image_channels = 3
        elif image.mode == "RGBA":
            image_channels = 4
        else:
            image_channels = 1
        image_size = image.size
        image_extension = image.format
        if (
            meta_size != image_size
            or meta_extension != image_extension
            or meta_channels != image_channels
            or meta_dtype != image_dtype
        ):
            raise SampleCorruptedError(image_path)
    return {
        "bytes": image_bytes,
        "name": image_path.split("/")[-1],
        "dtype": meta_dtype,
        "shape": meta_size + (meta_channels,),
        "compression": meta_extension,
        "is_compressed": True,
    }
