import pathlib
from deeplake.core.sample import Sample  # type: ignore
from typing import Optional, Dict, Union

from deeplake.core.storage.provider import StorageProvider
from deeplake.util.path import convert_pathlib_to_string_if_needed


def read(
    path: Union[str, pathlib.Path],
    verify: bool = False,
    creds: Optional[Dict] = None,
    compression: Optional[str] = None,
    storage: Optional[StorageProvider] = None,
) -> Sample:
    """Utility that reads raw data from supported files into Deep Lake format.

    - Recompresses data into format required by the tensor if permitted by the tensor htype.
    - Simply copies the data in the file if file format matches sample_compression of the tensor, thus maximizing upload speeds.

    Examples:

        >>> ds.create_tensor("images", htype="image", sample_compression="jpeg")
        >>> ds.images.append(deeplake.read("path/to/cat.jpg"))
        >>> ds.images.shape
        (1, 399, 640, 3)

        >>> ds.create_tensor("videos", htype="video", sample_compression="mp4")
        >>> ds.videos.append(deeplake.read("path/to/video.mp4"))
        >>> ds.videos.shape
        (1, 136, 720, 1080, 3)

        >>> ds.create_tensor("images", htype="image", sample_compression="jpeg")
        >>> ds.images.append(deeplake.read("https://picsum.photos/200/300"))
        >>> ds.images[0].shape
        (300, 200, 3)

    Supported file types::

        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"
        Audio: "flac", "mp3", "wav"
        Video: "mp4", "mkv", "avi"
        Dicom: "dcm"

    Args:
        path (str): Path to a supported file.
        verify (bool):  If True, contents of the file are verified.
        creds (optional, Dict): Credentials for s3, gcp and http urls.
        compression (optional, str): Format of the file. Only required if path does not have an extension.
        storage (optional, StorageProvider): Storage provider to use to retrieve remote files. Useful if multiple files are being read from same storage to minimize overhead of creating a new provider.

    Returns:
        Sample: Sample object. Call ``sample.array`` to get the ``np.ndarray``.

    Note:
        No data is actually loaded until you try to get a property of the returned :class:`Sample`.
        This is useful for passing along to :func:`Tensor.append <deeplake.core.tensor.Tensor.append>` and :func:`Tensor.extend <deeplake.core.tensor.Tensor.extend>`.
    """
    path = convert_pathlib_to_string_if_needed(path)
    return Sample(
        path, verify=verify, compression=compression, creds=creds, storage=storage
    )
