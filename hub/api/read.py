from hub.core.sample import Sample  # type: ignore
from typing import Optional, Dict

from hub.core.storage.provider import StorageProvider


def read(
    path: str,
    verify: bool = False,
    creds: Optional[Dict] = None,
    compression: Optional[str] = None,
    storage: Optional[StorageProvider] = None,
) -> Sample:
    """Utility that reads raw data from supported files into hub format.

    - Recompresses data into format required by the tensor if permitted by the tensor htype.
    - Simply copies the data in the file if file format matches sample_compression of the tensor, thus maximizing upload speeds.

    Note:
        No data is actually loaded until you try to get a property of the returned `Sample`. This is useful for passing along to
            `tensor.append` and `tensor.extend`.

    Examples:
        >>> ds.create_tensor("images", htype="image", sample_compression="jpeg")
        >>> ds.images.append(hub.read("path/to/cat.jpg"))
        >>> ds.images.shape
        (1, 399, 640, 3)

        >>> ds.create_tensor("videos", htype="video", sample_compression="mp4")
        >>> ds.videos.append(hub.read("path/to/video.mp4"))
        >>> ds.videos.shape
        (1, 136, 720, 1080, 3)

        >>> ds.create_tensor("images", htype="image", sample_compression="jpeg")
        >>> ds.images.append(hub.read("https://picsum.photos/200/300"))
        >>> ds.images[0].shape
        (300, 200, 3)

    Supported file types:

        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"
        Audio: "flac", "mp3", "wav"
        Video: "mp4", "mkv", "avi"
        Dicom: "dcm"

    Args:
        path (str): Path to a supported file.
        verify (bool):  If True, contents of the file are verified.
        creds (optional, Dict): Credentials for s3 and gcp for urls.
        compression (optional, str): Format of the file (see `hub.compression.SUPPORTED_COMPRESSIONS`). Only required if path does not have an extension.
        storage (optional, StorageProvider): Storage provider to use to retrieve remote files. Useful if multiple files are being read from same storage to minimize overhead of creating a new provider.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.
    """
    return Sample(
        path, verify=verify, compression=compression, creds=creds, storage=storage
    )
