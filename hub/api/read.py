from hub.core.sample import Sample  # type: ignore
from typing import Optional, Dict


def read(
    path: str,
    verify: bool = False,
    creds: Optional[Dict] = None,
    compression: Optional[str] = None,
) -> Sample:
    """Utility that reads raw data from a file into a `np.ndarray` in 1 line of code. Also provides access to all important metadata.

    Note:
        No data is actually loaded until you try to get a property of the returned `Sample`. This is useful for passing along to
            `tensor.append` and `tensor.extend`.

    Examples:
        >>> sample = hub.read("path/to/cat.jpeg")
        >>> type(sample.array)
        <class 'numpy.ndarray'>
        >>> sample.compression
        'jpeg'

    Supported file types:
        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"
        Audio: "flac", "mp3", "wav"

    Args:
        path (str): Path to a supported file.
        verify (bool):  If True, contents of the file are verified.
        creds (optional, Dict): Credentials for s3 and gcp for urls.
        compression (optional, str): Format of the file (see `hub.compression.SUPPORTED_COMPRESSIONS`). Only required if path does not have an extension.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.
    """
    return Sample(path, verify=verify, compression=compression, creds=creds)
