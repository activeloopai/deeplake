from hub.core.sample import Sample  # type: ignore
from hub.constants import UNSPECIFIED
from hub.util.path import get_path_type, is_remote_path
from typing import Optional, Dict


def read(
    path: str,
    verify: bool = False,
    creds: Optional[Dict] = None,
    compression: Optional[str] = UNSPECIFIED,
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
        compression (optional, str): Format of the file (see `hub.compression.SUPPORTED_COMPRESSIONS`). Only required for remote urls.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.

    Raises:
        Exception: If compression argument is not specified for remote urls.
    """

    if is_remote_path(path):
        if compression == UNSPECIFIED:
            raise Exception(
                "compression argument should be specified while reading remote urls with hub.read()."
            )
    else:
        compression = None
    sample = Sample(path, verify=verify, compression=compression)
    return sample
