from hub.core.linked_sample import LinkedSample
from typing import Optional, Dict


def link(
    path: str,
    creds_key: Optional[str] = None,
) -> LinkedSample:
    """Utility that stores a link to raw data. Used to add data to a Hub Dataset without copying it. See :ref:`Link htype`.

    Supported file types::

        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"
        Audio: "flac", "mp3", "wav"
        Video: "mp4", "mkv", "avi"
        Dicom: "dcm"

    Args:
        path (str): Path to a supported file.
        creds_key (optional, str): The credential key to use to read data for this sample. The actual credentials are fetched from the dataset.

    Returns:
        LinkedSample: LinkedSample object that stores path and creds.
    """
    return LinkedSample(path, creds_key)
