# FIXME NOT WORKING YET, NEED SOME CHANGES
# DONT INCLUDE INTO __init__.py YET
from typing import Tuple

from hub.features.image import Image
from hub.features.sequence import Sequence


class Video(Sequence):
    """`HubFeature` for videos, encoding frames individually on disk.

    The connector accepts as input a 4 dimensional `uint8` array
    representing a video.

    Returns
    ----------
    Tensor: `uint8` and shape [num_frames, height, width, channels],
         where channels must be 1 or 3
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype: str = "uint8",
        encoding_format: str = "png",
        chunks=True,
    ):
        """Initializes the connector.

        Parameters
        ----------

        shape: tuple of ints
            The shape of the video (num_frames, height, width,
            channels), where channels is 1 or 3.
        encoding_format: str
            The video is stored as a sequence of encoded images.
            You can use any encoding format supported by Image.
        dtype: `uint16` or `uint8` (default)

        Raises
        ----------
        ValueError: If the shape, dtype or encoding formats are invalid
        """
        super(Video, self).__init__(
            feature=Image(
                shape=shape[1:], dtype=dtype, encoding_format=encoding_format
            ),
            length=shape[0],
            chunks=chunks,
        )

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
