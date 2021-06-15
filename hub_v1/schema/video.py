"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple

from hub_v1.schema import Tensor


class Video(Tensor):
    """`HubSchema` for videos, encoding frames individually on disk.

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
        # TODO Add back encoding_format (probably named compress) when support for png and jpg support will be added
        max_shape: Tuple[int, ...] = None,
        # ffmpeg_extra_args=(),
        chunks=None,
        compressor="lz4",
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
            dtype=dtype,
            shape=shape,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def __str__(self):
        out = super().__str__()
        out = "Video" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
