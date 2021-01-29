"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple

from hub.schema.features import Tensor


class Audio(Tensor):
    def __init__(
        self,
        shape: Tuple[int, ...] = (None,),
        dtype="int64",
        file_format=None,
        sample_rate: int = None,
        max_shape: Tuple[int, ...] = None,
        chunks=None,
        compressor="lz4",
    ):
        """Constructs the connector.

        Parameters
        ----------
        file_format: `str`
            the audio file format. Can be any format ffmpeg
            understands. If `None`, will attempt to infer from the file extension.
        shape: `tuple`
             shape of the data.
        dtype: str
            The dtype of the data.
        sample_rate: `int`
            additional metadata exposed to the user through
            `info.schema['audio'].sample_rate`. This value isn't used neither in
            encoding nor decoding.


        Raises
        ----------
        ValueError: If the shape is invalid
        """
        self.file_format = file_format
        if len(shape) != 1:
            raise ValueError(
                f"Audio schema currently only supports 1-D values, got {shape}"
            )
        # self._shape = shape
        self.sample_rate = sample_rate
        super().__init__(
            shape=shape,
            dtype=dtype,
            max_shape=max_shape,
            chunks=chunks,
            compressor=compressor,
        )

    def __str__(self):
        out = super().__str__()
        out = "Audio" + out[6:-1]
        out = (
            out + ", file_format=" + self.file_format
            if self.file_format is not None
            else out
        )
        out = (
            out + ", sample_rate=" + str(self.sample_rate)
            if self.sample_rate is not None
            else out
        )
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()
