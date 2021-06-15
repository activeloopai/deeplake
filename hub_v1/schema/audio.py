"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import Tuple

from hub_v1.schema.features import Tensor


class Audio(Tensor):

    """Schema for audio would define the maximum shape of the audio dataset and their respective sampling rate.

    Example: This example uploads an `audio file` to a Hub dataset `audio_dataset` with `HubSchema` and retrieves it.

    ----------
    >>> import hub_v1
    >>> from hub_v1.schema import Audio
    >>> from hub_v1 import transform, schema
    >>> import librosa
    >>> from librosa import display
    >>> import numpy as np

    >>> # Define schema
    >>> my_schema={
    >>>     "wav": Audio(shape=(None,), max_shape=(1920000,), file_format="wav", dtype=float),
    >>>     "sampling_rate": Primitive(dtype=int),
    >>> }
    >>>
    >>> sample = glob("audio.wav")

    >>> # Define transform
    >>> @transform(schema=my_schema)
    >>> def load_transform(sample):
    >>>     audio, sr = librosa.load(sample, sr=None)
    >>>
    >>>     return {
    >>>         "wav": audio,
    >>>         "sampling_rate": sr
    >>>     }
    >>>
    >>> # Returns a transform object
    >>> ds = load_transform(sample)

    >>> # Load data
    >>> ds = Dataset(tag)
    >>>
    >>> tag = "username/audio_dataset"
    >>>
    >>> # Pushes to hub
    >>> ds2 = ds.store(tag)

    >>> # Fetching from hub_v1
    >>> data = Dataset(tag)
    >>>
    >>> # Fetch the first sample
    >>> audio_sample = data["wav"][0].compute()
    >>>
    >>> # Audio file
        array([ 9.15527344e-05,  2.13623047e-04,  0.00000000e+00, ...,
        -2.73132324e-02, -2.99072266e-02, -2.44750977e-02])


    """

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
