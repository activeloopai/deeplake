# FIXME NOT WORKING YET, NEED SOME CHANGES
# DONT INCLUDE INTO __init__.py YET

from typing import Tuple

from hub.features.features import Tensor


class Audio(Tensor):
    def __init__(
        self,
        shape: Tuple[int, ...] = (None,),
        dtype="int64",
        file_format=None,
        sample_rate: int = None,
        max_shape: Tuple[int, ...] = None,
        chunks=True,
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
            `info.features['audio'].sample_rate`. This value isn't used neither in
            encoding nor decoding.
        

        Raises
        ----------
        ValueError: If the shape is invalid
        """
        self.file_format = file_format
        if len(shape) != 1:
            raise TypeError(
                "Audio feature currently only supports 1-D values, got %s." % shape
            )
        # self._shape = shape
        self.sample_rate = sample_rate
        super(Audio, self).__init__(
            shape=shape, dtype=dtype, max_shape=max_shape, chunks=chunks
        )

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
