from typing import Tuple, Union
from hub.core.compression import BaseCompressor
import numpy as np

from miniaudio import mp3_read_file_f32, mp3_read_f32, mp3_get_file_info, mp3_get_info  # type: ignore


_MP3_DTYPE = "<f4"


class MP3(BaseCompressor):
    def read_shape_and_dtype(self) -> Tuple[Tuple[int, ...], str]:
        if self.file is not None:
            info = mp3_get_file_info(self.file)
        else:
            info = mp3_get_info(self.buffer)

        return (info.num_frames, info.nchannels), _MP3_DTYPE

    def decompress(self):
        # TODO: put in base class

        if self.file is not None:
            raw_audio = mp3_read_file_f32(self.file)
        else:
            raw_audio = mp3_read_f32(self.buffer)

        return np.frombuffer(raw_audio.samples, dtype=_MP3_DTYPE).reshape(
            raw_audio.num_frames, raw_audio.nchannels
        )
