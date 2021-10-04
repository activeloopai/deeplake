from typing import Tuple, Union
from hub.core._compression import BaseCompressor
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

        raise NotImplementedError

        if self.file is not None:
            raw_audio = mp3_read_file_f32(self.file)
        else:
            raw_audio = mp3_read_f32(self.buffer)
        
        return np.frombuffer(raw_audio.samples, dtype=_MP3_DTYPE).reshape(
            raw_audio.num_frames, raw_audio.nchannels
        )



def decompress_mp3(file: Union[bytes, memoryview, str]) -> np.ndarray:
    # TODO: move into class

    decompressor = mp3_read_file_f32 if isinstance(file, str) else mp3_read_f32
    if isinstance(file, memoryview):
        if (
            isinstance(file.obj, bytes)
            and file.strides == (1,)
            and file.shape == (len(file.obj),)
        ):
            file = file.obj
        else:
            file = bytes(file)
    raw_audio = decompressor(file)
    return np.frombuffer(raw_audio.samples, dtype="<f4").reshape(
        raw_audio.num_frames, raw_audio.nchannels
    )