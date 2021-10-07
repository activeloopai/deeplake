from typing import Tuple
from hub.compression import AUDIO_COMPRESSIONS
from hub.core.compression import BaseCompressor
import numpy as np

from miniaudio import (  # type: ignore
    mp3_read_file_f32,
    mp3_read_f32,
    mp3_get_file_info,
    mp3_get_info,
    flac_read_file_f32,
    flac_read_f32,
    flac_get_file_info,
    flac_get_info,
    wav_read_file_f32,
    wav_read_f32,
    wav_get_file_info,
    wav_get_info,
)
from numpy.core.fromnumeric import compress  # type: ignore


_AUDIO_DTYPE = "<f4"


def _validate_audio_compression(compression: str):
    if compression not in AUDIO_COMPRESSIONS:
        raise ValueError(
            f"{compression} is not an audio compression. Audio compressions: {AUDIO_COMPRESSIONS}"
        )

class Audio(BaseCompressor):
    def __init__(self, item, compression: str):
        _validate_audio_compression(compression)
        super().__init__(item)

        self.compression = compression

    def read_shape_and_dtype(self) -> Tuple[Tuple[int, ...], str]:
        if self.file is not None:
            info_getter = globals()[f"{self.compression}_get_file_info"]
            info = info_getter(self.file)
        else:
            info_getter = globals()[f"{self.compression}_get_info"]
            info = info_getter(self.buffer)

        return (info.num_frames, info.nchannels), _AUDIO_DTYPE

    def decompress(self):
        if self.file is not None:
            decompressor = globals()[f"{self.compression}_read_file_f32"]
            raw_audio = decompressor(self.file)
        else:
            decompressor = globals()[f"{self.compression}_read_f32"]
            raw_audio = decompressor(self.buffer)

        return np.frombuffer(raw_audio.samples, dtype=_AUDIO_DTYPE).reshape(
            raw_audio.num_frames, raw_audio.nchannels
        )
