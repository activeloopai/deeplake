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

from hub.core.compression.compressions import AUDIO_COMPRESSIONS
from hub.core.compression.compressor import BaseCompressor


class AudioCompressor(BaseCompressor):
    supported_compressions = AUDIO_COMPRESSIONS

    def __init__(self, compression):
        super(AudioCompressor, self).__init__(compression=compression)

    def compress(self, data: np.ndarray) -> bytes:
        raise NotImplementedError(
            "In order to store audio data, you should use `hub.read(path_to_file)`. Compressing raw data is not yet supported."
        )

    def decompress(self, compressed: bytes) -> np.ndarray:
        if isinstance(compressed, memoryview):
            if (
                isinstance(compressed.obj, bytes)
                and compressed.strides == (1,)
                and compressed.shape == (len(compressed.obj),)
            ):
                compressed = compressed.obj
            else:
                compressed = bytes(compressed)
        raw_audio = decompressor(compressed)
        return np.frombuffer(raw_audio.samples, dtype="<f4").reshape(
            raw_audio.num_frames, raw_audio.nchannels
        )
