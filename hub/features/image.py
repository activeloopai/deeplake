from typing import Tuple

from hub.features.features import Tensor


class Image(Tensor):
    def __init__(self,
                 shape: Tuple[int, ...] = None,
                 dtype = None,
                 encoding_format: str = None,
                 channels = None,
    ):        
        self._set_dtype(dtype)
        super(Image, self).__init__(shape, dtype)
        self._set_channels(channels)

    @property
    def encoding_format(self):
        raise NotImplementedError()

    def _set_dtype(self, dtype):
        if dtype not in ('uint8', 'uint16'):
            raise ValueError(f'Not acceptable dtype for {self.__class__.__name__}')
        self.dtype = dtype

    def _set_channels(self, channels):        
        if channels and len(channels) != self.shape[-1]:
            raise ValueError(f'Channels are incompatible with image shape {self.shape}')
        self.channels = channels
        