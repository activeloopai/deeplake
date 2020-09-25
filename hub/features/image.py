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
        self._set_encoding_format(encoding_format)

    def _set_encoding_format(self, encoding_format):
        if encoding_format and encoding_format not in ('png', 'jpeg'):
            raise ValueError('Not supported encoding format')
        else:
            encoding_format = 'png'
        self.encoding_format = encoding_format

    def _set_dtype(self, dtype):
        if dtype not in ('uint8', 'uint16'):
            raise ValueError(f'Not supported dtype for {self.__class__.__name__}')
        self.dtype = dtype

    def _set_channels(self, channels):        
        if channels and len(channels) != self.shape[-1]:
            raise ValueError(f'Channels are incompatible with image shape {self.shape}')
        self.channels = channels
        
    def get_attribute_dict(self):
        """Return class attributes
        """
        return self.__dict__        


if __name__ == "__main__":
    image = Image((300, 400, 3), channels=['red', 'blue', 'green'], dtype='uint8')
    import pdb; pdb.set_trace()