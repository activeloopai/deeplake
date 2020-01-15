import numpy
from hub.codec.codec import Codec
from PIL import Image
import pickle
import io

class ImageCodec(Codec):
    def __init__(self, format: str):
        self._format = format
        super().__init__()

    def encode(self, array: numpy.ndarray) -> bytes:
        info = {}
        info['shape'] = array.shape
        info['dtype'] = array.dtype

        assert array.dtype == 'uint8'
        assert len(array.shape) >= 3
        assert array.shape[-1] == 3

        if len(array.shape) == 3:
            bs = io.BytesIO()
            img = Image.fromarray(array)
            img.save(bs, format=self._format)
            info['image'] = bs.getvalue()
        else:
            images = []
            for i, index in enumerate(numpy.ndindex(array.shape[:-3])):
                bs = io.BytesIO()
                img = Image.fromarray(array[index])
                img.save(bs, format=self._format)
                images.append(bs.getvalue())
        info['images'] = images         
        return pickle.dumps(info)
    
    def decode(self, content: bytes) -> numpy.ndarray:
        info = pickle.loads(content)
        if 'image' in info:
            img = Image.open(io.BytesIO(info['image']))
            img.reshape(info['shape'])
            return img
        else:
            array = numpy.zeros(info['shape'], info['dtype'])
            images = info['images']
            for i, index in enumerate(numpy.ndindex(info['shape'][:-3])):
                img = Image.open(io.BytesIO(images[i]))
                arr = numpy.asarray(img)
                array[index] = arr
            return array

    