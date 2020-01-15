import numpy
from .image_codec import ImageCodec
from PIL import Image
import pickle
import io

class JpegCodec(ImageCodec):
    def __init__(self):
        super().__init__('jpeg')