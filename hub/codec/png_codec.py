from .image_codec import ImageCodec

class PngCodec(ImageCodec):
    def __init__(self):
        super().__init__('png')
    