from hub.core._compression.base_compressor import BaseCompressor
from PIL import Image  # type: ignore


class PNG(BaseCompressor):
    def verify(self):
        self.image.verify()
        return Image._conv_type_shape(self.image)
