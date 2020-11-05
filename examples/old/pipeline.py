import numpy as np
from hub import Transform, dataset


class Crop(Transform):
    def forward(self, input):
        return {"image": input[0:1, :256, :256]}

    def meta(self):
        return {"image": {"shape": (1, 256, 256), "dtype": "uint8"}}


class Flip(Transform):
    def forward(self, input):
        img = np.expand_dims(input["image"], axis=0)
        img = np.flip(img, axis=(1, 2))
        return {"image": img}

    def meta(self):
        return {"image": {"shape": (1, 256, 256), "dtype": "uint8"}}


images = [np.ones((1, 512, 512), dtype="uint8") for i in range(20)]
ds = dataset.generate(Crop(), images)
ds = dataset.generate(Flip(), ds)
ds.store("/tmp/cropflip")
