from typing import Tuple

from hub.features.features import Tensor


class BBox(Tensor):
    def __init__(self, xmin: float, ymin: float, 
                 xmax: float, ymax: float):
        super(BBox, self).__init__(shape=(4,), dtype='float32')
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.validate()

    def validate(self):
        assert 0 < self.xmin < self.xmax 
        assert 0 < self.ymin < self.ymax

if __name__ == "__main__":
    box = BBox(0.1, 0.3, 0.5, 0.7)
    print(box.dtype)