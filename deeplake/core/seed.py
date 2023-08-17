import numpy as np
from typing import Optional

class DeeplakeRandom(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DeeplakeRandom, cls).__new__(cls)
            cls.instance.internal_seed = None
        return cls.instance

    def seed(self, seed: Optional[int] = None):
        if isinstance(seed, Optional[int]):
            self.internal_seed = seed
        else:
            raise TypeError(f"provided seed type `{type(seed)}` is increect seed must be an integer")

    def get_seed(self) -> Optional[int]:
        return self.internal_seed

    