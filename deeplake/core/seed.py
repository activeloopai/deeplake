import numpy as np
from typing import Optional

class DeeplakeRandom(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DeeplakeRandom, cls).__new__(cls)
            cls.instance.internal_seed = None
            cls.instance.indra_api = None
        return cls.instance


    def seed(self, seed: Optional[int] = None):
        if isinstance(seed, Optional[int]):
            self.internal_seed = seed
            if self.indra_api is None:
                from deeplake.enterprise.convert_to_libdeeplake import import_indra_api_silent
                self.indra_api = import_indra_api_silent()
            if self.indra_api is not None:
                self.indra_api.set_seed(self.internal_seed)
        else:
            raise TypeError(f"provided seed type `{type(seed)}` is increect seed must be an integer")

    def get_seed(self) -> Optional[int]:
        return self.internal_seed