import numpy as np
from typing import Optional


class DeeplakeRandom(object):
    def __new__(cls):
        """Returns a :class:`~deeplake.random.DeeplakeRandom` object songlton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(DeeplakeRandom, cls).__new__(cls)
            cls.instance.internal_seed = None
            cls.instance.indra_api = None
        return cls.instance

    def seed(self, seed: Optional[int] = None):
        """Set random seed to the deeplake engines

        Args:
            seed (int, optional): integer seed to initialise the engines, used to control random behaviour and bring reproducability. Set number to initialise the engines to reset the seed set None. Defaults to None.

        Raises:
            TypeError: If the provided value is not expected one.

        """
        if seed is None or isinstance(seed, int):
            self.internal_seed = seed
            if self.indra_api is None:  # type: ignore
                from deeplake.enterprise.convert_to_libdeeplake import (
                    import_indra_api_silent,
                )

                self.indra_api = import_indra_api_silent()
            if self.indra_api is not None:
                self.indra_api.set_seed(self.internal_seed)
        else:
            raise TypeError(
                f"provided seed type `{type(seed)}` is incorrect seed must be an integer"
            )

    def get_seed(self) -> Optional[int]:
        """ Returns the seed which set to the deeplake to control the flows
        """
        return self.internal_seed
