import numpy as np
from typing import Optional


class DeeplakeRandom(object):
    def __new__(cls):
        """Returns a :class:`~deeplake.core.seed.DeeplakeRandom` object singleton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(DeeplakeRandom, cls).__new__(cls)
            cls.instance.internal_seed = None
            cls.instance.indra_api = None
        return cls.instance

    def seed(self, seed: Optional[int] = None):
        """Set random seed to the deeplake engines

        Args:
            seed (int, optional): integer seed to initialize the engines, used to control random behavior and bring reproducibility. Set number to initialize the engines, to reset the seed set None. Defaults to None.

        Raises:
            TypeError: If the provided value type is not an expected one.

        Background
        ----------

        To train and operate with the deeplake features and keep reproducibility between all the engines
        See the examples below on how to operate with the random engines.

        Deeplake does not provide in-house random number generator but to control the reproducibility
        and keep track on flow stages can be used :meth:`deeplake.random.seed` to seed the RNG for all the engines (both
        enterprise and non enterprise)::

            import deeplake
            deeplake.random.seed(0)

        Random number generators in other libraries
        -------------------------------------------
        If you or any of the libraries you are using rely on NumPy, you can seed the global
        NumPy RNG with::

            import numpy as np
            np.random.seed(0)

            import random
            random.seed(0)

        Those will impact to all the places where deeplake uses those libraries but with use of use :meth:`deeplake.random.seed` we are nor changing any library
        global state instead just initializing singleton instances in the random engines


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
        """Returns the seed which set to the deeplake to control the flows"""
        return self.internal_seed
