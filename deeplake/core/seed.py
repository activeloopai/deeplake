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
            seed (int, optional): Integer seed for initializing the computational engines, used for bringing reproducibility to random operations.
                Set to ``None`` to reset the seed. Defaults to ``None``.

        Raises:
            TypeError: If the provided value type is not supported.

        Background
        ----------

        Specify a seed to train models and run randomized Deep Lake operations reproducibly. Features affected are:

            - Dataloader shuffling
            - Sampling and random operations in Tensor Query Language (TQL)
            - :meth:`Dataset.random_split <deeplake.core.dataset.Dataset.random_split>`


        The random seed can be specified using ``deeplake.random.seed``:

            >>> import deeplake
            >>> deeplake.random.seed(0)

        Random number generators in other libraries
        -------------------------------------------

        The Deep Lake random seed does not affect random number generators in other libraries such as ``numpy``.

        However, seeds in other libraries will affect code where Deep Lake uses those libraries, but it will not impact
        the methods above where Deep Lake uses its internal seed.

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
