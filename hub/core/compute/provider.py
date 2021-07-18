from abc import ABC, abstractmethod


# TODO, will probably also need Queue to get tqdm working properly
class ComputeProvider(ABC):
    """An abstract base class for implementing a compute provider."""

    def __init__(self, workers):
        self.workers = workers

    @abstractmethod
    def map(self, func, iterable):
        """Apply 'func' to each element in 'iterable', collecting the results
        in a list that is returned.
        """
