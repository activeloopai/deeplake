from typing import Tuple
import numpy 

class HubArray():
    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def chunk(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def dtype(self) -> str:
        raise NotImplementedError()

    @property
    def compress(self) -> str:
        raise NotImplementedError()

    @property
    def compresslevel(self) -> float:
        raise NotImplementedError()

    def __getitem__(self, slices: Tuple[slice]) -> numpy.ndarray:
        raise NotImplementedError()

    def __setitem__(self, slices: Tuple[slice], content: numpy.ndarray):
        raise NotImplementedError()

