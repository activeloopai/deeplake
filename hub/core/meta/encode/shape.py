from typing import Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


SHAPE_ENCODING_DTYPE = np.uint64


class ShapeEncoder:
    def __init__(self, storage: StorageProvider):
        self.storage = storage
        self._encoded = None
        self.load_shapes()

    def load_shapes(self):
        if self.key in self.storage:
            # TODO: read/parse from storage
            raise NotImplementedError()

    def __getitem__(self, sample_index: int) -> np.ndarray:
        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty shape encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded[:, -1], sample_index)
        return tuple(self._encoded[idx, :-1])

    @property
    def num_samples(self) -> int:
        if self._encoded is None:
            return 0
        return int(self._encoded[-1, -1] + 1)

    def add_shape(
        self,
        shape: Tuple[int],
        count: int,
    ):
        if count <= 0:
            raise ValueError(f"Shape `count` should be > 0. Got {count}.")

        if self.num_samples != 0:
            last_shape = self[-1]

            if len(shape) != len(last_shape):
                raise ValueError(
                    f"All sample shapes in a tensor must have the same len(shape). Expected: {len(last_shape)} got: {len(shape)}."
                )

            if shape == last_shape:
                # increment last shape's index by `count`
                self._encoded[-1, -1] += count

            else:
                last_shape_index = self._encoded[-1, -1]
                shape_entry = np.array(
                    [[*shape, last_shape_index + count]], dtype=SHAPE_ENCODING_DTYPE
                )

                self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)

        else:
            self._encoded = np.array([[*shape, count - 1]], dtype=SHAPE_ENCODING_DTYPE)
