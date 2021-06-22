from hub.util.adjacency import calculate_adjacent_runs
from hub.constants import SHAPE_META_FILENAME
from typing import Sequence, Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


SHAPE_META_DTYPE = np.uint64


class ShapeMetaEncoder:
    def __init__(self, storage: StorageProvider):
        self.storage = storage
        self.key = SHAPE_META_FILENAME
        self._encoded_shapes = None
        # TODO: `num_samples`: bottom right corner of `self._encoded_shapes` will be = the length of the tensor MINUS 1***** (no longer need to store in tensor_meta.length)
        self.load_shapes()

    def load_shapes(self):
        if self.key in self.storage:
            # TODO: read/parse from storage
            raise NotImplementedError()

    def __getitem__(self, sample_index: int) -> np.ndarray:
        shape_index = np.searchsorted(self._encoded_shapes[:, -1], sample_index)
        return self._encoded_shapes[shape_index, :-1]

    @property
    def num_samples(self) -> int:
        if self._encoded_shapes is None:
            return 0
        return int(self._encoded_shapes[-1, -1] + 1)

    def add_shapes(
        self,
        shapes: Sequence[Tuple[int]],
    ):
        last_shape_entry = None
        if self._encoded_shapes is not None:
            last_shape_entry = self._encoded_shapes[-1, :-1]

        shape_entries = []
        for i, shape in enumerate(shapes):
            if last_shape_entry is None or last_shape_entry[:-1] != list(shape):
                shape_entries.append(list(shape) + [i + self.num_samples])
                last_shape_entry = shape_entries[-1]
            else:
                shape_entries[-1][-1] = i + self.num_samples

        shape_entries = np.array(shape_entries, dtype=SHAPE_META_DTYPE)
        if self._encoded_shapes is None:
            self._encoded_shapes = shape_entries

        else:
            self._encoded_shapes = np.concatenate(
                [self._encoded_shapes, shape_entries], axis=0
            )
