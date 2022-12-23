from deeplake.core.tensor import Tensor
import numpy as np
from io import BytesIO

class ParquetTensor(Tensor):
    def _parquet_store_tensor(self):
        return self.dataset._parquet_store

    def _parquet_store_lengths_tensor(self):
        return self.dataset._parquet_store_lengths

    def __len__(self):
        return sum(self._parquet_store_lengths_tensor().numpy())

    def _get_store_index(self, idx: int):
        lengths = self._parquet_store_lengths_tensor().numpy()
        st_idx = np.searchsorted(lengths, idx, side="right")
        offset = idx - lengths[:st_idx].sum()
        return st_idx, offset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = None
        self._cache_index = None

    def _get_value(self, idx: int):
        import pandas as pd
        idx, offset = self._get_store_index(idx)
        if self._cache_index != idx:
            self._cache = pd.read_parquet(BytesIO(self._parquet_store_tensor()[idx].numpy().tobytes()), columns=[self.key])
            self._cache_index = idx
        return self._cache[self.key][offset]

    def numpy(self, aslist=False):
        vals = [self._get_value(idx) for idx in range(self.index.values[0].indices(len(self)))]
        if aslist:
            return vals
        return np.array(vals)
