from hub.features import featurify, FeatureConnector, FlatTensor
from hub.store.storage_tensor import StorageTensor
# from hub.api.dataset import slice_split_tuple, slice_extract_info
class TensorView:
    def __init__(
        self,
        token=None,
        mode: str = None,
        dtype=None,
        path=None,
        slice_=None,
        _tensors=None,
    ):
        assert dtype is not None
        # assert num_samples is not None
        assert mode is not None
        assert path is not None
        assert _tensors is not None

        self.token = token
        self.mode = mode
        # self.num_samples = num_samples
        self.dtype: FeatureConnector = dtype
        # self._flat_tensors: Tuple[FlatTensor] = tuple(self.dtype._flatten())
        self._tensors = _tensors
        self.path=path
        # self.offset = offset
        self.slice_=slice_

    
    def numpy(self):
        if self.slice_ is None:
            return self._tensors[self.path][:]
        return self._tensors[self.path][self.slice_]