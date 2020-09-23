from typing import Tuple

import hub.features.flat_tensor as flat_tensor

FlatTensor = flat_tensor.FlatTensor


class FeatureConnector:
    def _flatten(self) -> Tuple[FlatTensor]:
        raise NotImplementedError()

    # @property
    # def shape(self):
    #     raise NotImplementedError()

    # @property
    # def dtype(self):
    #     raise NotImplementedError()
