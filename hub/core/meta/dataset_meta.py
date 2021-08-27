from typing import Any, Dict
from hub.core.meta.meta import Meta


class DatasetMeta(Meta):
    def __init__(self):

        """Dataset metadata is responsible for keeping track of global sample metadata within a dataset.

        Note:
            tensors: A list of tensors associated with this dataset.
            hidden_tensors: Hidden tensors aren't accessible and can't be appended upon. An example of
                            a hidden tensor is the "_hashes" tensor generated.

        """

        self.tensors = []
        self.hidden_tensors = []

        super().__init__()

    @property
    def nbytes(self):
        # TODO: can optimize this
        return len(self.tobytes())

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()
        d["tensors"] = self.tensors
        d["hidden_tensors"] = self.hidden_tensors
        return d
