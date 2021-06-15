from hub_v1.api.dataset import Dataset, DatasetView, TensorView
import numpy as np


# a list of Datasets or DatasetViews or Tensorviews that supports compute operation
class ComputeList:
    # Doesn't support further get item operations currently
    def __init__(self, ls):
        self.ls = ls

    def compute(self):
        results = [
            item.compute()
            if isinstance(item, (Dataset, DatasetView, TensorView))
            else item
            for item in self.ls
        ]
        return np.concatenate(results)

    def numpy(self):
        return self.compute()
