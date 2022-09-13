import os
from hub.util.exceptions import InvalidPathException
import numpy as np
from .base import StructuredDataset
from hub.core.dataset import Dataset
from tqdm import tqdm  # type: ignore


class DataFrame(StructuredDataset):
    def __init__(self, source):
        """Convert a pandas dataframe to a Hub dataset.

        Args:
            source: Pandas dataframe object.

        Raises:
            Exception: If source is not a pandas dataframe object.
        """
        import pandas as pd  # type: ignore

        super().__init__(source)
        if not isinstance(self.source, pd.DataFrame):
            raise Exception("Source is not a pandas dataframe object.")

    def fill_dataset(self, ds: Dataset, use_progress_bar: bool = True) -> Dataset:
        """Fill dataset with data from the dataframe - one tensor per column

        Args:
            ds (Dataset) : A Hub dataset object.
            use_progress_bar (bool) : Defines if the method uses a progress bar. Defaults to True.

        Returns:
            A hub dataset.

        """
        keys = list(self.source.columns)
        skipped_keys: list = []
        iterator = tqdm(
            keys,
            desc="Ingesting... (%i keys skipped)" % (len(skipped_keys)),
            disable=not use_progress_bar,
        )
        with ds, iterator:
            for key in iterator:
                try:
                    dtype = self.source[key].dtype
                    if dtype == np.dtype("object"):
                        self.source[key].fillna("", inplace=True)
                        ds.create_tensor(key, dtype=str, htype="text")
                    else:
                        self.source[key].fillna(0, inplace=True)
                        ds.create_tensor(key)
                    ds[key].extend(self.source[key].values.tolist())
                except Exception as e:
                    skipped_keys.append(key)
                    iterator.set_description(
                        "Ingesting... (%i keys skipped)" % (len(skipped_keys))
                    )
                    continue

        return ds
