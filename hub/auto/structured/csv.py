import numpy as np
import pandas as pd
from .base import StructuredDataset
from hub.core.dataset import Dataset
from tqdm import tqdm
import os
import hub


class CSVFile(StructuredDataset):
    def __init__(self, source: str):
        super().__init__(source)
        self.df = None

    def fill_dataset(self, ds: Dataset, use_progress_bar: bool):
        self.df = pd.read_csv(self.source, quotechar='"', skipinitialspace=True)
        keys = list(self.df.columns)
        skipped_keys = []
        iterator = tqdm(
            keys,
            desc='Ingesting "%s" (%i keys skipped)'
            % (self.source.name, len(skipped_keys)),
            disable=not use_progress_bar,
        )
        with ds, iterator:
            for key in iterator:
                try:
                    dtype = self.df[key].dtype
                    if dtype == np.dtype("object"):
                        self.df[key].fillna("", inplace=True)
                        ds.create_tensor(key, dtype=str, htype="text")
                    else:
                        self.df[key].fillna(0, inplace=True)
                        ds.create_tensor(key)
                    ds[key].extend(self.df[key].values.tolist())
                except Exception as e:
                    skipped_keys.append(key)
                    iterator.set_description(
                        'Ingesting "%s" (%i files skipped)'
                        % (self.source.name, len(skipped_keys))
                    )
                    continue
