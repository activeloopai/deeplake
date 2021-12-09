import os
from hub.util.exceptions import InvalidPathException
import numpy as np
import pandas as pd  # type: ignore
from .base import StructuredDataset
from hub.core.dataset import Dataset
from tqdm import tqdm  # type: ignore


class CSVFile(StructuredDataset):
    def __init__(self, source: str):
        """Convert a CSV file to a Hub dataset.

        Args:
            source (str): The full path to the dataset.
                Can be a Hub cloud path of the form hub://username/datasetname. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                Can be a s3 path of the form s3://bucketname/path/to/dataset. Credentials are required in either the environment or passed to the creds argument.
                Can be a local file system path of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
                Can be a memory path of the form mem://path/to/dataset which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.

        Raises:
            InvalidPathException: If source is invalid.
        """
        super().__init__(source)
        if not os.path.isfile(self.source):
            raise InvalidPathException(f"{self.source} is not a valid CSV file.")

    def fill_dataset(self, ds: Dataset, use_progress_bar: bool = True) -> Dataset:
        """Fill dataset with data from csv file - one tensor per column

        Args:
            ds (Dataset) : A Hub dataset object.
            use_progress_bar (bool) : Defines if the method uses a progress bar. Defaults to True.

        Returns:
            A hub dataset.

        """
        self.df = pd.read_csv(self.source, quotechar='"', skipinitialspace=True)
        keys = list(self.df.columns)
        skipped_keys: list = []
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

        return ds
