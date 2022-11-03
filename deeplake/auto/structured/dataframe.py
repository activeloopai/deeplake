import numpy as np
from .base import StructuredDataset
from deeplake import Dataset
from tqdm import tqdm  # type: ignore


class DataFrame(StructuredDataset):
    def __init__(self, source):
        """Convert a pandas dataframe to a Deep Lake dataset.

        Args:
            source: Pandas dataframe object.

        Raises:
            Exception: If source is not a pandas dataframe object.
        """
        import pandas as pd  # type: ignore

        super().__init__(source)
        if not isinstance(self.source, pd.DataFrame):
            raise Exception("Source is not a pandas dataframe object.")

    def fill_dataset(self, ds: Dataset, progressbar: bool = True) -> Dataset:
        """Fill dataset with data from the dataframe - one tensor per column

        Args:
            ds (Dataset) : A Deep Lake dataset object.
            progressbar (bool) : Defines if the method uses a progress bar. Defaults to True.

        Returns:
            A Deep Lake dataset.

        """
        keys = list(self.source.columns)
        skipped_keys: list = []
        iterator = tqdm(
            keys,
            desc="Ingesting... (%i keys skipped)" % (len(skipped_keys)),
            disable=not progressbar,
        )
        with ds, iterator:
            for key in iterator:
                if progressbar:
                    print(f"\nkey={key}, dtype={self.source[key].dtype}")
                try:
                    dtype = self.source[key].dtype
                    if dtype == np.dtype("object"):
                        if key not in ds.tensors:
                            ds.create_tensor(key, htype="text")
                    else:
                        if key not in ds.tensors:
                            ds.create_tensor(
                                key, dtype=dtype, create_shape_tensor=False
                            )
                    ds[key].extend(self.source[key].values, progressbar=progressbar)
                except Exception as e:
                    skipped_keys.append(key)
                    iterator.set_description(
                        "Ingesting... (%i keys skipped)" % (len(skipped_keys))
                    )
                    continue
        return ds
