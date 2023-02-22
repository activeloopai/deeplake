import numpy as np
from .base import StructuredDataset
from deeplake import Dataset
from deeplake import read, link
from tqdm import tqdm  # type: ignore


class DataFrame(StructuredDataset):
    def __init__(self, source, column_params=None):
        """Convert a pandas dataframe to a Deep Lake dataset.

        Args:
            source: Pandas dataframe object.
            column_params: Optional setting for the tensors corresponding to the dataframe columns

        Raises:
            Exception: If source is not a pandas dataframe object.
        """
        import pandas as pd  # type: ignore

        super().__init__(source)
        if not isinstance(self.source, pd.DataFrame):
            raise Exception("Source is not a pandas dataframe object.")

        self.column_params = column_params if column_params is not None else {}

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
            desc="Ingesting... (%i columns skipped)" % (len(skipped_keys)),
            disable=not progressbar,
        )
        with ds, iterator:
            for key in iterator:
                if progressbar:
                    print(f"\column={key}, dtype={self.source[key].dtype}")
                try:
                    tensor_params = self.column_params.get(
                        key, {"name": key}
                    )  # Pull columns settings if specified. If not, default the tensor name to the column name
                    if "name" not in tensor_params.keys():
                        tensor_params.update(
                            name=key
                        )  # If column settings were specified but the name was omitted, the key should be used as the tensor name

                    dtype = self.source[key].dtype
                    if (
                        dtype == np.dtype("object")
                        and "htype" not in tensor_params.keys()
                    ):
                        tensor_params.update(htype="text") # Use "text" htype for text data, and if the htype is not specified
                    else:
                        tensor_params.update(dtype=dtype, create_shape_tensor=tensor_params.get('create_shape_tensor', False)) # htype will be auto-inferred for numeric data unless the htype is specified in tensor_params

                    if key not in ds.tensors:
                        ds.create_tensor(**tensor_params)

                    if "htype" in tensor_params.keys() and "link" in tensor_params["htype"]:
                        creds_key = tensor_params.get("creds_key", None)
                        extend_values = [
                            link(value, creds_key=creds_key)
                            for value in self.source[key].values
                        ]
                    elif "htype" in tensor_params.keys() and "image" in tensor_params["htype"]:
                        extend_values = [read(value) for value in self.source[key].values]
                    else:
                        extend_values = self.source[key].values

                    ds[tensor_params["name"]].extend(extend_values, progressbar=progressbar)
                except Exception as e:
                    print(str(e))
                    skipped_keys.append(key)
                    iterator.set_description(
                        "Ingesting... (%i columns skipped)" % (len(skipped_keys))
                    )
                    continue
        return ds
