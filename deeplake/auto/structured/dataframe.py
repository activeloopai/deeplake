import numpy as np
from .base import StructuredDataset
from deeplake import Dataset
from deeplake import read, link
from tqdm import tqdm  # type: ignore
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from collections import defaultdict
from typing import DefaultDict
import pathlib


from deeplake.client.log import logger


class DataFrame(StructuredDataset):
    def __init__(self, source, column_params=None, creds=None, creds_key=None):
        """Convert a pandas dataframe to a Deep Lake dataset.

        Args:
            source: Pandas dataframe object.
            column_params: Optional setting for the tensors corresponding to the dataframe columns
            creds: Optional credentials for accessing the source data

        Raises:
            Exception: If source is not a pandas dataframe object.
        """
        import pandas as pd  # type: ignore

        super().__init__(source)
        if not isinstance(self.source, pd.DataFrame):
            raise Exception("Source is not a pandas dataframe object.")

        self.creds = creds
        self.creds_key = creds_key
        self._initialize_params(column_params)

    def _sanitize_tensor(self, input: str):
        """Sanitize a string to be a valid tensor name

        Args:
            input: A string that will be sanitized
        """

        invalid_chars = ["?", "!", "/", "\\", "#", "'", '"']
        new_char = "_"
        for char in invalid_chars:
            input = input.replace(char, new_char)

        return input

    def _initialize_params(self, column_params):
        column_params = column_params if column_params is not None else {}

        keys = list(self.source.columns)

        for key in keys:
            if key in column_params.keys():
                column_params[key]["name"] = column_params[key].get(
                    "name", self._sanitize_tensor(key)
                )  # Get the specified name, and if it doesn't exist, use the sanitized column name
            else:
                column_params[key] = {
                    "name": self._sanitize_tensor(key)
                }  # If column parameters are not specified for a column, use the sanitized column name

        self.column_params = column_params

    def _get_most_frequent_image_extension(self, fn_iterator):

        # TODO: Make this generic and work for any htype that requires compression

        supported_image_extensions = tuple(
            HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        invalid_files = []
        image_extensions: DefaultDict[str, int] = defaultdict(int)

        for file in fn_iterator:
            if file.endswith(supported_image_extensions):
                ext = pathlib.Path(file).suffix[
                    1:
                ]  # Get extension without the . symbol
                image_extensions[ext] += 1
            else:
                invalid_files.append(file)

        if len(invalid_files) > 0:
            logger.warning(
                f"Encountered {len(invalid_files)} unsupported files the data folders and annotation folders (if specified)."
            )

        most_frequent_image_extension = max(
            image_extensions, key=lambda k: image_extensions[k], default=None
        )

        return most_frequent_image_extension

    def _parse_tensor_params(self, key, inspect_limit = 1000):
        """Parse the tensor parameters for a column. Required parameters that are not specified will be inferred by inspecting up to 'inspect_limit' rows in the data."""

        tensor_params = {}

        tensor_params = self.column_params[key]

        dtype = self.source[key].dtype
        if (
            "htype" not in tensor_params.keys()
        ):  # Auto-set some typing parameters if htype is not specified
            
            if dtype == np.dtype("object"):

                types = [type(v) for v in self.source[key][0:inspect_limit].values]
                
                if len(set(types))!=1:
                    raise IngestionError("Dataframe has different data types inside '{}' column. Please make sure all data is given column is compatible with a single Deep Lake htype, or try specifying the htype manually.".format(key))

                if types[0] == str:
                    tensor_params.update(
                        htype="text"
                    )  # Use "text" htype for text data when the htype is not specified tensor_params
            else:
                tensor_params.update(
                    dtype=dtype,
                    create_shape_tensor=tensor_params.get("create_shape_tensor", False),
                )  # htype will be auto-inferred for numeric data unless the htype is specified in tensor_params

        # TODO: Make this more robust so it works for all htypes where sample_compression is required and should be inferred from the data itself
        if (
            "image" in tensor_params.get("htype", "")
            and "sample_compression" not in tensor_params.keys()
            and "chunk_compression" not in tensor_params.keys()
        ):
            tensor_params.update(
                sample_compression=self._get_most_frequent_image_extension(
                    self.source[key].values
                )
            )

        return tensor_params

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
            if self.creds_key is not None and self.creds_key not in ds.get_creds_keys():
                ds.add_creds_key(self.creds_key, managed=True)
            for key in iterator:
                if progressbar:
                    logger.info(f"\column={key}, dtype={self.source[key].dtype}")
                # try:

                tensor_params = self._parse_tensor_params(key)

                if tensor_params["name"] not in ds.tensors:
                    ds.create_tensor(**tensor_params)

                if (
                    "htype" in tensor_params.keys()
                    and "link[" in tensor_params["htype"]
                ):
                    extend_values = [
                        link(value, creds_key=self.creds_key)
                        for value in self.source[key].values
                    ]
                elif (
                    "htype" in tensor_params.keys()
                    and "image" in tensor_params["htype"]
                ):

                    extend_values = [
                        read(value, creds=self.creds)
                        for value in self.source[key].values
                    ]
                else:
                    extend_values = self.source[key].values

                ds[tensor_params["name"]].extend(extend_values, progressbar=progressbar)
                # except Exception as e:
                #     print("Error: {}".format(str(e)))
                #     skipped_keys.append(key)
                #     iterator.set_description(
                #         "Ingesting... (%i columns skipped)" % (len(skipped_keys))
                #     )
                #     continue
        return ds
