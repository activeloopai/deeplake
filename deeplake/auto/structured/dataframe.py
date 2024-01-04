import numpy as np
from .base import StructuredDataset
from deeplake import Dataset
from deeplake import read, link
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from deeplake.util.dataset import sanitize_tensor_name

from collections import defaultdict
from typing import DefaultDict, List, Union, Optional, Dict
from deeplake.core.sample import Sample
from deeplake.core.linked_sample import LinkedSample
import pathlib


from deeplake.client.log import logger


class DataFrame(StructuredDataset):
    def __init__(self, source, column_params=None, creds=None, creds_key=None):
        """Convert a pandas dataframe to a Deep Lake dataset.

        Args:
            source: Pandas dataframe object.
            column_params: Optional setting for the tensors corresponding to the dataframe columns.
            creds: Optional credentials for accessing the source data.
            creds_key: Optional managed credentials key for accessing source data in linked tensors.


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

    def _initialize_params(self, column_params):
        column_params = column_params or {}
        for key in self.source.columns:
            current_params = column_params.get(key, None)
            params = current_params.copy() if current_params else None
            if params:
                if "name" not in params:
                    params["name"] = sanitize_tensor_name(key)
                    column_params[key] = params
            else:
                column_params[key] = {"name": sanitize_tensor_name(key)}
        self.column_params = column_params

    def _get_most_frequent_image_extension(self, fn_iterator: List[str]):
        # TODO: Make this generic and work for any htype that requires compression

        if len(fn_iterator) == 0:
            raise IngestionError(
                "Cannot determine the most frequent image compression because no valid image files were provided."
            )

        supported_image_extensions = tuple(
            "." + fmt for fmt in HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        image_extensions: DefaultDict[str, int] = defaultdict(int)
        for file in fn_iterator:
            if file.lower().endswith(supported_image_extensions):
                ext = file.rsplit(".", 1)[1]
                image_extensions[ext] += 1
            else:
                raise IngestionError(f"The following file is not supported: {file}")

        most_frequent_image_extension = max(
            image_extensions, key=lambda k: image_extensions[k], default=None
        )
        return most_frequent_image_extension

    def _parse_tensor_params(self, key: str, inspect_limit: int = 1000):
        """Parse the tensor parameters for a column. Required parameters that are not specified will be inferred by inspecting up to 'inspect_limit' rows in the data."""

        tensor_params: Dict = self.column_params[key]

        dtype = self.source[key].dtype

        if (
            "htype" not in tensor_params
        ):  # Auto-set some typing parameters if htype is not specified
            if dtype == np.dtype("object"):
                types = [
                    type(v)
                    for v in self.source[key][0:inspect_limit].values
                    if v is not None
                ]  # Can be length 0 if all data is None

                if len(set(types)) > 1:
                    raise IngestionError(
                        f"Dataframe has different data types inside '{key}' column. Please make sure all data is given column is compatible with a single Deep Lake htype, or try specifying the htype manually."
                    )

                if len(types) > 0 and types[0] == str:
                    tensor_params.update(
                        htype="text"
                    )  # Use "text" htype for text data when the htype is not specified tensor_params
            else:
                tensor_params.update(
                    dtype=dtype,
                    create_shape_tensor=tensor_params.get("create_shape_tensor", False),
                )

        # TODO: Make this more robust so it works for all htypes where sample_compression is required and should be inferred from the data itself
        if (
            "image" in tensor_params.get("htype", "")
            and "sample_compression" not in tensor_params
            and "chunk_compression" not in tensor_params
        ):
            tensor_params.update(
                sample_compression=self._get_most_frequent_image_extension(
                    self.source[key][self.source[key].notnull()].values
                )
            )

        return tensor_params

    def _get_extend_values(self, tensor_params: dict, key: str):  # type: ignore
        """Method creates a list of values to be extended to the tensor, based on the tensor parameters and the data in the dataframe column"""
        import pandas as pd  # type: ignore

        column_data = self.source[key]
        column_data = column_data.where(pd.notnull(column_data), None).values.tolist()

        extend_values: List[Optional[Union[Sample, LinkedSample, np.ndarray]]]

        if "htype" in tensor_params and "link[" in tensor_params["htype"]:
            extend_values = [
                link(value, creds_key=self.creds_key) if value is not None else None
                for value in column_data
            ]
        elif "htype" in tensor_params and "image" in tensor_params["htype"]:
            extend_values = [
                read(value, creds=self.creds) if value is not None else None
                for value in column_data
            ]
        else:
            extend_values = column_data

        return extend_values

    def fill_dataset(self, ds: Dataset, progressbar: bool = True) -> Dataset:
        """Fill dataset with data from the dataframe - one tensor per column

        Args:
            ds (Dataset) : A Deep Lake dataset object.
            progressbar (bool) : Defines if the method uses a progress bar. Defaults to True.

        Returns:
            A Deep Lake dataset.

        """

        keys = list(self.source.columns)

        with ds:
            if self.creds_key is not None and self.creds_key not in ds.get_creds_keys():
                ds.add_creds_key(self.creds_key, managed=True)
            for key in keys:
                if progressbar:
                    logger.info(f"Ingesting column '{key}'")

                tensor_params = self._parse_tensor_params(key)

                if tensor_params["name"] not in ds.tensors:
                    ds.create_tensor(**tensor_params)

                ds[tensor_params["name"]].extend(
                    self._get_extend_values(tensor_params, key),
                    progressbar=progressbar,
                )

        return ds
