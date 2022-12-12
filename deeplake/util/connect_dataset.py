from typing import Tuple, Optional

import deeplake

from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.log import logger
from deeplake.util.bugout_reporter import feature_report_path
from deeplake.util.path import is_hub_cloud_path, get_path_type, get_org_id_and_ds_name
from deeplake.util.exceptions import InvalidDestinationPathError, InvalidSourcePathError
from deeplake.util.logging import log_visualizer_link


def log_dataset_connection_success(ds_path: str):
    logger.info("Dataset connected successfully.")
    log_visualizer_link(ds_path)


def is_path_connectable(path: str) -> bool:
    return get_path_type(path) in ("s3", "gcs")


def connect_dataset_entry(
    src_path: str,
    creds_key: str,
    dest_path: Optional[str] = None,
    org_id: Optional[str] = None,
    ds_name: Optional[str] = None,
    token: Optional[str] = None,
    verbose: bool = True,
) -> str:
    dataset_entry = DatasetEntry(src_path, creds_key, dest_path, org_id, ds_name, token)
    dataset_entry.validate()
    connected_id = dataset_entry.connect_dataset_entry()

    result_path = f"hub://{connected_id}"
    feature_report_path(
        result_path,
        "connect",
        parameters={"Connected_Id": connected_id},
        token=token,
    )

    if verbose:
        log_dataset_connection_success(result_path)

    return result_path


class DatasetEntry:
    """Contains the information necessary to connect a dataset entry.

    Attributes:
        src_path (str): Cloud path to the source dataset.
        creds_key (str): The managed credentials used for accessing the source path.
        dest_path (str, optional): The explicit Deep Lake path to where the connected Deep Lake dataset will reside.
        org_id (str, optional): The organization to where the connected Deep Lake dataset will be added.
        ds_name (str, optional): Explicit name of the connected Deep Lake dataset.
        token (str, optional): Activeloop token used to fetch the managed credentials.
    """

    def __init__(
        self,
        src_path: str,
        creds_key: str,
        dest_path: Optional[str] = None,
        org_id: Optional[str] = None,
        ds_name: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.client = DeepLakeBackendClient(token)
        self.src_path = src_path
        self.creds_key = creds_key

        ds_info = DsInfo(dest_path=dest_path, org_id=org_id, ds_name=ds_name)
        ds_info.validate()

        self.org_id, self.ds_name = ds_info.get_org_id_and_ds_name()

    def validate(self):
        """Validates the attributes to make that dataset at ``src_path`` can be connected.

        Raises:
            InvalidSourcePathError: If the ``src_path`` is not a valid s3 or gcs path.
        """
        if is_hub_cloud_path(self.src_path):
            raise InvalidSourcePathError(
                "Source dataset is already accessible via a Deep Lake path."
            )

        if not is_path_connectable(self.src_path):
            raise InvalidSourcePathError(
                f"Source path may only be an s3 or gcs path. Got {self.src_path}."
            )

    def connect_dataset_entry(self) -> str:
        connected_id = self.client.connect_dataset_entry(
            src_path=self.src_path,
            org_id=self.org_id,
            ds_name=self.ds_name,
            creds_key=self.creds_key,
        )

        return connected_id


class DsInfo:
    """This class contains the information necessary to identify the destination place for the
    connected dataset.

    Attributes:
        dest_path (str, optional): Explicit destination Deep Lake path.
        org_id (str, optional): Organization to where the connected dataset entry is put.
        ds_name (str, optional): Explicit name for the connected dataset.
    """

    def __init__(
        self,
        *,
        dest_path: Optional[str] = None,
        org_id: Optional[str] = None,
        ds_name: Optional[str] = None,
    ) -> None:
        self.dest_path = dest_path
        self.org_id = org_id
        self.ds_name = ds_name

    def validate(self):
        """Validates the attributes to make sure that a valid destination place can be constructed.

        Raises:
            InvalidDestinationPathError: If explicit ``dest_path`` is not a valid Deep Lake path, or if neither ``dest_path`` nor ``org_id`` are specified.
        """
        if self.org_id is None:
            if self.dest_path is None:
                raise InvalidDestinationPathError(
                    "Invalid destination path. Either the organization or the destination path must be provided."
                )

            if not is_hub_cloud_path(self.dest_path):
                raise InvalidDestinationPathError(
                    "Destination path must be a Deep Lake path."
                )

    def get_org_id_and_ds_name(self) -> Tuple[str, Optional[str]]:
        """Returns the destination organization and name for the connected dataset entry to be put into.
        Guarantees to return valid ``org_id``, while ``ds_name`` is optional.

        Returns:
            A tuple containing an str ``org_id`` and optional str ``ds_name``
        """
        if self.org_id is None:
            self.org_id, self.ds_name = get_org_id_and_ds_name(self.dest_path)

        return self.org_id, self.ds_name
