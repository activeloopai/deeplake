from typing import Tuple, Optional

from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.log import logger
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
) -> str:
    dataset_entry = DatasetEntry(src_path, creds_key, dest_path, org_id, ds_name, token)
    dataset_entry.validate()

    return dataset_entry.connect_dataset_entry()


class DatasetEntry:
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
        """Org id must either be specified or be properly infered from the explicit destination path."""
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
        if self.org_id is None:
            self.org_id, self.ds_name = get_org_id_and_ds_name(self.dest_path)

        return self.org_id, self.ds_name
