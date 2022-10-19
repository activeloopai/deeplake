from typing import Tuple, Optional
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.path import is_hub_cloud_path, get_path_type, get_org_id_and_ds_name
from deeplake.util.exceptions import InvalidDestinationPathError, InvalidSourcePathError


def connect_dataset_entry(
    src_path: str,
    creds_key: str,
    dest_path: Optional[str],
    org_id: Optional[str],
    ds_name: Optional[str],
    token: Optional[str],
) -> str:
    if is_hub_cloud_path(src_path):
        raise InvalidSourcePathError(
            "Source dataset is already a Deep Lake Cloud dataset."
        )

    if not is_path_connectable(src_path):
        raise InvalidSourcePathError(
            f"Source path may only be an s3 or gcs path. Got {src_path}"
        )

    client = DeepLakeBackendClient(token)
    org_id, ds_name = _get_org_id_and_ds_name(
        dest_path=dest_path, org_id=org_id, ds_name=ds_name
    )

    connected_id = client.connect_dataset_entry(
        src_path=src_path, org_id=org_id, ds_name=ds_name, creds_key=creds_key
    )

    return connected_id


def is_path_connectable(path: str) -> bool:
    return get_path_type(path) in ("s3", "gcs")


def _get_org_id_and_ds_name(
    *, dest_path: Optional[str], org_id: Optional[str], ds_name: Optional[str]
) -> Tuple[str]:
    if org_id is None:
        if dest_path is None:
            raise InvalidDestinationPathError(
                "Invalid destination path. Either the organization id or the destination path must be provided."
            )

        if not is_hub_cloud_path(dest_path):
            raise InvalidDestinationPathError(
                "Destination path must be a path like hub://organization/dataset_name"
            )

        org_id, ds_name = get_org_id_and_ds_name(dest_path)

    return org_id, ds_name
