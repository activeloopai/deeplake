from typing import Optional
from hub.client.client import HubBackendClient
from hub.util.path import is_hub_cloud_path


def remove_path_from_backend(path: str, token: Optional[str] = None) -> None:
    if is_hub_cloud_path(path):
        client = HubBackendClient(token=token)
        split_path = path.split("/")
        org_id, ds_name = split_path[2], split_path[3]
        client.delete_dataset_entry(org_id, ds_name)
