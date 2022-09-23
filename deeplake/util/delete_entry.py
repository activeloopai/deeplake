from typing import Optional
from deeplake.client.client import DeepLakeBackendClient
from deeplake.util.path import is_hub_cloud_path


def remove_path_from_backend(path: str, token: Optional[str] = None) -> None:
    if is_hub_cloud_path(path):
        client = DeepLakeBackendClient(token=token)
        split_path = path.split("/")
        org_id, ds_name = split_path[2], split_path[3]
        client.delete_dataset_entry(org_id, ds_name)
