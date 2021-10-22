from .dataset import Dataset
from .hub_cloud_dataset import HubCloudDataset

from hub.util.path import is_hub_cloud_path


def get_dataset_instance(path: str, *args, **kwargs):
    # TODO docstring / rename func

    if is_hub_cloud_path(path):
        clz = HubCloudDataset
    else:
        clz = Dataset

    return clz(*args, **kwargs)
