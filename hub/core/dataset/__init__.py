from .dataset import Dataset
from .hub_cloud_dataset import HubCloudDataset

from hub.util.path import is_hub_cloud_path


# NOTE: experimentation helper
FORCE_CLASS = None


def get_dataset_instance(path, *args, **kwargs):
    # TODO docstring / rename func

    if FORCE_CLASS is not None:
        clz = FORCE_CLASS
    elif is_hub_cloud_path(path):
        clz = HubCloudDataset
    else:
        clz = Dataset

    if clz is Dataset:
        return clz(*args, **kwargs)
    elif clz is HubCloudDataset:
        return clz(path, *args, **kwargs)
    raise TypeError(f"Invalid dataset class {clz}")
