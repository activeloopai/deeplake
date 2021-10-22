from .dataset import Dataset
from .hub_cloud_dataset import HubCloudDataset

from hub.util.path import is_hub_cloud_path


# NOTE: overriding this is how we can manually create sign-wall datasets for testing
FORCE_CLASS = None


def get_dataset_instance(path, *args, **kwargs):
    # TODO docstring / rename func

    if FORCE_CLASS is not None:
        clz = FORCE_CLASS
    elif is_hub_cloud_path(path):
        clz = HubCloudDataset
    else:
        clz = Dataset

    return clz(path, *args, **kwargs)
