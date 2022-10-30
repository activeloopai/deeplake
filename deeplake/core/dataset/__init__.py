from .dataset import Dataset  # type: ignore
from .deeplake_cloud_dataset import DeepLakeCloudDataset
from .view_entry import ViewEntry

from deeplake.util.path import is_hub_cloud_path

# NOTE: experimentation helper
FORCE_CLASS = None


def dataset_factory(path, *args, **kwargs):
    """Returns a Dataset object from the appropriate class. For example: If `path` is a Deep Lake
    cloud path (prefixed with `hub://`), the returned Dataset object will be of DeepLakeCloudDataset.
    """
    if FORCE_CLASS is not None:
        clz = FORCE_CLASS
    elif is_hub_cloud_path(path):
        clz = DeepLakeCloudDataset
    else:
        clz = Dataset

    if clz in {Dataset, DeepLakeCloudDataset}:
        ds = clz(path=path, *args, **kwargs)
        if ds.root.info.get("virtual-datasource", False):
            ds = ds._get_view()
        return ds
    raise TypeError(f"Invalid dataset class {clz}")
