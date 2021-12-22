from .dataset import Dataset
from .hub_cloud_dataset import HubCloudDataset

from hub.util.path import is_hub_cloud_path

# NOTE: experimentation helper
FORCE_CLASS = None


def dataset_factory(path, *args, **kwargs):
    """Returns a Dataset object from the appropriate class. For example: If `path` is a hub
    cloud path (prefixed with `hub://`), the returned Dataset object will be of HubCloudDataset.
    """
    if FORCE_CLASS is not None:
        clz = FORCE_CLASS
    elif is_hub_cloud_path(path):
        clz = HubCloudDataset
        if "/queries/" in path:
            path, query_hash = path.split("/queries/", 1)
            return dataset_factory(
                f"{path}/queries/.queries/{query_hash}", *args, **kwargs
            )
        if "/.queries/" in path:
            path, query_hash = path.split("/.queries/", 1)
            return dataset_factory(path, *args, **kwargs)._get_stored_vds(
                query_hash, as_view=True
            )
    else:
        clz = Dataset

    if clz in {Dataset, HubCloudDataset}:
        ds = clz(path=path, *args, **kwargs)
        if ds.info.get("virtual-datasource", False):
            ds = ds._get_view()
        return ds
    raise TypeError(f"Invalid dataset class {clz}")
