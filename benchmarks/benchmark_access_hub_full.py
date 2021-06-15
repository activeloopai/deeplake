from hub_v1 import Dataset


def benchmark_access_hub_full_setup(dataset_name, field=None):
    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    keys = dset.keys
    if field is not None:
        keys = (field,)
    return (dset, keys)


def benchmark_access_hub_full_run(params):
    dset, keys = params
    for k in keys:
        dset[k].compute()
