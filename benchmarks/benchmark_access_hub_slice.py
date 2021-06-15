from hub_v1 import Dataset


def benchmark_access_hub_slice_setup(dataset_name, slice_bounds, field=None):
    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    keys = dset.keys
    if field is not None:
        keys = (field,)
    return (dset, slice_bounds, keys)


def benchmark_access_hub_slice_run(params):
    dset, slice_bounds, keys = params
    for k in keys:
        dset[k][slice_bounds[0] : slice_bounds[1]].compute()
