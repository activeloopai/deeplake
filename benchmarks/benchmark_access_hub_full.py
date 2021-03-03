from hub import dataset


def benchmark_access_hub_full_setup(dataset_name):
    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    return (dset,)


def benchmark_access_hub_full_run(params):
    dset, = params
    for k in dset.keys:
        dset[k].compute()
