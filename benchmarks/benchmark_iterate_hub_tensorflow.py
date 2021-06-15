from hub_v1 import Dataset


def benchmark_iterate_hub_tensorflow_setup(dataset_name, batch_size, prefetch_factor):
    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)
    return (loader,)


def benchmark_iterate_hub_tensorflow_run(params):
    (loader,) = params
    for _ in loader:
        pass
