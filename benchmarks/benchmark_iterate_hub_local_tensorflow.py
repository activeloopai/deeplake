from hub_v1 import Dataset
import os


def benchmark_iterate_hub_local_tensorflow_setup(
    dataset_name, dataset_split, batch_size, prefetch_factor
):
    dset = Dataset.from_tfds(dataset_name, split=dataset_split)
    path = os.path.join(".", "hub_data", "tfds")
    dset.store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")

    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    return (loader,)


def benchmark_iterate_hub_local_tensorflow_run(params):
    (loader,) = params
    for _ in loader:
        pass
