import torch

from hub import Dataset


def benchmark_iterate_hub_local_pytorch_setup(dataset_name, dataset_split, batch_size, prefetch_factor, num_workers=1):
    dset = Dataset.from_tfds(dataset_name, split=dataset_split)
    path = os.path.join(".", "hub_data", "tfds")
    dset.store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")

    loader = torch.utils.data.DataLoader(dset.to_pytorch(
    ), batch_size=batch_size, prefetch_factor=prefetch_factor, num_workers=num_workers)

    return (loader,)


def benchmark_iterate_hub_local_pytorch_run(params):
    loader, = params
    for _ in loader:
        pass
