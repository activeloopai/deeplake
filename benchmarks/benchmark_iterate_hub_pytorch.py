import torch

from hub import Dataset


def benchmark_iterate_hub_pytorch_setup(
    dataset_name, batch_size, prefetch_factor, num_workers=1
):
    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    return (loader,)


def benchmark_iterate_hub_pytorch_run(params):
    (loader,) = params
    for _ in loader:
        pass
