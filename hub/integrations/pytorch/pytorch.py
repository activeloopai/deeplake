from typing import Callable, Optional, Sequence
from hub.util.dataset import try_flushing
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn


def dataset_to_pytorch(
    dataset,
    transform: Optional[Callable] = None,
    tensors: Optional[Sequence[str]] = None,
    num_workers: int = 1,
    batch_size: int = 1,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    shuffle: bool = False,
    buffer_size: int = 10 * 1000,
    use_local_cache: bool = False,
):

    import torch
    from hub.integrations.pytorch.dataset import TorchDataset

    try_flushing(dataset)

    torch.multiprocessing.set_sharing_strategy("file_system")

    return torch.utils.data.DataLoader(
        TorchDataset(
            dataset,
            tensors=tensors,
            use_local_cache=use_local_cache,
            transform=transform,
            num_workers=num_workers,
            shuffle=shuffle,
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=default_convert_fn if batch_size is None else default_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=batch_size * 2,
    )
