from typing import Callable, Optional, Sequence
from hub.util.dataset import try_flushing
from .common import convert_fn as default_convert_fn, collate_fn as default_collate_fn


def create_dataloader_nesteddataloader(
    dataset,
    tensors,
    use_local_cache,
    transform,
    num_workers,
    buffer_size,
    batch_size,
    collate_fn,
    pin_memory,
    drop_last,
):
    import torch
    from hub.integrations.pytorch.dataset import SubIterableDataset

    return torch.utils.data.DataLoader(
        # this data set is more efficient also shuffles
        # using threads race conditions as source of entropy
        SubIterableDataset(
            dataset,
            tensors=tensors,
            use_local_cache=use_local_cache,
            transform=transform,
            num_workers=num_workers,
            buffer_size=buffer_size,
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloader_shufflingdataloader(
    dataset,
    tensors,
    use_local_cache,
    transform,
    num_workers,
    buffer_size,
    batch_size,
    collate_fn,
    pin_memory,
    drop_last,
):
    import torch
    from hub.integrations.pytorch.dataset import ShufflingIterableDataset

    return torch.utils.data.DataLoader(
        # this data set is more efficient also shuffles
        # using threads race conditions as source of entropy
        ShufflingIterableDataset(
            dataset,
            tensors=tensors,
            use_local_cache=use_local_cache,
            transform=transform,
            num_workers=num_workers,
            buffer_size=buffer_size,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


create_dataloader = create_dataloader_shufflingdataloader


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
    buffer_size: int = 0,
    use_local_cache: bool = False,
):

    import torch
    from hub.integrations.pytorch.dataset import TorchDataset

    try_flushing(dataset)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn

    if shuffle and num_workers > 0:
        return create_dataloader(
            dataset,
            tensors,
            use_local_cache,
            transform,
            num_workers,
            buffer_size,
            batch_size,
            collate_fn,
            pin_memory,
            drop_last,
        )
    else:
        return torch.utils.data.DataLoader(
            TorchDataset(
                dataset,
                tensors=tensors,
                use_local_cache=use_local_cache,
                transform=transform,
                num_workers=num_workers,
                shuffle=False,
                buffer_size=buffer_size,
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
        )
