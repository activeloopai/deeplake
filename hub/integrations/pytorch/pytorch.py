from typing import Callable, Dict, Optional, Sequence, Union
from hub.util.dataset import try_flushing
from hub.util.dataset import map_tensor_keys
from .common import (
    PytorchTransformFunction,
    convert_fn as default_convert_fn,
    collate_fn as default_collate_fn,
)


def create_dataloader_nesteddataloader(
    dataset,
    tensors,
    tobytes,
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
    import torch.utils.data
    from hub.integrations.pytorch.dataset import SubIterableDataset

    return torch.utils.data.DataLoader(
        # this data set is more efficient also shuffles
        # using threads race conditions as source of entropy
        SubIterableDataset(
            dataset,
            tensors=tensors,
            tobytes=tobytes,
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


def create_dataloader_shufflingdataloader(
    dataset,
    tensors,
    tobytes,
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
    import torch.utils.data
    from hub.integrations.pytorch.dataset import ShufflingIterableDataset

    return torch.utils.data.DataLoader(
        # this data set is more efficient also shuffles
        # using threads race conditions as source of entropy
        ShufflingIterableDataset(
            dataset,
            tensors=tensors,
            tobytes=tobytes,
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


create_dataloader = create_dataloader_nesteddataloader


def dataset_to_pytorch(
    dataset,
    num_workers: int,
    batch_size: int,
    drop_last: bool,
    collate_fn: Optional[Callable],
    pin_memory: bool,
    shuffle: bool,
    buffer_size: int,
    use_local_cache: bool,
    transform: Optional[Union[Dict, Callable]] = None,
    tensors: Optional[Sequence[str]] = None,
    tobytes: Union[bool, Sequence[str]] = False,
):

    import torch
    from hub.integrations.pytorch.dataset import TorchDataset

    try_flushing(dataset)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn

    tensors = map_tensor_keys(dataset, tensors)
    if isinstance(transform, dict):
        tensors = list(transform.keys())
        transform = PytorchTransformFunction(transform_dict=transform, tensors=tensors)
    else:
        transform = PytorchTransformFunction(composite_transform=transform)

    if shuffle and num_workers > 0:
        return create_dataloader(
            dataset,
            tensors,
            tobytes,
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
                tobytes=tobytes,
                use_local_cache=use_local_cache,
                transform=transform,
                num_workers=num_workers,
                shuffle=shuffle,
                buffer_size=buffer_size,
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
        )
