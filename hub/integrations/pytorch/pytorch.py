from typing import Callable, Dict, Optional, Sequence, Union
from hub.util.dataset import try_flushing
from hub.util.dataset import map_tensor_keys
from .common import (
    PytorchTransformFunction,
    convert_fn as default_convert_fn,
    collate_fn as default_collate_fn,
)
from hub.util.exceptions import EmptyTensorError


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
    return_index,
    pad_tensors,
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
            batch_size=batch_size,
            num_workers=num_workers,
            buffer_size=buffer_size,
            return_index=return_index,
            pad_tensors=pad_tensors,
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
    return_index: bool = True,
    pad_tensors: bool = True,
):

    import torch
    from hub.integrations.pytorch.dataset import TorchDataset

    try_flushing(dataset)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn

    if tensors is not None and "index" in tensors:
        raise ValueError("index is not a tensor, to get index, pass return_index=True")

    tensors = map_tensor_keys(dataset, tensors)
    if isinstance(transform, dict):
        tensors = [k for k in transform.keys() if k != "index"]
        transform = PytorchTransformFunction(transform_dict=transform)
    else:
        transform = PytorchTransformFunction(composite_transform=transform)

    # check whether we have an empty tensor inside of tensors
    for tensor_name in tensors:
        tensor = dataset._get_tensor_from_root(tensor_name)
        if len(tensor) == 0:
            raise EmptyTensorError(
                f" the dataset has an empty tensor {tensor_name}, pytorch dataloader can't be created."
                f" Please either populate the tensor or pass tensors argument to .pytorch that excludes this"
                f" tensor."
            )

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
            return_index,
            pad_tensors,
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
                return_index=return_index,
                pad_tensors=pad_tensors,
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
        )
