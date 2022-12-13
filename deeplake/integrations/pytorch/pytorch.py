from typing import Callable, Dict, Optional, Sequence, Union
from deeplake.util.dataset import try_flushing
from deeplake.util.dataset import map_tensor_keys
from .common import (
    PytorchTransformFunction,
    convert_fn as default_convert_fn,
    collate_fn as default_collate_fn,
)


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
    return_index,
    pad_tensors,
    decode_method,
):
    import torch
    import torch.utils.data
    from deeplake.integrations.pytorch.dataset import SubIterableDataset

    return torch.utils.data.DataLoader(
        # this data set is more efficient also shuffles
        # using threads race conditions as source of entropy
        SubIterableDataset(
            dataset,
            tensors=tensors,
            use_local_cache=use_local_cache,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            buffer_size=buffer_size,
            return_index=return_index,
            pad_tensors=pad_tensors,
            decode_method=decode_method,
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
    from deeplake.integrations.pytorch.dataset import ShufflingIterableDataset

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
    *args,
    collate_fn: Optional[Callable],
    pin_memory: bool,
    shuffle: bool,
    buffer_size: int,
    use_local_cache: bool,
    transform: Optional[Union[Dict, Callable]] = None,
    tensors: Optional[Sequence[str]] = None,
    return_index: bool = True,
    pad_tensors: bool = True,
    torch_dataset=None,
    decode_method: Optional[Dict[str, str]] = None,
    **kwargs,
):

    import torch
    from deeplake.integrations.pytorch.dataset import TorchDataset

    if torch_dataset is None:
        torch_dataset = TorchDataset

    try_flushing(dataset)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if collate_fn is None:
        collate_fn = default_convert_fn if batch_size is None else default_collate_fn

    if tensors is not None and "index" in tensors:
        raise ValueError("index is not a tensor, to get index, pass return_index=True")

    if isinstance(transform, dict):
        tensors = [k for k in transform.keys() if k != "index"]
        transform = PytorchTransformFunction(transform_dict=transform)
    else:
        transform = PytorchTransformFunction(composite_transform=transform)

    if tensors is None:
        tensors = dataset.tensors

    for t in tensors:
        if dataset[t].is_sequence:
            raise NotImplementedError(
                f"Deep Lake’s OSS pure-python dataloader is not compatible with tensor `{t}` with htype = sequence[…]. Please use the C++ dataloader via ds.dataloader(…), which can be installed using ‘pip install deeplake[enterprise]’."
            )

    tensors = map_tensor_keys(dataset, tensors)

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
            return_index,
            pad_tensors,
            decode_method,
        )
    else:
        return torch.utils.data.DataLoader(
            torch_dataset(
                dataset,
                *args,
                tensors=tensors,
                use_local_cache=use_local_cache,
                transform=transform,
                num_workers=num_workers,
                shuffle=shuffle,
                buffer_size=buffer_size,
                return_index=return_index,
                pad_tensors=pad_tensors,
                decode_method=decode_method,
                **kwargs,
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
        )
