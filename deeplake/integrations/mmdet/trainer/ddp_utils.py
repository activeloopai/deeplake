import torch
from mmdet.utils.util_distribution import *  # type: ignore
import os


def ddp_setup(rank: int, world_size: int, port: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        port: Port number
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )


def force_cudnn_initialization(device_id):
    dev = torch.device(f"cuda:{device_id}")
    torch.nn.functional.conv2d(
        torch.zeros(32, 32, 32, 32, device=dev), torch.zeros(32, 32, 32, 32, device=dev)
    )


def build_ddp(model, device, *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is mlu, return a MLUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.
        args (List): arguments to be passed to ddp_factory
        kwargs (dict): keyword arguments to be passed to ddp_factory

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """

    assert device in ["cuda", "mlu"], "Only available for cuda or mlu devices."
    if device == "cuda":
        model = model.cuda(kwargs["device_ids"][0])  # patch
    elif device == "mlu":
        from mmcv.device.mlu import MLUDistributedDataParallel  # type: ignore

        ddp_factory["mlu"] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)
