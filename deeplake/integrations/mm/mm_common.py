import os
import torch
import mmcv  # type: ignore
import deeplake as dp
from deeplake.util.warnings import always_warn


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


def load_ds_from_cfg(cfg: mmcv.utils.config.ConfigDict):
    creds = cfg.get("deeplake_credentials", {})
    token = creds.get("token", None)
    if token is None:
        uname = creds.get("username")
        if uname is not None:
            raise NotImplementedError(
                "Username/Password based authentication from deeplake has been deprecated. Please specify a token in the config."
            )
    ds_path = cfg.deeplake_path
    ds = dp.load(ds_path, token=token, read_only=True)
    deeplake_commit = cfg.get("deeplake_commit")
    deeplake_view_id = cfg.get("deeplake_view_id")
    deeplake_query = cfg.get("deeplake_query")

    if deeplake_view_id and deeplake_query:
        raise Exception(
            "A query and view_id were specified simultaneously for a dataset in the config. Please specify either the deeplake_query or the deeplake_view_id."
        )

    if deeplake_commit:
        ds.checkout(deeplake_commit)

    if deeplake_view_id:
        ds = ds.load_view(id=deeplake_view_id)

    if deeplake_query:
        ds = ds.query(deeplake_query)

    return ds


def get_collect_keys(cfg):
    pipeline = cfg.train_pipeline
    for transform in pipeline:
        if transform["type"] == "Collect":
            return transform["keys"]
    raise ValueError("collection keys were not specified")


def check_persistent_workers(train_persistent_workers, val_persistent_workers):
    if train_persistent_workers != val_persistent_workers:
        if train_persistent_workers:
            always_warn(
                "persistent workers for training and evaluation should be identical, "
                "otherwise, this could lead to performance issues. "
                "Either both of then should be `True` or both of them should `False`. "
                "If you want to use persistent workers set True for validation"
            )
        else:
            always_warn(
                "persistent workers for training and evaluation should be identical, "
                "otherwise, this could lead to performance issues. "
                "Either both of then should be `True` or both of them should `False`. "
                "If you want to use persistent workers set True for training"
            )


def find_tensor_with_htype(ds: dp.Dataset, htype: str, mm_class=None):
    tensors = [k for k, v in ds.tensors.items() if v.meta.htype == htype]
    if mm_class is not None:
        always_warn(
            f"No deeplake tensor name specified for '{mm_class} in config. Fetching it using htype '{htype}'."
        )
    if not tensors:
        always_warn(f"No tensor found with htype='{htype}'")
        return None
    t = tensors[0]
    if len(tensors) > 1:
        always_warn(f"Multiple tensors with htype='{htype}' found. choosing '{t}'.")
    return t
