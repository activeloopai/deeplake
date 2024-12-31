import os
import torch
import warnings
import mmcv  # type: ignore
import deeplake as dp
from deeplake.types import TypeKind
from deeplake.integrations.mm.warnings import always_warn
from deeplake.integrations.mm.exceptions import EmptyTokenException
from deeplake.integrations.constants import DEEPLAKE_AUTH_TOKEN


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
    deeplake_commit = cfg.get("deeplake_commit")
    deeplake_tag_id = cfg.get("deeplake_tag_id")
    deeplake_query = cfg.get("deeplake_query")
    token = token or os.environ.get(DEEPLAKE_AUTH_TOKEN)
    if token is None:
        raise EmptyTokenException()

    try:
        ds = dp.open_read_only(cfg.deeplake_path, token=token, creds=creds)
    except:
        if not deeplake_query:
            raise
        ds = dp.query(deeplake_query)

    if deeplake_tag_id and deeplake_query:
        raise Exception(
            "A query and view_id were specified simultaneously for a dataset in the config. Please specify either the deeplake_query or the deeplake_tag_id."
        )

    if deeplake_commit:
        ds.checkout(deeplake_commit)

    if deeplake_tag_id:
        ds = ds.tags(deeplake_tag_id).open()

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


def find_image_tensor(ds: dp.Dataset, mm_class=None):
    images = [
        col.name
        for col in ds.schema.columns
        if ds.schema[col.name].dtype.is_image
    ]
    if mm_class is not None:
        always_warn(
            f"No deeplake column name specified for '{mm_class} in config. Fetching it using type_kind '{TypeKind.Image}'."
        )
    if not images:
        always_warn(f"No column found with type_kind='{TypeKind.Image}'")
        return None
    t = images[0]
    if len(images) > 1:
        always_warn(
            f"Multiple columns with type_kind='{TypeKind.Image}' found. choosing '{t}'."
        )
    print(f"columns {images} kind {TypeKind.Image} mm_class {mm_class} t {t}")
    return t


def find_smask_tensor(ds: dp.Dataset, mm_class=None):
    smasks = [
        col.name
        for col in ds.schema.columns
        if ds.schema[col.name].dtype.is_segment_mask
    ]
    if mm_class is not None:
        always_warn(
            f"No deeplake column name specified for '{mm_class} in config. Fetching it using type_kind '{TypeKind.SegmentMask}'."
        )
    if not smasks:
        always_warn(f"No column found with type_kind='{TypeKind.SegmentMask}'")
        return None
    t = smasks[0]
    if len(smasks) > 1:
        always_warn(
            f"Multiple columns with type_kind='{TypeKind.SegmentMask}' found. choosing '{t}'."
        )
    print(f"columns {smasks} kind {TypeKind.SegmentMask} mm_class {mm_class} t {t}")
    return t


def find_tensor_with_htype(ds: dp.Dataset, type_kind=TypeKind.Image, mm_class=None):
    colunms = [col.name for col in ds.schema.columns if col.dtype.kind == type_kind]
    if mm_class is not None:
        always_warn(
            f"No deeplake column name specified for '{mm_class} in config. Fetching it using type_kind '{type_kind}'."
        )
    if not colunms:
        always_warn(f"No column found with type_kind='{type_kind}'")
        return None
    t = colunms[0]
    if len(colunms) > 1:
        always_warn(
            f"Multiple columns with type_kind='{type_kind}' found. choosing '{t}'."
        )

    print(f"columns {colunms} kind {type_kind} mm_class {mm_class} t {t}")
    return t


def check_unsupported_functionalities(cfg):
    check_unused_dataset_fields(cfg)
    check_unsupported_train_pipeline_fields(cfg, mode="train")
    check_unsupported_train_pipeline_fields(cfg, mode="val")
    check_dataset_augmentation_formats(cfg)


def check_unused_dataset_fields(cfg):
    if cfg.get("dataset_type"):
        always_warn(
            "The deeplake mmdet integration does not use dataset_type to work with the data and compute metrics. All deeplake datasets are in the same deeplake format. To specify a metrics format, you should deeplake_metrics_format "
        )

    if cfg.get("data_root"):
        always_warn(
            "The deeplake mmdet integration does not use data_root, this input will be ignored"
        )


def check_unsupported_train_pipeline_fields(cfg, mode="train"):
    transforms = cfg.data[mode].pipeline

    for transform in transforms:
        transform_type = transform.get("type")

        if transform_type == "LoadImageFromFile":
            always_warn(
                "LoadImageFromFile is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "LoadAnnotations":
            always_warn(
                "LoadAnnotations is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "Corrupt":
            raise Exception("Corrupt augmentation is not supported yet.")

        elif transform_type == "CopyPaste":  # TO DO: @adolkhan resolve this
            raise Exception("CopyPaste augmentation is not supported yet")

        elif transform_type == "CutOut":  # TO DO: @adolkhan resolve this
            raise Exception("CutOut augmentation is not supported yet")

        elif transform_type == "Mosaic":  # TO DO: @adolkhan resolve this
            raise Exception("Mosaic augmentation is not supported yet")


def check_dataset_augmentation_formats(cfg):
    if cfg.get("train_dataset"):
        always_warn(
            "train_dataset is going to be unused. Dataset types like: ConcatDataset, RepeatDataset, ClassBalancedDataset, MultiImageMixDataset are not supported."
        )


def get_pipeline(cfg, *, name: str, generic_name: str):
    pipeline = cfg.data[name].get("pipeline", None)
    if pipeline is None:
        warnings.warn(
            f"Warning: The '{name}' data pipeline is missing in the configuration. Attempting to locate in '{generic_name}'."
        )

        pipeline = cfg.get(generic_name, [])

    return pipeline
