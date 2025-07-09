# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import numpy as np
import warnings
import logging

from mmengine.config import Config, DictAction  # type: ignore
from mmengine.logging import print_log  # type: ignore
from mmengine.runner import Runner  # type: ignore
from mmseg.registry import RUNNERS  # type: ignore

from typing import Dict

import mmengine  # type: ignore

from deeplake.integrations.mmlab.segmentation.basedataset import DeeplakeBaseDataset

from deeplake.integrations.mmlab.segmentation.builder_patch import build_func_patch

mmengine.dataset.BaseDataset = DeeplakeBaseDataset
from deeplake.integrations.mmlab.segmentation.transform import transform

mmengine.registry.DATASETS.build = build_func_patch

from mmengine.runner import Runner  # type: ignore
from typing import Optional, Union
from torch.utils.data import DataLoader  # type: ignore
import copy
from functools import partial
from mmengine.utils.dl_utils import TORCH_VERSION  # type: ignore
from mmengine.dataset import worker_init_fn as default_worker_init_fn  # type: ignore

from mmengine.registry import DATA_SAMPLERS, DATASETS, FUNCTIONS  # type: ignore
from mmengine.utils import digit_version  # type: ignore
from mmengine.runner.utils import _get_batch_size  # type: ignore
from deeplake.enterprise.dataloader import DeepLakeDataLoader
from deeplake.util.exceptions import ClassNamesEmptyError

_original_build_dataloader = Runner.build_dataloader

from mmengine.dist import (  # type: ignore
    broadcast,
    get_dist_info,
    get_rank,
    get_world_size,
    init_dist,
    is_distributed,
    master_only,
)


def __generate_palette(num_classes):
    """Function to generate a random but distinguishable color palette"""
    import random

    random.seed(42)
    return [
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for _ in range(num_classes)
    ]


def build_dataloader(
    dataloader: Union[DeepLakeDataLoader, Dict],
    seed: Optional[int] = None,
    diff_rank_seed: bool = False,
) -> DeepLakeDataLoader:
    if isinstance(dataloader, DeepLakeDataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop("dataset")
    dataset, transform_pipeline = build_func_patch(dataset_cfg)

    num_batch_per_epoch = dataloader_cfg.pop("num_batch_per_epoch", None)
    # if num_batch_per_epoch is not None:
    #     world_size = get_world_size()
    #     num_samples = (
    #         num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
    #         world_size)
    #     dataset = _SlicedDataset(dataset, num_samples)

    # build sampler
    sampler_cfg = dataloader_cfg.pop("sampler")
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg, default_args=dict(dataset=dataset, seed=sampler_seed)
        )
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop("batch_sampler", None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler, batch_size=dataloader_cfg.pop("batch_size")
            ),
        )
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]
    if "worker_init_fn" in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop("worker_init_fn")
        worker_init_fn_type = worker_init_fn_cfg.pop("type")
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
        elif callable(worker_init_fn_type):
            worker_init_fn = worker_init_fn_type
        else:
            raise TypeError(
                "type of worker_init_fn should be string or callable "
                f"object, but got {type(worker_init_fn_type)}"
            )
        assert callable(worker_init_fn)
        init_fn = partial(worker_init_fn, **worker_init_fn_cfg)  # type: ignore
    else:
        if seed is not None:
            disable_subprocess_warning = dataloader_cfg.pop(
                "disable_subprocess_warning", False
            )
            assert isinstance(disable_subprocess_warning, bool), (
                "disable_subprocess_warning should be a bool, but got "
                f"{type(disable_subprocess_warning)}"
            )
            init_fn = partial(
                default_worker_init_fn,
                num_workers=dataloader_cfg.get("num_workers"),
                rank=get_rank(),
                seed=seed,
                disable_subprocess_warning=disable_subprocess_warning,
            )
        else:
            init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if "persistent_workers" in dataloader_cfg and digit_version(
        TORCH_VERSION
    ) < digit_version("1.7.0"):
        print_log(
            "`persistent_workers` is only available when " "pytorch version >= 1.7",
            logger="current",
            level=logging.WARNING,
        )
        dataloader_cfg.pop("persistent_workers")

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop("collate_fn", dict(type="pseudo_collate"))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop("type")
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            "collate_fn should be a dict or callable object, but got "
            f"{collate_fn_cfg}"
        )

    deeplake_ds = dataset.deeplake_dataset
    images_tensor = dataset.images_tensor
    masks_tensor = dataset.masks_tensor
    classes = deeplake_ds[masks_tensor].info.class_names
    dataset.CLASSES = classes
    if not classes or not len(classes):
        raise ClassNamesEmptyError(masks_tensor)

    num_workers = dataloader_cfg.get("num_workers", 0)
    batch_size = dataloader_cfg.get("batch_size", 1)
    shuffle = dataloader_cfg.get("shuffle", False)
    tensors = [images_tensor, masks_tensor]
    drop_last = dataloader_cfg.get("drop_last", False)
    persistent_workers = dataloader_cfg.get("persistent_workers", False)

    transform_fn = partial(
        transform,
        images_tensor=images_tensor,
        masks_tensor=masks_tensor,
        pipeline=transform_pipeline,
    )

    loader = (
        deeplake_ds.dataloader(ignore_errors=True)
        .transform(transform_fn)
        .shuffle(shuffle)
        .batch(batch_size=batch_size, drop_last=drop_last)
        .pytorch(
            num_workers=num_workers,
            collate_fn=collate_fn,
            tensors=tensors,
            distributed=is_distributed(),
            persistent_workers=persistent_workers,
        )
    )

    loader.dataset.__setattr__(
        "metainfo",
        {"classes": tuple(classes), "palette": __generate_palette(len(classes))},
    )

    if init_fn:
        loader.worker_init_fn = init_fn

    return loader


Runner.build_dataloader = staticmethod(build_dataloader)
