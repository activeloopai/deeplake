"""
Deep Lake offers an integration with MMSegmentation, a popular open-source semantic segmentation toolbox based on PyTorch.
The integration enables users to train models while streaming Deep Lake dataset using the transformation, training, and evaluation tools built by MMSeg.

Learn more about MMSegmentation `here <https://mmsegmentation.readthedocs.io/en/latest/>`_.

Integration Interface
~~~~~~~~~~~~~~~~~~~~~
MMSegmentation works with configs. Deeplake adopted this strategy, and in order to train MMSeg models, you need to create/specify your model
and training/validation config. Deep Lake integration's logic is almost the same as MMSegmentation's with some minor modifications. The integrations
with MMSeg occurs in the deeplake.integrations.mmseg module. At a high-level, Deep Lake is responsible for the pytorch dataloader that streams data
to the training framework, while MMSeg is used for the training, transformation, and evaluation logic. Let us take a look at the config with deeplake changes:

Deeplake integration requires the following parameters to be specified in the configuration file:

- ``data``: Just like in the MMSegmentation configuration files, in data dictionary you can specify everything that you want to be applied to the data during training and validation
    - ``train``: Keyword argument of data, a dictionary where one can specify dataset path, credentials, transformations of the training data
    - ``val``: Keyword argument of data, a dictionary where one can specify dataset path, credentials, transformations of the validation data
    - ``pipeline``: List of transformations. This parameter exists for train as well as for val.

        - Example:

            >>> pipeline =  [dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True), dict(type="RandomFlip", flip_ratio=0.5), dict(type="PhotoMetricDistortion")]

    - ``deeplake_path``: Path to the deeplake dataset. This parameter exists for train as well as for val.
    - ``deeplake_credentials``: Optional parameter. Required only when using private nonlocal datasets. See documendataion for `deeplake.open_read_only() https://docs.deeplake.ai/latest/api/dataset/#deeplake.open_read_only`_ for details. This parameter exists for train as well as for val.
    - ``deeplake_tag_id``: Optional parameter. If specified, the dataset will checkout to the commit. This parameter exists for train as well as for val. See documentation for `Dataset.commit_id <https://deep-lake--2152.org.readthedocs.build/en/2152/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.commit_id>`_
    - ``deeplake_query``: Optional paramterer. If specified, the dataset can be loaded from the query is dataset_path was not been specified as well as the applied on that dataset of dataset_path was specified before
    - ``deeplake_tensors``: Optional parameter. If specified maps MMSegmentation tensors to the associated tensors in the dataset. MMSeg tensors are: "img", "gt_semantic_seg". This parameter exists for train as well as for val.
        - ``"img"``: Stands for image tensor.
        - ``"gt_semantic_seg"``: Stands for semantic segmenataion tensor.

    - ``deeplake_dataloader``: Optional parameter. If specified represents the parameters of the deeplake dataloader. Deeplake dataloader parameters are: "shuffle", "batch_size", "num_workers". This parameter exists for train as well as for val.
        - ``"shuffle"``: If ``True`` shuffles the dataset.
        - ``"batch_size"``: Size of batch. If not specified, dataloader will use ``samples_per_gpu``.
        - ``"num_workers"``: Number of workers to use. If not specified, dataloader will use ``workers_per_gpu``.

Example:

>>> evaluation = dict(metric=["mIoU"], interval=1)

- ``train_segmentor``: Function to train the MMSegmentation model.

    Parameters:

        - ``model``: MMSegmentation model that is going to be used.
        - ``cfg``: mmcv.ConfigDict, Configuration of the model as well as of the datasets and transforms that's going to be used.
        - ``ds_train``: Optional parameter. If provided will overwrite deeplake_path in train, and will pass this tensor directly to the dataloader.
        - ``ds_val``: Optional parameter. If provided will overwrite deeplake_path in val, and will pass this tensor directly to the dataloader.
        - ``ds_train_tensors``: Optional parameter. If provided will overwrite deeplake_tensors in train, and will pass this tensor mapping directly to dataloader.
        - ``ds_val_tensors``: Optional parameter. If provided will overwrite deeplake_tensors in val, and will pass this tensor mapping directly to dataloader.
        - ``distributed``: Optional parameter. If provided will run the code on all available gpus. Meta data used to build runner.
        - ``timestamp``: Variable used in runner to make .log and .log.json filenames the same.
        - ``validate``: Bool, whether validation should be run, defaults to ``True``.


MMSegmentation Config Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Below is the example of the deeplake mmseg configuration:


>>> _base_ = "../mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_4x4_160k_coco-stuff164k.py"
>>> device = "cuda"
>>> work_dir "./"
>>> seed = 0
>>> gpu_ids = range(1)
>>> # use caffe img_norm
>>> img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
>>> crop_size = (256, 256)
>>> reduce_zero_label = False
>>> train_pipeline = [
...     dict(type="LoadImageFromFile"),
...     dict(type="LoadAnnotations"),
...     dict(type="Resize", img_scale=(320, 240), ratio_range=(0.5, 2.0)),
...     dict(type="RandomCrop", crop_size=crop_size),
...     dict(type="RandomFlip", flip_ratio=0.5),
...     dict(type="PhotoMetricDistortion"),
...     dict(type="Normalize", **img_norm_cfg),
...     dict(type="Pad", size=crop_size, pad_val=0),
...     dict(type="DefaultFormatBundle"),
...     dict(type="Collect", keys=["img", "gt_semantic_seg"]),
... ]
>>> test_pipeline = [
...     dict(type="LoadImageFromFile"),
...     dict(
...         type="MultiScaleFlipAug",
...         img_scale=(320, 240),
...         flip=False,
...         transforms=[
...             dict(type="Resize", img_scale=(320, 240), keep_ratio=True),
...             dict(type="RandomFlip"),
...             dict(type="Normalize", **img_norm_cfg),
...             dict(type="ImageToTensor", keys=["img"]),
...             dict(type="Collect", keys=["img"]),
...         ],
...     ),
... ]
>>> evaluation = dict(metric=["mIoU"], interval=500)
>>> #--------------------------------------DEEPLAKE INPUTS------------------------------------------------------------#
>>> TOKEN = "INSERT_YOUR_DEEPLAKE_TOKEN"
>>> data = dict(
...     # samples_per_gpu=4, # Is used instead of batch_size if deeplake_dataloader is not specified below
...     # workers_per_gpu=8, # Is used instead of num_workers if deeplake_dataloader is not specified below
...     train=dict(
...         pipeline=train_pipeline,
...         # Credentials for authentication. See documendataion for deeplake.load() for details
...         deeplake_path="hub://activeloop/semantic-seg-train",
...          deeplake_credentials={
...             "token": TOKEN,
...             "creds": None,
...         },
...         #OPTIONAL - load deeplake dataset from a query
...         deeplake_query = "",
...         #OPTIONAL - Loads a dataset tag for training based on tag_id
...         deeplake_tag_id="",
...         # OPTIONAL - {"mmseg_key": "deep_lake_tensor",...} - Maps Deep Lake tensors to MMSeg dictionary keys.
...         # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
...         deeplake_tensors = {"img": "images", "gt_semantic_seg": "semantic_seg"},
...         # OPTIONAL - Parameters to use for the Deep Lake dataloader. If unspecified, the integration uses
...         # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
...         deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
...     ),
...     # Parameters as the same as for train
...     val=dict(
...         pipeline=test_pipeline,
...         deeplake_path="hub://activeloop/semantic-seg-val",
...         deeplake_credentials={
...             "token": TOKEN,
...             "creds": None,
...         },
...         deeplake_tensors = {"img": "images", "gt_semantic_seg": "semantic_seg"},
...         deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
...     ),
... )
>>> # Which dataloader to use
>>> # Which metrics to use for evaulation. In MMSeg (without Deeplake), this is inferred from the dataset type.
>>> # In the Deep Lake integration, since the format is standardized, a variety of metrics can be used for a given dataset.
>>> #----------------------------------END DEEPLAKE INPUTS------------------------------------------------------------#

And config for training:

>>> import os
>>> from mmcv import Config
>>> import mmcv
>>> from deeplake.integrations import mmseg as mmseg_deeplake
>>> cfg = Config.fromfile(cfg_file)
>>> # Build the segmentor
>>> model = mmseg_deeplake.build_segmentor(cfg.model)
>>> # Create work_dir
>>> mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
>>> # Run the training
>>> mmseg_deeplake.train_segmentor(model, cfg, distributed=args.distributed, validate=args.validate)
"""

import warnings
import torch
import numpy as np
import io
import math
import types
from functools import partial

from typing import Callable, Optional, List, Dict, Sequence, Union
from PIL import Image  # type: ignore

from mmseg.core import DistEvalHook, EvalHook  # type: ignore
from mmseg.core import build_optimizer
from mmseg.utils import (  # type: ignore
    build_dp,
    find_latest_checkpoint,
    get_root_logger,
)

from mmseg.datasets.samplers import DistributedSampler  # type: ignore
from mmseg.utils.util_distribution import *  # type: ignore
from deeplake.integrations.mm.get_indexes import get_indexes
from deeplake.integrations.mm.worker_init_fn import worker_init_fn
from deeplake.integrations.mm.ipc import _get_free_port
from deeplake.integrations.mm.exceptions import ValidationDatasetMissingError

from mmcv.utils import build_from_cfg, digit_version  # type: ignore
from mmcv.parallel import collate  # type: ignore
import mmcv  # type: ignore
from mmcv.runner import init_dist  # type: ignore
from mmcv.runner import (  # type: ignore
    DistSamplerSeedHook,
    EpochBasedRunner,
    OptimizerHook,
    build_runner,
    get_dist_info,
    HOOKS,
)


import deeplake as dp
from deeplake.types import TypeKind
from deeplake.integrations.mm.warnings import always_warn

from deeplake.integrations.mm.mm_runners import DeeplakeIterBasedRunner
from deeplake.integrations.mm.mm_common import (
    load_ds_from_cfg,
    get_collect_keys,
    check_persistent_workers,
    find_image_tensor,
    find_smask_tensor,
    ddp_setup,
    force_cudnn_initialization,
    check_unsupported_functionalities,
    get_pipeline,
)
from deeplake.integrations.mmseg.mmseg_dataset_ import MMSegDataset, MMSegTorchDataset
from deeplake.integrations.mmseg.compose_transform_ import compose_transform

from torch.utils.data import DataLoader, IterableDataset


# Monkey-patch the function
from deeplake.integrations.mmseg.test_ import single_gpu_test as custom_single_gpu_test
from deeplake.integrations.mmseg.test_ import multi_gpu_test as custom_multi_gpu_test

import mmseg.apis

mmseg.apis.single_gpu_test = custom_single_gpu_test
mmseg.apis.multi_gpu_test = custom_multi_gpu_test


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


def mmseg_subiterable_dataset_eval(
    self,
    *args,
    **kwargs,
):
    return self.dataset.mmseg_dataset.evaluate(*args, **kwargs)


def train_segmentor(
    model,
    cfg: mmcv.ConfigDict,
    ds_train=None,
    ds_train_tensors=None,
    ds_val: Optional[dp.Dataset] = None,
    ds_val_tensors=None,
    distributed: bool = False,
    timestamp=None,
    meta=None,
    validate: bool = True,
):
    """
    Creates runner and trains evaluates the model:
    Args:
        model: model to train, should be built before passing
        cfg: mmcv.ConfigDict object containing all necessary configuration.
            In cfg we have several changes to support deeplake integration:
                _base_: still serves as a base model to inherit from
                data: where everything related to data processing, you will need to specify the following parameters:
                    train: everything related to training data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmseg keys to deeplake dataset tensor. Example:  `{"img": "images", "gt_semantic_seg": "semantic_seg"}`.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image" and "segment_mask".
                            keys that needs to be mapped are: `img` and "gt_semantic_seg". `img` and `gt_semantic_seg` are always required, if they not specified they
                            are always searched, if you specify in collect `gt_semantic_seg` then you need to either specify it in config or it will be searched based on
                            `segment_mask` htype.
                        deeplake_credentials: dictionary with deeplake credentials that allow you to access the specified data. It has following arguments: `token`.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal account on: https://www.activeloop.ai/.
                    val (Optional): everything related to validating data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmseg keys to deeplake dataset tensor. Example:  `{"img": "images", "gt_semantic_seg": "semantic_seg"}`.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image" and "segment_mask".
                            keys that needs to be mapped are: `img` and "gt_semantic_seg". `img` and `gt_semantic_seg` are always required, if they not specified they
                            are always searched, if you specify in collect `gt_semantic_seg` then you need to either specify it in config or it will be searched based on
                            `segment_mask` htype.
                        deeplake_credentials: deeplake credentials that allow you to access the specified data. It has following arguments: `token`.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal account on: https://www.activeloop.ai/.
                    test (Optional): everything related to testing data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmseg keys to deeplake dataset tensor. Example:  `{"img": "images", "gt_semantic_seg": "semantic_seg"}`.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image" and "segment_mask".
                            keys that needs to be mapped are: `img` and "gt_semantic_seg". `img` and `gt_semantic_seg` are always required, if they not specified they
                            are always searched, if you specify in collect `gt_semantic_seg` then you need to either specify it in config or it will be searched based on
                            `segment_mask` htype.
                        deeplake_credentials: deeplake credentials that allow you to access the specified data. It has following arguments: `token`.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal acccount on: https://www.activeloop.ai/.
                    samples_per_gpu: number of samples to be processed per gpu
                    workers_per_gpu: number of workers per gpu
                optimizer: dictionary containing information about optimizer initialization
                optimizer_config: some optimizer configuration that might be used during training like grad_clip etc.
                runner: training type e.g. EpochBasedRunner, here you can specify maximum number of epochs to be conducted. For instance: `runner = dict(type='EpochBasedRunner', max_epochs=273)`
        ds_train: train dataset of type dp.Dataset. This can be a view of the dataset.
        ds_train_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image" and "segment_mask".
            keys that needs to be mapped are: `img` and "gt_semantic_seg". `img` and `gt_semantic_seg` are always required, if they not specified they
            are always searched, if you specify in collect `gt_semantic_seg` then you need to either specify it in config or it will be searched based on
            `segment_mask` htype.
        ds_val: validation dataset of type dp.Dataset. This can be view of the dataset.
        ds_val_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image" and "segment_mask".
            keys that needs to be mapped are: `img` and "gt_semantic_seg". `img` and `gt_semantic_seg` are always required, if they not specified they
            are always searched, if you specify in collect `gt_semantic_seg` then you need to either specify it in config or it will be searched based on
            `segment_mask` htype.
        evaluation: dictionary that contains all information needed for evaluation apart from data processing, like how often evaluation should be done and what metrics we want to use.
            For instance, `evaluation = dict(interval=1, metric=['mIoU'])`
        distributed: bool, whether ddp training should be started, by default `False`
        timestamp: variable used in runner to make .log and .log.json filenames the same
        meta: meta data used to build runner
        validate: bool, whether validation should be conducted, by default `True`
    """
    check_unsupported_functionalities(cfg)

    if not hasattr(cfg, "gpu_ids"):
        cfg.gpu_ids = range(torch.cuda.device_count() if distributed else range(1))
    if distributed:
        return torch.multiprocessing.spawn(
            _train_segmentor,
            args=(
                model,
                cfg,
                ds_train,
                ds_train_tensors,
                ds_val,
                ds_val_tensors,
                distributed,
                timestamp,
                meta,
                validate,
                _get_free_port(),
            ),
            nprocs=len(cfg.gpu_ids),
        )
    _train_segmentor(
        0,
        model,
        cfg,
        ds_train,
        ds_train_tensors,
        ds_val,
        ds_val_tensors,
        distributed,
        timestamp,
        meta,
        validate,
    )


def register_validation_hook_(
    batch_size: int,
    num_workers: int,
    distributed: bool,
    cfg: mmcv.ConfigDict,
    ignore_index: int,
    reduce_zero_label: bool,
    train_persistent_workers: bool = False,
    ds_val: Optional[dp.Dataset] = None,
    ds_val_tensors=None,
    runner=None,
):
    eval_cfg = cfg.get("evaluation", {})
    val_dataloader_default_args = dict(
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        dist=distributed,
        shuffle=False,
        mode="val",
        seed=cfg.seed,
        num_gpus=len(cfg.gpu_ids),
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
    )

    val_dataloader_args = {
        **cfg.data.val.get("deeplake_dataloader", {}),
        **val_dataloader_default_args,
    }

    val_persistent_workers = val_dataloader_args.get("persistent_workers", False)
    check_persistent_workers(train_persistent_workers, val_persistent_workers)

    if val_dataloader_args.get("shuffle", False):
        always_warn("shuffle argument for validation dataset will be ignored.")

    if ds_val is None:
        cfg_ds_val = cfg.data.get("val")
        if not cfg_ds_val or not any(
            cfg_ds_val.get(key) is not None
            for key in ["deeplake_path", "deeplake_query"]
        ):
            raise ValidationDatasetMissingError()
        ds_val = load_ds_from_cfg(cfg.data.val)
        ds_val_tensors = cfg.data.val.get("deeplake_tensors", {})
    else:
        cfg_data = cfg.data.val.get("deeplake_path")
        if cfg_data is not None:
            always_warn(
                "A Deep Lake dataset was specified in the cfg as well as in the dataset input to train_segmentor. The dataset input to train_segmentor will be used in the workflow."
            )

    if ds_val is None:
        raise ValidationDatasetMissingError()

    if ds_val_tensors:
        val_images_tensor = ds_val_tensors["img"]
        val_masks_tensor = ds_val_tensors.get("gt_semantic_seg")
    else:
        val_images_tensor = find_image_tensor(ds_val, mm_class="img")
        val_masks_tensor = None
        collection_keys = get_collect_keys(cfg)
        if "gt_semantic_seg" in collection_keys:
            val_masks_tensor = find_smask_tensor(ds_val, mm_class="gt_semantic_seg")

    val_pipeline = get_pipeline(cfg, name="val", generic_name="test_pipeline")

    val_dataloader = build_dataloader(
        ds_val,
        val_images_tensor,
        val_masks_tensor,
        pipeline=val_pipeline,
        **val_dataloader_args,
    )

    eval_cfg["by_epoch"] = cfg.runner["type"] != "DeeplakeIterBasedRunner"
    eval_cfg["pre_eval"] = False
    eval_hook = EvalHook
    if distributed:
        eval_hook = DistEvalHook
    # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
    # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")


def _train_segmentor(
    local_rank,
    model,
    cfg: mmcv.ConfigDict,
    ds_train=None,
    ds_train_tensors=None,
    ds_val: Optional[dp.Dataset] = None,
    ds_val_tensors=None,
    distributed: bool = False,
    timestamp=None,
    meta=None,
    validate: bool = True,
    port=None,
):
    batch_size = cfg.data.get("samples_per_gpu", 256)
    num_workers = cfg.data.get("workers_per_gpu", 1)

    ignore_index = cfg.get("ignore_index", 255)
    reduce_zero_label = cfg.get("reduce_zero_label", False)

    if ds_train is None:
        ds_train = load_ds_from_cfg(cfg.data.train)
        ds_train_tensors = cfg.data.train.get("deeplake_tensors", {})
    else:
        cfg_data = cfg.data.train.get("deeplake_path")
        if cfg_data:
            always_warn(
                "A Deep Lake dataset was specified in the cfg as well as in the dataset input to train_segmentor. The dataset input to train_segmentor will be used in the workflow."
            )

    if ds_train_tensors:
        train_images_tensor = ds_train_tensors["img"]
        train_masks_tensor = ds_train_tensors.get("gt_semantic_seg")
    else:
        train_images_tensor = find_image_tensor(ds_train, mm_class="img")
        train_masks_tensor = None

        collection_keys = get_collect_keys(cfg)
        if "gt_semantic_seg" in collection_keys:
            train_masks_tensor = find_smask_tensor(ds_train, mm_class="gt_semantic_seg")

    model.CLASSES = ds_train[train_masks_tensor].metadata["class_names"]

    logger = get_root_logger(log_level=cfg.log_level)
    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        ignore_index=ignore_index,
        reduce_zero_label=reduce_zero_label,
    )

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get("train_dataloader", {}),
        **cfg.data.train.get("deeplake_dataloader", {}),
    }

    # # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # # torch.nn.parallel.DistributedDataParallel
        # model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
        #                                           device_ids=[local_rank],
        #                                           output_device=local_rank,
        #                                           broadcast_buffers=False,
        #                                           find_unused_parameters=find_unused_parameters)
        force_cudnn_initialization(cfg.gpu_ids[local_rank])
        ddp_setup(local_rank, len(cfg.gpu_ids), port)
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[cfg.gpu_ids[local_rank]],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    train_pipeline = get_pipeline(cfg, name="train", generic_name="train_pipeline")

    data_loader = build_dataloader(
        ds_train,
        train_images_tensor,
        train_masks_tensor,
        pipeline=train_pipeline,
        **train_loader_cfg,
    )

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # check runner
    cfg.custom_imports = dict(
        imports=["deeplake.integrations.mm.mm_runners"],
        allow_failed_imports=False,
    )
    if cfg.runner.type == "IterBasedRunner":
        cfg.runner.type = "DeeplakeIterBasedRunner"
    elif cfg.runner.type == "EpochBasedRunner":
        cfg.runner.type = "DeeplakeEpochBasedRunner"

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            force_cleanup=False,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        register_validation_hook_(
            batch_size=batch_size,
            num_workers=num_workers,
            distributed=distributed,
            train_persistent_workers=train_loader_cfg.get("persistent_workers", False),
            cfg=cfg,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            ds_val=ds_val,
            ds_val_tensors=ds_val_tensors,
            runner=runner,
        )

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run([data_loader], cfg.workflow)


def build_dataloader(
    dataset: dp.Dataset,
    images_tensor: str,
    masks_tensor: Optional[str],
    pipeline: List,
    mode: str = "train",
    **loader_config,
):
    persistent_workers = loader_config.get("persistent_workers", False)
    _ = loader_config.get("ignore_index")
    _ = loader_config.get("reduce_zero_label")
    dist = loader_config["dist"]
    seed = loader_config["seed"]
    transform_fn = compose_transform(
        images_tensor=images_tensor, masks_tensor=masks_tensor, pipeline=pipeline
    )

    num_workers = loader_config.get("num_workers")
    pin_memory = loader_config.get("pin_memory", False)
    if num_workers is None:
        num_workers = loader_config["workers_per_gpu"]

    shuffle = loader_config.get("shuffle", True)

    tensors_dict = {
        "images_tensor": images_tensor,
    }
    tensors = [images_tensor]
    if masks_tensor is not None:
        tensors.append(masks_tensor)
        tensors_dict["masks_tensor"] = masks_tensor

    batch_size = loader_config.get("batch_size")
    drop_last = loader_config.get("drop_last", False)
    if batch_size is None:
        batch_size = loader_config["samples_per_gpu"]

    collate_fn = partial(collate, samples_per_gpu=batch_size)

    mmseg_ds = MMSegDataset(
        dataset=dataset,
        transform=transform_fn,
        # pipeline=pipeline,
        tensors_dict=tensors_dict,
        tensors=tensors,
        mode=mode,
        num_gpus=loader_config["num_gpus"],
        batch_size=batch_size,
    )
    rank, world_size = get_dist_info()
    if dist:
        sl = get_indexes(
            dataset, rank=rank, num_replicas=world_size, drop_last=drop_last
        )
        dataset = dataset.query(
            f"select * LIMIT {sl.stop - sl.start} OFFSET {sl.start}"
        )

    pytorch_ds = MMSegTorchDataset(dataset, transform=transform_fn)
    pytorch_ds.mmseg_dataset = mmseg_ds

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    if digit_version(torch.__version__) >= digit_version("1.8.0"):
        loader = DataLoader(
            pytorch_ds,
            batch_size=batch_size,
            sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
    else:
        loader = DataLoader(
            pytorch_ds,
            batch_size=batch_size,
            sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
        )

    eval_fn = partial(mmseg_subiterable_dataset_eval, loader)
    loader.dataset.evaluate = eval_fn

    return loader
