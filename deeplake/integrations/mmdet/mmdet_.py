"""
Deep Lake offers an integration with MMDetection, a popular open-source object detection toolbox based on PyTorch. 
The integration enables users to train models while streaming Deep Lake dataset using the transformation, training, and evaluation tools built by MMDet.

Learn more about MMDetection `here <https://mmsegmentation.readthedocs.io/en/latest/>`_.

Integration Interface
~~~~~~~~~~~~~~~~~~~~~
MMDetection works with configs. Deeplake adopted the strategy, and in order to train MMDet models, you need to create/specify your model and training/validation config. 
Deeplake integration's logic is almost the same as MMDetection's with some minor modifications. The integrations with MMDET occurs in the deeplake.integrations.mmdet module. 
At a high-level, Deep Lake is responsible for the pytorch dataloader that streams data to the training framework, while MMDET is used for the training, transformation, and evaluation logic. Let us take a look at the config with deeplake changes:

Deeplake integration requires the following parameters to be specified in the configuration file:
- data: just like in the MMDetection configuration files, in data dictionary you can specify everything that you want to be applied to the data during training and validation
    - train: is the keyword argument of data, and also a dictionary where one can specify dataset path, credentials, transformations of the training data
    - val: is the keyword argument of data, and also a dictionary where one can specify dataset path, credentials, transformations of the validation data
    - pipeline: list of transformations. This parameter exists for train as well as for val.
    Example:
    
>>>    pipeline =  [dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True), dict(type="RandomFlip", flip_ratio=0.5), dict(type="PhotoMetricDistortion")]

    - deeplake_path: path to the deeplake dataset. This parameter exists for train as well as for val.
    - deeplake_credentials: optional parameter. Required only when using private nonlocal datasets. See documendataion for `deeplake.load() <https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.load>`_ for details. This parameter exists for train as well as for val.
    - deeplake_commit_id: optional parameter. If specified, the dataset will checkout to the commit. This parameter exists for train as well as for val. See documentation for `Dataset.commit_id <https://deep-lake--2152.org.readthedocs.build/en/2152/deeplake.core.dataset.html#deeplake.core.dataset.Dataset.commit_id>`_
    - deeplake_view_id: optional parameter. If specified the dataset will load saved view. This parameter exists for train as well as for val.
    - deeplake_tensors: optional parameter. If specified maps MMDetection tensors to the associated tensors in the dataset. MMDet tensors are: "img", "gt_bboxes", "gt_labels", "gt_masks". This parameter exists for train as well as for val.
        - "img": stands for image tensor.
        - "gt_bboxes": stands for bounding box tensor.
        - "gt_labels": stand for labels tensor.
        - "gt_masks": stand for masks tensor.
    - deeplake_dataloader: optional parameter. If specified represents the parameters of the deeplake dataloader. Deeplake dataloader parameters are: "shuffle", "batch_size", "num_workers". This parameter exists for train as well as for val.
        - "shuffle": if True shuffles the dataset.
        - "batch_size": size of batch. If not specified, dataloader will use samples_per_gpu.
        - "num_workers": number of workers to use. If not specified, dataloader will use workers_per_gpu.
- deeplake_dataloader_type: optional parameter. If specified represents the type of deeplake dataloader to use.
- deeplake_metrics_format: optional parameter. If specified represents the format of the deeplake metrics that will be used during evaluation. Default COCO. Avaliable values are: "COCO", "PascalVOC". 
  If COCO format is used, you can specify whether you want to evaluate on bbox only or also want to evaluate on masks. To do that you need to specify the format of the metric in metric. 
  
Ex:

>>>  deeplake_metrics_format = "COCO"
>>>  evaluation = dict(metric=["bbox"], interval=1)

- train_detector: Function to train the MMDetection model. Parameters are: model, cfg: mmcv.ConfigDict, ds_train=None, ds_train_tensors=None, ds_val: Optional[dp.Dataset] = None, ds_val_tensors=None, distributed: bool = False, timestamp=None, meta=None, validate: bool = True.
    - model: MMDetection model that is going to be used.
    - cfg: Configuration of the model as well as of the datasets and transforms that's going to be used.
    - ds_train: Optional parameter. If provided will overwrite deeplake_path in train, and will pass this tensor directly to the dataloader.
    - ds_val: Optional parameter. If provided will overwrite deeplake_path in val, and will pass this tensor directly to the dataloader.
    - ds_train_tensors: Optional parameter. If provided will overwrite deeplake_tensors in train, and will pass this tensor mapping directly to dataloader.
    - ds_val_tensors: Optional parameter. If provided will overwrite deeplake_tensors in val, and will pass this tensor mapping directly to dataloader.
    - distributed: Optional parameter. If provided will run the code on all available gpus. Meta data used to build runner
    - timestamp: variable used in runner to make .log and .log.json filenames the same.'
    - validate: bool, whether validation should be conducted, by default True

NOTE:
    gt_masks is optional parameter and lets say you want to train pure detecter this part is going to exclude. Other mappings are mandatory
    if you don't specify them explicitly they are going to be searched in the dataset according to tensor htype. Better to specify them explicitly

MMDetection Config Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Below is the example of the deeplake mmdet configuration:


>>> _base_ = "../mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py"
>>> 
>>> # use caffe img_norm
>>> img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
>>> 
>>> train_pipeline = [
>>>     dict(type='LoadImageFromFile'),
>>>     dict(type='LoadAnnotations', with_bbox=True),
>>>     dict(
>>>         type='Expand',
>>>         mean=img_norm_cfg['mean'],
>>>         to_rgb=img_norm_cfg['to_rgb'],
>>>         ratio_range=(1, 2)),
>>>     dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
>>>     dict(type='RandomFlip', flip_ratio=0.0),
>>>     dict(type='PhotoMetricDistortion'),
>>>     dict(type='Normalize', **img_norm_cfg),
>>>     dict(type='Pad', size_divisor=32),
>>>     dict(type='DefaultFormatBundle'),
>>>     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
>>> ]
>>> test_pipeline = [
>>>     dict(type='LoadImageFromFile'),
>>>     dict(
>>>         type='MultiScaleFlipAug',
>>>         img_scale=(416, 416),
>>>         flip=False,
>>>         transforms=[
>>>             dict(type='Resize', keep_ratio=True),
>>>             dict(type='RandomFlip', flip_ratio=0.0),
>>>             dict(type='Normalize', **img_norm_cfg),
>>>             dict(type='Pad', size_divisor=32),
>>>             dict(type='ImageToTensor', keys=['img']),
>>>             dict(type='Collect', keys=['img'])
>>>         ])
>>> ]
>>> #--------------------------------------DEEPLAKE INPUTS------------------------------------------------------------#
>>> TOKEN = "INSERT_YOUR_DEEPLAKE_TOKEN"
​>>> 
>>> 
>>> data = dict(
>>>     # samples_per_gpu=4, # Is used instead of batch_size if deeplake_dataloader is not specified below
>>>     # workers_per_gpu=8, # Is used instead of num_workers if deeplake_dataloader is not specified below
>>>     train=dict(
>>>         pipeline=train_pipeline,
​>>> 
>>>         # Credentials for authentication. See documendataion for deeplake.load() for details
>>>         deeplake_path="hub://activeloop/coco-train",
>>>         deeplake_credentials={
>>>             "username": None,
>>>             "password": None,
>>>             "token": TOKEN,
>>>             "creds": None,
>>>         },
>>>         #OPTIONAL - Checkout the specified commit_id before training
>>>         deeplake_commit_id="",
>>>         #OPTIONAL - Loads a dataset view for training based on view_id
>>>         deeplake_view_id="",
​
>>>         # OPTIONAL - {"mmdet_key": "deep_lake_tensor",...} - Maps Deep Lake tensors to MMDET dictionary keys. 
>>>         # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
>>>         deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories", "gt_masks": "masks},
>>>         
>>>         # OPTIONAL - Parameters to use for the Deep Lake dataloader. If unspecified, the integration uses
>>>         # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
>>>         deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
>>>     ),
>>> ​
>>>     # Parameters as the same as for train
>>>     val=dict(
>>>         pipeline=test_pipeline,
>>>         deeplake_path="hub://activeloop/coco-val",
>>>         deeplake_credentials={
>>>             "username": None,
>>>             "password": None,
>>>             "token": TOKEN,
>>>             "creds": None,
>>>         },
>>>         deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
>>>         deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
>>>     ),
>>> )
​>>> 
>>> # Which dataloader to use
>>> deeplake_dataloader_type = "c++"  # "c++" is available to enterprise users. Otherwise use "python"
​
>>> # Which metrics to use for evaulation. In MMDET (without Deeplake), this is inferred from the dataset type.
>>> # In the Deep Lake integration, since the format is standardized, a variety of metrics can be used for a given dataset.
>>> deeplake_metrics_format = "COCO"
​
>>> #----------------------------------END DEEPLAKE INPUTS------------------------------------------------------------#

And config for training:

>>> import os
>>> from mmcv import Config
>>> import mmcv
>>> from deeplake.integrations import mmdet as mmdet_deeplake
>>> 
>>> 
>>> cfg = Config.fromfile(cfg_file)
>>> 
>>> cfg.model.bbox_head.num_classes = num_classes
>>> 
>>> # Build the detector
>>> model = mmdet_deeplake.build_detector(cfg.model)
>>> 
>>> # Create work_dir
>>> mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
>>> 
>>> # Run the training
>>> mmdet_deeplake.train_detector(model, cfg, distributed=args.distributed, validate=args.validate)
"""


from typing import Optional

from deeplake.core.ipc import _get_free_port
import deeplake as dp
from deeplake.util.bugout_reporter import deeplake_reporter
import mmcv  # type: ignore
import torch
from mmdet.utils.util_distribution import *  # type: ignore
from deeplake.integrations.mmdet import mmdet_utils

from utils import unsupported_functionalities
from trainer import trainer

@deeplake_reporter.record_call
def train_detector(
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
        train_dataset: dataset to train of type dp.Dataset
        cfg: mmcv.ConfigDict object containing all necessary configuration.
            In cfg we have several changes to support deeplake integration:
                _base_: still serbes as a base model to inherit from
                data: where everything related to dataprocessing, you will need to specify the following parameters:
                    train: everything related to training data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  `{"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}`.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image", "class_label, "bbox", "segment_mask" or "polygon".
                            keys that needs to be mapped are: `img`, `gt_labels`, `gt_bboxes`, `gt_masks`. `img`, `gt_labels`, `gt_bboxes` are always required, they if not specified they
                            are always searched, while masks are optional, if you specify in collect `gt_masks` then you need to either specify it in config or it will be searched based on
                            `segment_mask` and `polygon` htypes.
                        deeplake_credentials: dictionary with deeplake credentials that allow you to acess the specified data. It has following arguments: `username`, `password`, `token`.
                            `username` and `password` are your CLI credentials, if not specified public read and write access will be granted.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal acccount on: https://www.activeloop.ai/.
                            if both `username`, `password` and `token` are specified, token's read write access will be granted.
                    val (Optional): everything related to validating data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image", "class_label, "bbox", "segment_mask" or "polygon".
                            keys that needs to be mapped are: `img`, `gt_labels`, `gt_bboxes`, `gt_masks`. `img`, `gt_labels`, `gt_bboxes` are always required, they if not specified they
                            are always searched, while masks are optional, if you specify in collect `gt_masks` then you need to either specify it in config or it will be searched based on
                            `segment_mask` and `polygon` htypes.
                        deeplake_credentials: deeplake credentials that allow you to acess the specified data. It has following arguments: `username`, `password`, `token`.
                            `username` and `password` are your CLI credentials, if not specified public read and write access will be granted.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal acccount on: https://www.activeloop.ai/.
                            if both `username`, `password` and `token` are specified, token's read write access will be granted.
                    test (Optional): everything related to testing data, it has the following attributes:
                        pipeline: dictionary where all training augmentations and transformations should be specified, like in mmdet
                        deeplake_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
                            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image", "class_label, "bbox", "segment_mask" or "polygon".
                            keys that needs to be mapped are: `img`, `gt_labels`, `gt_bboxes`, `gt_masks`. `img`, `gt_labels`, `gt_bboxes` are always required, they if not specified they
                            are always searched, while masks are optional, if you specify in collect `gt_masks` then you need to either specify it in config or it will be searched based on
                            `segment_mask` and `polygon` htypes.
                        deeplake_credentials: deeplake credentials that allow you to acess the specified data. It has following arguments: `username`, `password`, `token`.
                            `username` and `password` are your CLI credentials, if not specified public read and write access will be granted.
                            `token` is the token that gives you read or write access to the datasets. It is available in your personal acccount on: https://www.activeloop.ai/.
                            if both `username`, `password` and `token` are specified, token's read write access will be granted.
                    samples_per_gpu: number of samples to be processed per gpu
                    workers_per_gpu: number of workers per gpu
                optimizer: dictionary containing information about optimizer initialization
                optimizer_config: some optimizer configuration that might be used during training like grad_clip etc.
                runner: training type e.g. EpochBasedRunner, here you can specify maximum number of epcohs to be conducted. For instance: `runner = dict(type='EpochBasedRunner', max_epochs=273)`
        ds_train: train dataset of type dp.Dataset. This can be a view of the dataset.
        ds_train_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image", "class_label, "bbox", "segment_mask" or "polygon".
            keys that needs to be mapped are: `img`, `gt_labels`, `gt_bboxes`, `gt_masks`. `img`, `gt_labels`, `gt_bboxes` are always required, they if not specified they
            are always searched, while masks are optional, if you specify in collect `gt_masks` then you need to either specify it in config or it will be searched based on
            `segment_mask` and `polygon` htypes.
        ds_val: validation dataset of type dp.Dataset. This can be view of the dataset.
        ds_val_tensors: dictionary that maps mmdet keys to deeplake dataset tensor. Example:  {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}.
            If this dictionary is not specified, these tensors will be searched automatically using htypes like "image", "class_label, "bbox", "segment_mask" or "polygon".
            keys that needs to be mapped are: `img`, `gt_labels`, `gt_bboxes`, `gt_masks`. `img`, `gt_labels`, `gt_bboxes` are always required, they if not specified they
            are always searched, while masks are optional, if you specify in collect `gt_masks` then you need to either specify it in config or it will be searched based on
            `segment_mask` and `polygon` htypes.
        runner: dict(type='EpochBasedRunner', max_epochs=273)
        evaluation: dictionary that contains all information needed for evaluation apart from data processing, like how often evaluation should be done and what metrics we want to use. In deeplake
            integration version you also need to specify what kind of output you want to be printed during evalaution. For instance, `evaluation = dict(interval=1, metric=['bbox'], metrics_format="COCO")`
        distributed: bool, whether ddp training should be started, by default `False`
        timestamp: variable used in runner to make .log and .log.json filenames the same
        meta: meta data used to build runner
        validate: bool, whether validation should be conducted, by default `True`
    """
    unsupported_functionalities.check_unsupported_functionalities(cfg)

    if not hasattr(cfg, "gpu_ids"):
        cfg.gpu_ids = range(torch.cuda.device_count() if distributed else 1)
    if distributed:
        return torch.multiprocessing.spawn(
            trainer._train_detector,
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
    trainer._train_detector(
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
