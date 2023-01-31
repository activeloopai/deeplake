MMDetection
==================

Deep Lake offers an integration with MMDetection, a popular open-source object detection toolbox based on PyTorch. 
The integration enables users to train models while streaming Deep Lake dataset using the transformation, training, and evaluation tools built by MMDet.

Learn more about MMDetection `here <https://mmsegmentation.readthedocs.io/en/latest/>`_.

Integration Interface
~~~~~~~~~~~~~~~~~~~~~
MMDetection works with configs. Deeplake addopted the strategy, and in order to train MMDet models, you need to create/specify your model and training/validation config. 
Deeplake integration's logic is almost the same as MMDetection's with some minor modifications. The integrations with MMDET occurs in the deeplake.integrations.mmdet module. 
At a high-level, Deep Lake is responsible for the pytorch dataloader that streams data to the training framework, while MMDET is used for the training, transformation, and evaluation logic. Let us take a look at the config with deeplake changes:

Deeplake integration requires the following parameters to be specified in the configuration file:
- data: just like in the MMDetection configuration files, in data dictionary you can specify everything that you want to be applied to the data during training and validation
    - train: is the keyword argument of data, and also a dictionary where one can specify dataset path, credentials, transformations of the training data
    - val: is the keyword argument of data, and also a dictionary where one can specify dataset path, credentials, transformations of the validation data
    - pipeline: list of transformations. Example: `pipeline =  [dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True),
        dict(type="RandomFlip", flip_ratio=0.5), dict(type="PhotoMetricDistortion")]`. This parameter exists for train as well as for val.
    - deeplake_path: path to the deeplake dataset. This parameter exists for train as well as for val.
    - deeplake_credentials: optional parameter. Required only when using private nonlocal datasets. See documendataion for deeplake.load() <https://docs.deeplake.ai/en/latest/deeplake.html#deeplake.load> for details. This parameter exists for train as well as for val.
    - deeplake_commit_id: optional parameter. If specified. the dataset will checkout to the commit. This parameter exists for train as well as for val.
    - deeplake_view_id: optional parameter. If specified the dataset will load saved view. This parameter exists for train as well as for val.
    - deeplake_tensors: optional parameter. If specified maps MMDetection tensors to the associated tensors in the dataset. MMDet tensors are: "img", "gt_bboxes", "gt_labels", "gt_masks". This parameter exists for train as well as for val.
        - "img": stands for image tensor.
        - "gt_bboxes": stands for bounding box tensor.
        - "gt_labels": stand for labels tensor.
        - "gt_masks": stand for masks tensor.
        NOTE:
            gt_masks is optional parameter and lets say you want to train poure detecter this part is going to exclude. Other mappings are mandatory
            if you don't specify them explicitly they are going to be searched in the dataset according to tensor htype. Better to specify them explicitly
    because they are not always fetched correctly
    - deeplake_dataloader: optional parameter. If specified represents the parameters of the deeplake dataloader. Deeplake dataloader parameters are: "shuffle", "batch_size", "num_workers". This parameter exists for train as well as for val.
        - "shuffle": if True shuffles the dataset.
        - "batch_size": size of batch. If not specified, dataloader will use samples_per_gpu.
        - num_workers": number of workers to use. If not specified, dataloader will use workers_per_gpu.
- deeplake_dataloader_type: optional parameter. If specified represents the type of deeplake dataloader to use.
- deeplake_metrics_format: optional parameter. If specified represents the format of the deeplake metrics that will be used during evaluation. Default COCO. Avaliable values are: "COCO", "PascalVOC". 
  If COCO format is used, you can specify whether you want to evaluate on bbox only or also want to evaluate on masks. To do that you need to specify the format of the metric in metric. 
  Ex: `deeplake_metrics_format = "COCO"
       evaluation = dict(metric=["bbox"], interval=1)`
- train_detector: Function to train the MMDetection model. Parameters are: `model, cfg: mmcv.ConfigDict, ds_train=None, ds_train_tensors=None, ds_val: Optional[dp.Dataset] = None,
    ds_val_tensors=None, distributed: bool = False, timestamp=None, meta=None, validate: bool = True,`.
    - model: MMDetection model that is going to be used.
    - cfg: Configuration of the model as well as of the datasets and transforms that's going to be used.
    - ds_train: Optional parameter. It provided will overwrite deeplake_path in train, and will pass this tensor directly to the dataloader.
    - ds_val: Optional parameter. It provided will overwrite deeplake_path in val, and will pass this tensor directly to the dataloader.
    - ds_train_tensors: Optional parameter. It provided will overwrite deeplake_tensors in train, and will pass this tensor mapping directly to dataloader.
    - ds_val_tensors: Optional parameter. It provided will overwrite deeplake_tensors in val, and will pass this tensor mapping directly to dataloader.
    - distributed: Optional parameter. If provided will run the code on all available gpus. Meta data used to build runner
    - timestamp: variable used in runner to make .log and .log.json filenames the same.'
    - validate: bool, whether validation should be conducted, by default `True`

Below is the example of the deeplake mmdet configuration:

`
_base_ = "../mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py"

# use caffe img_norm
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(type='Resize', img_scale=[(320, 320), (416, 416)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
#--------------------------------------DEEPLAKE INPUTS------------------------------------------------------------#
TOKEN = "INSERT_YOUR_DEEPLAKE_TOKEN"
​

data = dict(
    # samples_per_gpu=4, # Is used instead of batch_size if deeplake_dataloader is not specified below
    # workers_per_gpu=8, # Is used instead of num_workers if deeplake_dataloader is not specified below
    train=dict(
        pipeline=train_pipeline,
​
        # Credentials for authentication. See documendataion for deeplake.load() for details
        deeplake_path="hub://activeloop/coco-train",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        #OPTIONAL - Checkout teh specified commit_id before training
        deeplake_commit_id="",
        #OPTIONAL - Loads a dataset view for training based on view_id
        deeplake_view_id="",
​
        # OPTIONAL - {"mmdet_key": "deep_lake_tensor",...} - Maps Deep Lake tensors to MMDET dictionary keys. 
        # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories", "gt_masks": "masks},
        
        # OPTIONAL - Parameters to use for the Deep Lake dataloader. If unspecified, the integration uses
        # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
        deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
    ),
​
    # Parameters as the same as for train
    val=dict(
        pipeline=test_pipeline,
        deeplake_path="hub://activeloop/coco-val",
        deeplake_credentials={
            "username": None,
            "password": None,
            "token": TOKEN,
            "creds": None,
        },
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
    ),
)
​
# Which dataloader to use
deeplake_dataloader_type = "c++"  # "c++" is available to enterprise users. Otherwise use "python"
​
# Which metrics to use for evaulation. In MMDET (without Deeplake), this is inferred from the dataset type.
# In the Deep Lake integration, since the format is standardized, a variety of metrics can be used for a given dataset.
deeplake_metrics_format = "COCO"
​
#----------------------------------END DEEPLAKE INPUTS------------------------------------------------------------#`

And config for training:
`
import os
from mmcv import Config
import mmcv
from deeplake.integrations import mmdet as mmdet_deeplake


cfg = Config.fromfile(cfg_file)

cfg.model.bbox_head.num_classes = num_classes

# Build the detector
model = mmdet_deeplake.build_detector(cfg.model)

# Create work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

# Run the training
mmdet_deeplake.train_detector(model, cfg, distributed=args.distributed, validate=args.validate)`
