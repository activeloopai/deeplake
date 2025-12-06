---
seo_title: "MMSegmentation Integration | Semantic Segmentation Pipeline"
description: "Deeplake Dataset training for semantic segmentation using MMSegmentation framework."
---
# Semantic Segmentation with Deep Lake and MMSegmentation

## Integration Interface
MMSegmentation works with configs. Deeplake adopted this strategy, and in order to train MMSeg models, you need to create/specify your model and training/validation config. Deep Lake integration's logic is almost the same as MMSegmentation's with some minor modifications. The integrations with MMSeg occurs in the deeplake.integrations.mmseg module. At a high-level, Deep Lake is responsible for the pytorch dataloader that streams data to the training framework, while MMSeg is used for the training, transformation, and evaluation logic. Let us take a look at the config with deeplake changes:

Learn more about MMSegmentation [here](https://mmsegmentation.readthedocs.io/en/latest/).

### Example Configuration with Deep Lake
This tutorial shows how to train a semantic segmentation model using MMSegmentation with data stored in Deep Lake. We'll use a PSPNet model with ResNet-101 backbone trained on COCO data.

## Prerequisites

Install the required packages:

```bash
python -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.30.0
python -m pip install -e .
# Old pytorch version does not work with the new numpy versions
python -m pip install numpy==1.24.4 --force-reinstall
```

Note: We use MMSegmentation versions compatible with Deep Lake's integration.

## Setup

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")
```
-->

```python
import os
import deeplake
from mmcv import Config
import mmcv
from deeplake.integrations import mmseg as mmseg_deeplake

# Set your Deep Lake token
token = os.environ["ACTIVELOOP_TOKEN"]
```

## Configuration

Here's our PSPNet configuration with Deep Lake integration:

```python
from mmdet.apis import set_random_seed

_base_ = '<mmsegmentation_path>/configs/pspnet/pspnet_r101-d8_512x512_4x4_160k_coco-stuff164k.py'

# Normalize configuration
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

reduce_zero_label=False

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 240),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

evaluation = dict(metric=["mIoU"], interval=10000)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        deeplake_path="hub://activeloop/coco-train-seg-mask",
        deeplake_tensors = {"img": "images", "gt_semantic_seg": "seg_masks"},
        deeplake_dataloader={"shuffle": False, "num_workers": 0, "drop_last": True}
    ),
    val=dict(
        pipeline=test_pipeline,
        deeplake_path="hub://activeloop/coco-val-seg-mask"",
        deeplake_tensors = {"img": "images", "gt_semantic_seg": "seg_masks"},
        deeplake_dataloader={"shuffle": False, "batch_size": 1, "num_workers": 0, "drop_last": True}
    )
)

work_dir = "./deeplake_logs"

optimizer = dict(lr=0.02 / 8)
lr_config = dict(warmup=None)
log_config = dict(interval=50)
checkpoint_config = dict(interval=5000)

runner = dict(type="IterBasedRunner", max_iters=100000, max_epochs=None)
device = "cuda"

```

## Training

Now we can start the training:

```python

if __name__ == "__main__":
    current_loc = os.getcwd()
    cfg_file = f"{current_loc}/seg_mask_config.py"

    # Read the config file
    cfg = Config.fromfile(cfg_file)
    cfg.model.decode_head.num_classes = 81
    cfg.model.auxiliary_head.num_classes = 81

    # build segmentor
    model = mmseg_deeplake.build_segmentor(
        cfg.model
    )

    # Create work directory
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # train_segmentor
    mmseg_deeplake.train_segmentor(
        model,
        cfg,
        distributed=True,  # Set to True for multi-GPU training
        validate=True, # Set to True if you have validation data
    )
```

## Deep Lake Integration Benefits

1. **Efficient Mask Handling**: Deep Lake efficiently stores and loads segmentation masks, which can be large and memory-intensive.

2. **Automatic Format Conversion**: Deep Lake handles conversion between different mask formats (binary, RLE, polygon) automatically.

3. **Smart Batching**: Deep Lake's dataloader handles variable-sized images and masks efficiently.

4. **Memory Management**: Data is loaded on-demand, preventing out-of-memory issues with large datasets.

5. **Distributed Training Support**: Seamless integration with MMSegmentation's distributed training.

## Monitoring Training

Monitor training progress:

```python
# Check latest log file
log_file = os.path.join(cfg.work_dir, 'latest.log')
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        print(f.read())
```

## Inference

After training, use the model for inference:

```python
from mmseg.apis import inference_segmentor, init_segmentor

# Load trained model
checkpoint = os.path.join(cfg.work_dir, 'latest.pth')
model = init_segmentor(config_path, checkpoint)

# Load an image
img = 'path/to/test/image.jpg'

# Run inference
result = inference_segmentor(model, img)
```

### Key Integration Parameters

- **`data`**: Central to the MMSegmentation configuration file, it specifies the training and validation datasets, transformations, and paths.
    - **`train`**: Contains dataset path, credentials, and transformations for training data.
    - **`val`**: Contains dataset path, credentials, and transformations for validation data.
    - **`pipeline`**: A list of transformations applied to the dataset.
    - **`deeplake_path`**: Path to the Deep Lake dataset for training and validation.
    - **`deeplake_credentials`**: (Optional) Required for private, nonlocal datasets.
    - **`deeplake_tag_id`**: (Optional) Specifies a dataset commit for reproducibility.
    - **`deeplake_query`**: (Optional) Used to load datasets based on a query.
    - **`deeplake_tensors`**: Maps MMSegmentation tensors to Deep Lake tensors:
        - `"img"`: Image tensor.
        - `"gt_semantic_seg"`: Semantic segmentation tensor.

## Common Issues and Solutions

1. Memory Issues:
    - Reduce `samples_per_gpu` in config
    - Decrease image size in pipeline
    - Use smaller batch sizes

2. Performance Issues:
    - Increase `num_workers` in `deeplake_dataloader`
    - Enable distributed training
    - Use proper GPU settings

3. Mask Format Issues:
    - Verify mask format in dataset
    - Check normalization settings
    - Ensure proper padding configuration

### Custom Loss Functions

```python
# Add custom loss function to decode head
config['model']['decode_head']['loss_decode'] = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1.0,
    class_weight=[1.0] * 171  # Class weights for imbalanced datasets
)
```

### Multiple Optimization Strategies

```python
# Different learning rates for backbone and heads
config['optimizer'] = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'decode_head': dict(lr_mult=1.0),
            'auxiliary_head': dict(lr_mult=1.0)
        }
    )
)
```
