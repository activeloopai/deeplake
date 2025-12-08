---
seo_title: "MMDetection Integration | Object Detection Pipeline"
description: "Deeplake Dataset training for Object detection using MMDetection framework."
---

# Training Object Detection Models with Deep Lake and MMDetection

This tutorial shows how to train an object detection model using MMDetection with data stored in Deep Lake. We'll use a YOLOv3 model trained on ImageNet data to demonstrate the workflow.

## Prerequisites

First, let's install the required packages:

```bash
python -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

git clone  -b dev-2.x https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python3 -m pip install -e .
```

Note: We use MMDetection 2.x versions as they're currently supported by the Deep Lake integration.

## Setup

Let's set up our imports and authentication:

<!-- test-context
```python
import numpy as np
import deeplake
from deeplake import types

ds = deeplake.create("tmp://")
```
-->

```python
import deeplake
from mmcv import Config
from mmdet.models import build_detector
import os
import mmcv

# Set your Deep Lake token
token = os.environ["ACTIVELOOP_TOKEN"]
```

## Configuration

MMDetection uses config files to define models and training parameters. Here's our YOLOv3 config with Deep Lake integration:

```python

_base_ = "<mmdetection_path>/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py"

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


data = dict(
    train=dict(
        pipeline=train_pipeline,
        deeplake_path="hub://activeloop/coco-train",
        # If not specified, Deep Lake will auto-infer the mapping, but it might make mistakes if datasets have many tensors
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},

        # the parameters in other parts of the cfg file such as samples_per_gpu, and others.
        deeplake_dataloader = {"shuffle": True, "batch_size": 4, 'num_workers': 8}
    ),

    # Parameters as the same as for train
    val=dict(
        pipeline=test_pipeline,
        deeplake_path="hub://activeloop/coco-val",
        deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"},
        deeplake_dataloader = {"shuffle": False, "batch_size": 1, 'num_workers': 8}
    ),
)


deeplake_metrics_format = "COCO"

evaluation = dict(metric=["bbox"], interval=1)

load_from = "checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth"

work_dir = "./mmdet_outputs"

log_config = dict(interval=10)

checkpoint_config = dict(interval=5000)

seed = None

device = "cuda"

runner = dict(type='EpochBasedRunner', max_epochs=10)

```

## Training

Now we can start the training:

```python
# Load config
cfg = Config.fromfile(config_path)

# Build the detector
model = build_detector(cfg.model)

# Create work directory
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

# Start training
from deeplake.integrations import mmdet as mmdet_deeplake
mmdet_deeplake.train_detector(
    model, 
    cfg,
    distributed=False,  # Set to True for multi-GPU training
    validate=False      # Set to True if you have validation data
)
```

## Key Benefits of Using Deep Lake

1. **Simple Data Loading**: Deep Lake automatically handles data streaming and batching, so you don't need to write custom data loaders.

2. **Efficient Storage**: Data is stored in an optimized format and loaded on-demand, saving disk space and memory.

3. **Easy Tensor Mapping**: The `deeplake_tensors` config maps your dataset's tensor names to what MMDetection expects, making it easy to use any dataset.

4. **Built-in Authentication**: Deep Lake handles authentication and access control for your datasets securely.

5. **Distributed Training Support**: The integration works seamlessly with MMDetection's distributed training capabilities.

## Monitoring Training

You can monitor the training progress in the work directory:

```python
# Check latest log file
log_file = os.path.join(cfg.work_dir, 'latest.log')
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        print(f.read())
```

## Inference

After training, you can use the model for inference:

```python
from mmdet.apis import inference_detector, init_detector

# Load trained model
checkpoint = os.path.join(cfg.work_dir, 'latest.pth')
model = init_detector(config_path, checkpoint)

# Load an image
img = 'path/to/test/image.jpg'

# Run inference
result = inference_detector(model, img)
```

## Common Issues and Solutions

1. If you get CUDA out of memory errors:
    - Reduce `samples_per_gpu` in the config
    - Use smaller image sizes in the pipeline

2. If training is slow:
    - Increase `num_workers` in `deeplake_dataloader`
    - Use distributed training with multiple GPUs

3. If you see authentication errors:
    - Make sure your Deep Lake token is correct
    - Check if you have access to the dataset

## Next Steps

- Try different MMDetection models by changing the base config
- Add validation data to monitor model performance
- Experiment with different data augmentations in the pipeline
- Enable distributed training for faster processing
