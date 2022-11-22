import deeplake as dp
import os
import sys
import pytest


def get_test_config(mmdet_path):
    from mmcv import Config
    cfg = Config.fromfile(os.path.join(mmdet_path, "configs", "yolo", "yolov3_d53_mstrain-416_273e_coco.py"))
    img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
    cfg.img_norm_cfg = img_norm_cfg
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
        ])]
    cfg.data = dict(
    # train_dataloader={"shuffle": False},
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        # deeplake_path="hub://activeloop/coco-train",
        # deeplake_credentials={
        #     "username": None,
        #     "password": None,
        #     "token": None,
        # },
        # deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}
    ),
    val=dict(
        pipeline=test_pipeline,
        # deeplake_path="hub://activeloop/coco-val",
        # deeplake_credentials={
        #     "username": None,
        #     "password": None,
        #     "token": None,
        # },
        # deeplake_tensors = {"img": "images", "gt_bboxes": "boxes", "gt_labels": "categories"}
    ),
    )
    cfg.deeplake_dataloader_type = "c++"
    cfg.deeplake_metrics_format = "COCO"
    cfg.evaluation = dict(metric=["bbox"], interval=1)
    cfg.work_dir = "./mmdet_outputs"
    cfg.log_config = dict(interval=10)
    cfg.checkpoint_config = dict(interval=12)
    cfg.seed = None
    cfg.device = "cpu"
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=10)
    return cfg


@pytest.mark.skipif(sys.platform != "linux", reason="MMDet is installed on CI only for linux")
def test_mmdet(mmdet_path):
    import mmcv
    from deeplake.integrations import mmdet
    cfg = get_test_config(mmdet_path)
    num_classes = 80
    ds_train = dp.load("hub://activeloop/coco_train")[:100]
    ds_val = dp.load("hub://activeloop-test/coco-val")[:100]
    cfg.model.bbox_head.num_classes = num_classes
    model = mmdet.build_detector(cfg.model)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    mmdet.train_detector(model, cfg, ds_train=ds_train, ds_val=ds_val)
