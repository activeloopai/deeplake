import sys
import deeplake as dp
import os
import pytest
import pickle
import mmcv  # type: ignore
import numpy as np
import pathlib


_THIS_FILE = pathlib.Path(__file__).parent.absolute()


def get_path(path):
    return os.path.join(_THIS_FILE, f"indexes/{path}")


def load_pickle_file(pickle_file):
    with open(get_path(pickle_file), "rb") as f:
        return pickle.load(f)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_check_unused_dataset_fields():
    from deeplake.integrations.mmdet import mmdet_utils

    cfg = mmcv.utils.config.ConfigDict()
    cfg.dataset_type = "COCODataset"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)

    cfg.data_root = "./"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_check_unsupported_train_pipeline_fields():
    from deeplake.integrations.mmdet import mmdet_utils

    cfg = mmcv.utils.config.ConfigDict()
    cfg.train_pipeline = [dict(type="LoadImageFromFile")]

    with pytest.warns(UserWarning):
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="LoadAnnotations")]

    with pytest.warns(UserWarning):
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="Corrupt")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="MinIoURandomCrop")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="RandomCrop")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="YOLOXHSVRandomAug")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="CopyPaste")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="CutOut")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.train_pipeline = [dict(type="Mosaic")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_check_dataset_augmentation_formats():
    from deeplake.integrations.mmdet import mmdet_utils

    cfg = mmcv.utils.config.ConfigDict()
    cfg.train_dataset = dict(type="ConcatDataset")

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_coco_to_pascal_format():
    from deeplake.integrations.mmdet import mmdet_

    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"coords": {"mode": "LTWH", "type": "pixel"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 6, 7]])
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)

    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"coords": {"mode": "LTWH", "type": "fractional"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_yolo_to_pascal_format():
    from deeplake.integrations.mmdet import mmdet_

    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"coords": {"mode": "CCWH", "type": "pixel"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[3, 4, 5, 6]])
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"coords": {"mode": "CCWH", "type": "fractional"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_pascal_to_pascal_format():
    from deeplake.integrations.mmdet import mmdet_

    shape = (10, 10)
    bbox = np.array([[4, 5, 6, 7]])
    bbox_info = {"coords": {"mode": "LTRB", "type": "pixel"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 6, 7]])
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.6, 0.7]])
    bbox_info = {"coords": {"mode": "LTRB", "type": "fractional"}}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    np.testing.assert_array_equal(bbox_pascal, targ_bbox)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_pascal_to_coco_format():
    from deeplake.integrations.mmdet import mmdet_

    images = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 6, 7]]])
    bbox_info = ("LTRB", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, images)
    targ_bbox = np.array([[4, 5, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.6, 0.7]]])
    bbox_info = ("LTRB", "fractional")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, images)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_yolo_to_coco_format():
    from deeplake.integrations.mmdet import mmdet_

    image = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 2, 2]]])
    bbox_info = ("CCWH", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    targ_bbox = np.array([[3, 4, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.2, 0.2]]])
    bbox_info = ("CCWH", "fractional")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_coco_to_coco_format():
    from deeplake.integrations.mmdet import mmdet_

    image = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 2, 2]]])
    bbox_info = ("LTWH", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    targ_bbox = np.array([[4, 5, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.2, 0.2]]])
    bbox_info = ("LTWH", "fractional")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)


def get_test_config(mmdet_path):
    from mmcv import Config

    cfg = Config.fromfile(
        os.path.join(
            mmdet_path, "configs", "yolo", "yolov3_d53_mstrain-416_273e_coco.py"
        )
    )
    img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
    cfg.img_norm_cfg = img_norm_cfg
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            type="Expand",
            mean=img_norm_cfg["mean"],
            to_rgb=img_norm_cfg["to_rgb"],
            ratio_range=(1, 2),
        ),
        dict(type="Resize", img_scale=[(320, 320), (416, 416)], keep_ratio=True),
        dict(type="RandomFlip", flip_ratio=0.0),
        dict(type="PhotoMetricDistortion"),
        dict(type="Normalize", **img_norm_cfg),
        dict(type="Pad", size_divisor=32),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(416, 416),
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip", flip_ratio=0.0),
                dict(type="Normalize", **img_norm_cfg),
                dict(type="Pad", size_divisor=32),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img"]),
            ],
        ),
    ]
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
    cfg.runner = dict(type="EpochBasedRunner", max_epochs=10)
    return cfg


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
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
