import sys
import os
import pickle
import pathlib

import pytest
import numpy as np
import wandb

import deeplake as dp
from deeplake.client.client import DeepLakeBackendClient


os.system("wandb offline")

_THIS_FILE = pathlib.Path(__file__).parent.absolute()
_COCO_PATH = "hub://activeloop/coco-train"
_BALLOON_PATH = "hub://adilkhan/balloon-train"
_MMDET_KEYS = ["img", "gt_bboxes", "gt_labels", "gt_masks"]
_COCO_KEYS = ["images", "boxes", "categories", "masks"]
_BALLOON_KEYS = ["images", "bounding_boxes", "labels", "segmentation_polygons"]
_OBJECT_DETECTION = ["yolo"]
_INSTANCE_SEGMENTATION = ["mask_rcnn"]


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
    import mmcv  # type: ignore
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
    import mmcv  # type: ignore
    from deeplake.integrations.mmdet import mmdet_utils

    cfg = mmcv.utils.config.ConfigDict()
    cfg.data = mmcv.utils.config.ConfigDict()
    cfg.data.train = mmcv.utils.config.ConfigDict()
    cfg.data.train.pipeline = [dict(type="LoadImageFromFile")]

    with pytest.warns(UserWarning):
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.data.train.pipeline = [dict(type="LoadAnnotations")]

    with pytest.warns(UserWarning):
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.data.train.pipeline = [dict(type="Corrupt")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.data.train.pipeline = [dict(type="CopyPaste")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.data.train.pipeline = [dict(type="CutOut")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)

    cfg.data.train.pipeline = [dict(type="Mosaic")]

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
def test_check_dataset_augmentation_formats():
    import mmcv  # type: ignore
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

    bbox = np.empty(0)
    targ_bbox = np.empty((0, 4))
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

    bbox = np.empty(0)
    targ_bbox = np.empty((0, 4))
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

    bbox = np.empty(0)
    targ_bbox = []
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, images)
    np.testing.assert_array_equal(bbox_coco, targ_bbox)


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

    bbox = np.empty(0)
    targ_bbox = []
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    np.testing.assert_array_equal(bbox_coco, targ_bbox)


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


DATASET_PATH_TO_TENSOR_KEYS = {
    _COCO_PATH: _COCO_KEYS,
    _BALLOON_PATH: _BALLOON_KEYS,
}


def get_deeplake_tensors(dataset_path, model):
    if dataset_path not in DATASET_PATH_TO_TENSOR_KEYS:
        raise ValueError(f"{dataset_path} is not in DATASET_PATH_TO_TENSOR_KEYS")

    tensor_keys = DATASET_PATH_TO_TENSOR_KEYS[dataset_path]
    tensors_dict = {}

    for mmdet_key, tensor_key in zip(_MMDET_KEYS, tensor_keys):
        if model in _OBJECT_DETECTION and mmdet_key == "gt_masks":
            continue
        tensors_dict[mmdet_key] = tensor_key
    return tensors_dict


def get_test_config(
    mmdet_path,
    model_name,
    dataset_path,
):
    from mmcv import Config

    deeplake_tensors = get_deeplake_tensors(dataset_path, model_name)

    if model_name == "mask_rcnn":
        model_path = os.path.join(
            "mask_rcnn",
            "mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py",
        )

    elif model_name == "yolo":
        model_path = os.path.join(
            "yolo",
            "yolov3_d53_mstrain-608_273e_coco.py",
        )

    cfg = Config.fromfile(
        os.path.join(
            mmdet_path,
            "configs",
            model_path,
        )
    )

    cfg.data = dict(
        train_dataloader={"shuffle": False},
        samples_per_gpu=1,
        workers_per_gpu=0,
        train=dict(
            pipeline=cfg.train_pipeline,
            deeplake_tensors=deeplake_tensors,
        ),
        val=dict(
            pipeline=cfg.test_pipeline,
            deeplake_tensors=deeplake_tensors,
        ),
        test=dict(
            pipeline=cfg.test_pipeline,
        ),
    )
    cfg.deeplake_dataloader_type = "python"
    cfg.deeplake_metrics_format = "COCO"
    cfg.evaluation = dict(metric=["bbox"], interval=1)
    cfg.work_dir = "./mmdet_outputs"
    cfg.log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={"project": "mmdetection"},
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3,
        ),
    ]
    cfg.checkpoint_config = dict(interval=12)
    cfg.seed = None
    cfg.device = "cpu"
    cfg.runner = dict(type="EpochBasedRunner", max_epochs=1)
    return cfg


@pytest.mark.skipif(
    sys.platform != "linux" or sys.version_info < (3, 7),
    reason="MMDet is installed on CI only for linux and python version >= 3.7.",
)
@pytest.mark.parametrize(
    "model_name",
    [
        "mask_rcnn",
        "yolo",
    ],
)
@pytest.mark.parametrize(
    "dataset_path",
    [
        # "hub://activeloop/coco-train",
        "hub://adilkhan/balloon-train",
    ],
)
@pytest.mark.parametrize(
    "tensors_specified",
    [
        "True",
        "False",
    ],
)
def test_mmdet(mmdet_path, model_name, dataset_path, tensors_specified):
    import mmcv
    from deeplake.integrations import mmdet

    deeplake_tensors = None
    if tensors_specified:
        deeplake_tensors = get_deeplake_tensors(dataset_path, model_name)
    cfg = get_test_config(mmdet_path, model_name=model_name, dataset_path=dataset_path)
    cfg = process_cfg(cfg, model_name, dataset_path)
    ds_train = dp.load(dataset_path)[:2]
    ds_val = dp.load(dataset_path)[:2]
    if dataset_path == "hub://adilkhan/balloon-train":
        ds_train_with_none = dp.empty("ds_train", overwrite=True)
        ds_val_with_none = dp.empty("ds_val", overwrite=True)

        ds_train_with_none.create_tensor_like("images", ds_train.images)
        ds_train_with_none.create_tensor_like("bounding_boxes", ds_train.bounding_boxes)
        ds_train_with_none.create_tensor_like(
            "segmentation_polygons", ds_train.segmentation_polygons
        )
        ds_train_with_none.create_tensor_like("labels", ds_train.labels)

        ds_val_with_none.create_tensor_like("images", ds_val.images)
        ds_val_with_none.create_tensor_like("bounding_boxes", ds_val.bounding_boxes)
        ds_val_with_none.create_tensor_like(
            "segmentation_polygons", ds_val.segmentation_polygons
        )
        ds_val_with_none.create_tensor_like("labels", ds_val.labels)

        ds_train_with_none.append(ds_train[0])
        ds_train_with_none.images.append(ds_train.images[1])
        ds_train_with_none.bounding_boxes.append(None)
        ds_train_with_none.segmentation_polygons.append(None)
        ds_train_with_none.labels.append(None)

        ds_val_with_none.append(ds_val[0])
        ds_val_with_none.images.append(ds_val.images[1])
        ds_val_with_none.bounding_boxes.append(None)
        ds_val_with_none.segmentation_polygons.append(None)
        ds_val_with_none.labels.append(None)

        ds_train = ds_train_with_none
        ds_val = ds_val_with_none

    model = mmdet.build_detector(cfg.model)
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    mmdet.train_detector(
        model,
        cfg,
        ds_train=ds_train,
        ds_train_tensors=deeplake_tensors,
        ds_val=ds_val,
        ds_val_tensors=deeplake_tensors,
    )


def process_cfg(cfg, model_name, dataset_path):
    if dataset_path == _BALLOON_PATH:
        if model_name in _INSTANCE_SEGMENTATION:
            cfg.model.roi_head.bbox_head.num_classes = 1
            cfg.model.roi_head.mask_head.num_classes = 1
        elif model_name in _OBJECT_DETECTION:
            cfg.model.bbox_head.num_classes = 1
    return cfg
