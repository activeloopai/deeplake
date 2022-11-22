import deeplake as dp
from deeplake.integrations.mmdet import mmdet_, mmdet_utils

import os
import pytest
import pickle
import mmcv
import numpy as np
import pathlib


_THIS_FILE = pathlib.Path(__file__).parent.absolute()


def get_path(path):
    return os.path.join(_THIS_FILE, f"indexes/{path}")


def load_pickle_file(pickle_file):
    with open(get_path(pickle_file), "rb") as f:
        return pickle.load(f)


def test_check_unused_dataset_fields():
    cfg = mmcv.utils.config.ConfigDict()
    cfg.dataset_type = "COCODataset"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)

    cfg.data_root = "./"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)


def test_check_unsupported_train_pipeline_fields():
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


def test_check_dataset_augmentation_formats():
    cfg = mmcv.utils.config.ConfigDict()
    cfg.train_dataset = dict(type="ConcatDataset")

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)


def test_coco_to_pascal_format():
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


def test_yolo_to_pascal_format():
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


def test_pascal_to_pascal_format():
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


def test_pascal_to_coco_format():
    images = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 6, 7]]])
    bbox_info = ("LTRB", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, images)
    targ_bbox = np.array([[4, 5, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.6, 0.7]]])
    bbox_info = ("LTRB", "fraction")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, images)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)


def test_yolo_to_coco_format():
    image = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 2, 2]]])
    bbox_info = ("CCWH", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    targ_bbox = np.array([[3, 4, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.2, 0.2]]])
    bbox_info = ("CCWH", "fraction")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)


def test_coco_to_coco_format():
    image = [np.zeros((10, 10))]
    bbox = np.array([[[4, 5, 2, 2]]])
    bbox_info = ("LTWH", "pixel")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    targ_bbox = np.array([[4, 5, 2, 2]])
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)

    bbox = np.array([[[0.4, 0.5, 0.2, 0.2]]])
    bbox_info = ("LTWH", "fraction")
    bbox_coco = mmdet_.convert_to_coco_format(bbox, bbox_info, image)
    np.testing.assert_array_equal(bbox_coco[0], targ_bbox)
