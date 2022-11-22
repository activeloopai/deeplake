import deeplake as dp
from deeplake.integrations.mmdet import mmdet_, mmdet_utils

import pytest
import pickle
import mmcv
import numpy as np


def test_COCO_classes():
    hub_dataset = dp.load("hub://activeloop/coco-val")[:10]
    masks = []
    bboxes = hub_dataset.boxes.numpy(aslist=True)
    imgs = hub_dataset.images
    labels = hub_dataset.labels.numpy(aslist=True)
    class_names = hub_dataset.labels.info.class_names
    iscrowds = hub_dataset.iscrowd.numpy(aslist=True)
    COCO = mmdet_utils._COCO(
        hub_dataset=hub_dataset,
        imgs=imgs,
        masks=masks,
        bboxes=bboxes,
        labels=labels,
        iscrowds=iscrowds,
        class_names=class_names,
        bbox_format=("LTWH", "pixel"),
    )

    COCO.createHubIndex()

    cats = COCO.cats
    imgs = COCO.imgs
    anns = COCO.anns
    imgToAnns = COCO.imgToAnns
    catToImgs = COCO.catToImgs

    targ_cats = pickle.load("indexes/cats.pkl")
    targ_anns = pickle.load("indexes/anns.pkl")
    targ_imgs = pickle.load("indexes/imgs.pkl")
    targ_imgToAnns = pickle.load("indexes/imgToAnns.pkl")
    targ_catToImgs = pickle.load("indexes/catToImgs.pkl")
    targ_res = pickle.load("indexes/res.pkl")

    assert cats == targ_cats
    assert imgs == targ_imgs
    assert anns == targ_anns
    assert imgToAnns == targ_imgToAnns
    assert catToImgs == targ_catToImgs

    resFile = pickle.load("indexes/resFile.pkl")
    res = COCO.loadRes(resFile)
    assert res == targ_res


def test_HubCOCO():
    hub_dataset = dp.load("hub://activeloop/coco-val")[:10]
    masks = []
    bboxes = hub_dataset.boxes.numpy(aslist=True)
    imgs = hub_dataset.images
    labels = hub_dataset.labels.numpy(aslist=True)
    class_names = hub_dataset.labels.info.class_names
    iscrowds = hub_dataset.iscrowd.numpy(aslist=True)

    HubCOCO = mmdet_utils.HubCOCO(
        hub_dataset=hub_dataset,
        imgs=imgs,
        masks=masks,
        bboxes=bboxes,
        labels=labels,
        iscrowds=iscrowds,
        class_names=class_names,
        bbox_format=("LTWH", "pixel"),
    )

    ann_ids = HubCOCO.get_ann_ids()
    cat_ids = HubCOCO.get_cat_ids()
    img_ids = HubCOCO.get_img_ids()
    anns = HubCOCO.load_anns()
    cats = HubCOCO.load_cats()
    imgs = HubCOCO.load_imgs()

    targ_cats = pickle.load("indexes/cats.pkl")
    targ_anns = pickle.load("indexes/anns.pkl")
    targ_imgs = pickle.load("indexes/imgs.pkl")
    targ_ann_ids = pickle.load("indexes/ann_ids.pkl")
    targ_cat_ids = pickle.load("indexes/cat_ids.pkl")
    targ_img_ids = pickle.load("indexes/img_ids.pkl")

    assert imgs == targ_imgs
    assert ann_ids == targ_ann_ids
    assert cat_ids == targ_cat_ids
    assert anns == targ_anns
    assert cats == targ_cats
    assert img_ids == targ_img_ids


def test_COCODatasetEvaluater():
    hub_dataset = dp.load("hub://activeloop/coco-val")[:10]
    masks = []
    bboxes = hub_dataset.boxes.numpy(aslist=True)
    imgs = hub_dataset.images
    labels = hub_dataset.labels.numpy(aslist=True)
    class_names = hub_dataset.labels.info.class_names
    iscrowds = hub_dataset.iscrowd.numpy(aslist=True)

    evaluator = mmdet_utils.COCODatasetEvaluater(
        pipeline=None,
        hub_dataset=hub_dataset,
        classes=class_names,
        imgs=imgs,
        masks=masks,
        bboxes=bboxes,
        labels=labels,
        iscrowds=iscrowds,
        bbox_format=("LTWH", "pixel"),
    )

    data_infos = evaluator.load_annotations()

    targ_data_infos = pickle.load("indexes/data_infos.pkl")

    for idx, data_info in enumerate(data_infos):
        assert targ_data_infos[idx] == data_info


def test_check_unused_dataset_fields():
    cfg = mmcv.utils.config.ConfigDict()
    cfg.dataset_type = "COCODataset"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)

    cfg.data_root = "./"

    with pytest.warns(UserWarning):
        mmdet_utils.check_unused_dataset_fields(cfg)


def test_check_unsupported_train_pipeline_fields(cfg):
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


def test_check_dataset_augmentation_formats(cfg):
    cfg = mmcv.utils.config.ConfigDict()
    cfg.train_dataset = dict(type="ConcatDataset")

    with pytest.raises(Exception) as ex_info:
        mmdet_utils.check_unsupported_train_pipeline_fields(cfg)


def test_coco_to_pascal_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"mode": "LTWH", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 6, 7]])
    assert np.isclose(bbox_pascal, targ_bbox)

    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"mode": "LTWH", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)


def test_yolo_to_pascal_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"mode": "CCWH", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[3, 4, 5, 6]])
    assert np.isclose(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"mode": "CCWH", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)


def test_pascal_to_pascal_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 6, 7]])
    bbox_info = {"mode": "LTRB", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 6, 7]])
    assert np.isclose(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.6, 0.7]])
    bbox_info = {"mode": "LTRB", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_pascal_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)


def test_pascal_to_coco_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 6, 7]])
    bbox_info = {"mode": "LTRB", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 2, 2]])
    assert np.isclose(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.6, 0.7]])
    bbox_info = {"mode": "LTRB", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)


def test_yolo_to_coco_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"mode": "CCHW", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[3, 4, 2, 2]])
    assert np.isclose(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"mode": "CCHW", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)


def test_coco_to_coco_format():
    shape = (10, 10)
    bbox = np.array([[4, 5, 2, 2]])
    bbox_info = {"mode": "CCHW", "type": "pixel"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    targ_bbox = np.array([[4, 5, 2, 2]])
    assert np.isclose(bbox_pascal, targ_bbox)

    shape = (10, 10)
    bbox = np.array([[0.4, 0.5, 0.2, 0.2]])
    bbox_info = {"mode": "CCHW", "type": "fractional"}
    bbox_pascal = mmdet_.convert_to_coco_format(bbox, bbox_info, shape)
    assert np.isclose(bbox_pascal, targ_bbox)
