from collections import OrderedDict
from typing import Callable, Optional, List, Dict, Sequence

import os
import math
import types
import torch
import warnings
import tempfile
import numpy as np
import os.path as osp

from PIL import Image, ImageDraw  # type: ignore

from terminaltables import AsciiTable  # type: ignore

try:
    from mmdet.apis.train import auto_scale_lr  # type: ignore
except Exception:
    import mmdet  # type: ignore

    version = mmdet.__version__
    raise Exception(
        f"MMDet {version} version is not supported. The latest supported MMDet version with deeplake is 2.28.1."
    )

from mmdet.core import eval_map, eval_recalls
from mmdet.core import BitmapMasks, PolygonMasks

import mmcv  # type: ignore
from mmcv.utils import print_log

import deeplake as dp
from deeplake.types import TypeKind

from deeplake.integrations.mm.upcast_array import upcast_array
from deeplake.integrations.mm.warnings import always_warn
from deeplake.integrations.mmdet import mmdet_utils_

from torch.utils.data import DataLoader

# Monkey-patch the function
from deeplake.integrations.mm.exceptions import InvalidImageError
from deeplake.integrations.mmdet.test_ import single_gpu_test as custom_single_gpu_test
from deeplake.integrations.mmdet.test_ import multi_gpu_test as custom_multi_gpu_test

from torch.utils.data import Dataset


def coco_pixel_2_pascal_pixel(boxes, shape):
    """
    Converts bounding boxes from COCO pixel format (x, y, width, height)
    to Pascal VOC pixel format (x_min, y_min, x_max, y_max).

    Clipping ensures the bounding boxes have non-negative width and height.

    @param boxes: numpy array of shape (N, 4), containing bounding boxes in COCO format.
    @param shape: tuple, the shape of the image (height, width).

    @return: numpy array of shape (N, 4), bounding boxes in Pascal VOC format.
    """
    pascal_boxes = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        pascal_boxes = np.stack(
            (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 0] + boxes[:, 2],
                boxes[:, 1] + boxes[:, 3],
            ),
            axis=1,
        )
    return pascal_boxes


def poly_2_mask(polygons, shape):
    # TODO This doesnt fill the array inplace.    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    """
    Converts a list of polygons into a binary mask.

    @param polygons: list of polygons, where each polygon is a list of (x, y) coordinates.
    @param shape: tuple, the shape of the mask (height, width).

    @return: numpy array, binary mask of the same size as the image.
    """
    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        im = Image.fromarray(out[..., i])
        d = ImageDraw.Draw(im)
        d.polygon(polygon, fill=1)
        out[..., i] = np.asarray(im)
    return out


def coco_frac_2_pascal_pixel(boxes, shape):
    """
    Converts bounding boxes from fractional COCO format (relative to image size)
    to Pascal VOC pixel format.

    @param boxes: numpy array of shape (N, 4), bounding boxes in fractional COCO format.
    @param shape: tuple, the shape of the image (height, width).

    @return: numpy array of shape (N, 4), bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x = boxes[:, 0] * shape[1]
        y = boxes[:, 1] * shape[0]
        w = boxes[:, 2] * shape[1]
        h = boxes[:, 3] * shape[0]
        bbox = np.stack((x, y, w, h), axis=1)
    return coco_pixel_2_pascal_pixel(bbox, shape)


def pascal_frac_2_pascal_pixel(boxes, shape):
    """
    Converts bounding boxes from fractional Pascal VOC format (LTRB)
    to pixel Pascal VOC format.

    @param boxes: numpy array of shape (N, 4), bounding boxes in fractional format.
    @param shape: tuple, the shape of the image (height, width).

    @return: numpy array of shape (N, 4), bounding boxes in pixel format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_top = boxes[:, 0] * shape[1]
        y_top = boxes[:, 1] * shape[0]
        x_bottom = boxes[:, 2] * shape[1]
        y_bottom = boxes[:, 3] * shape[0]
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_pixel_2_pascal_pixel(boxes, shape):
    """
    Converts bounding boxes from YOLO pixel format (center_x, center_y, width, height)
    to Pascal VOC pixel format (LTRB).

    @param boxes: numpy array of shape (N, 4), bounding boxes in YOLO format.
    @param shape: tuple, the shape of the image (height, width).

    @return: numpy array of shape (N, 4), bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_top = np.array(boxes[:, 0]) - np.floor(np.array(boxes[:, 2]) / 2)
        y_top = np.array(boxes[:, 1]) - np.floor(np.array(boxes[:, 3]) / 2)
        x_bottom = np.array(boxes[:, 0]) + np.floor(np.array(boxes[:, 2]) / 2)
        y_bottom = np.array(boxes[:, 1]) + np.floor(np.array(boxes[:, 3]) / 2)
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_frac_2_pascal_pixel(boxes, shape):
    """
    Converts bounding boxes from YOLO fractional format to Pascal VOC pixel format.

    @param boxes: numpy array of shape (N, 4), bounding boxes in YOLO fractional format.
    @param shape: tuple, the shape of the image (height, width).

    @return: numpy array of shape (N, 4), bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_center = boxes[:, 0] * shape[1]
        y_center = boxes[:, 1] * shape[0]
        width = boxes[:, 2] * shape[1]
        height = boxes[:, 3] * shape[0]
        bbox = np.stack((x_center, y_center, width, height), axis=1)
    return yolo_pixel_2_pascal_pixel(bbox, shape)


def get_bbox_format(bbox, bbox_info):
    bbox_info = bbox_info.get("coords")
    if not bbox_info:
        bbox_info = {}
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)


BBOX_FORMAT_TO_PASCAL_CONVERTER = {
    ("LTWH", "pixel"): coco_pixel_2_pascal_pixel,
    ("LTWH", "fractional"): coco_frac_2_pascal_pixel,
    ("LTRB", "pixel"): lambda x, y: x,
    ("LTRB", "fractional"): pascal_frac_2_pascal_pixel,
    ("CCWH", "pixel"): yolo_pixel_2_pascal_pixel,
    ("CCWH", "fractional"): yolo_frac_2_pascal_pixel,
}


def convert_to_pascal_format(bbox, bbox_info, shape):
    bbox_format = get_bbox_format(bbox, bbox_info)
    converter = BBOX_FORMAT_TO_PASCAL_CONVERTER[bbox_format]
    return converter(bbox, shape)


def pascal_pixel_2_coco_pixel(boxes, images):
    """
    Converts bounding boxes from Pascal VOC pixel format (LTRB)
    to COCO pixel format (x, y, width, height).

    @param boxes: numpy array of images (N, 4), bounding boxes in Pascal VOC format.
    @param images: tuple, the images of the image (height, width).

    @return: numpy array of images (N, 4), bounding boxes in COCO pixel format.
    """
    pascal_boxes = []
    for box in boxes:
        if box.size != 0:
            pascal_boxes.append(
                np.stack(
                    (
                        box[:, 0],
                        box[:, 1],
                        box[:, 2] - box[:, 0],
                        box[:, 3] - box[:, 1],
                    ),
                    axis=1,
                )
            )
        else:
            pascal_boxes.append(box)
    return pascal_boxes


def pascal_frac_2_coco_pixel(boxes, images):
    pascal_pixel_boxes = []
    for i, box in enumerate(boxes):
        if box.size != 0:
            shape = images[i].shape
            x_top = box[:, 0] * shape[1]
            y_top = box[:, 1] * shape[0]
            x_bottom = box[:, 2] * shape[1]
            y_bottom = box[:, 3] * shape[0]
            bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
        pascal_pixel_boxes.append(bbox)
    return pascal_pixel_2_coco_pixel(pascal_pixel_boxes, images)


def yolo_pixel_2_coco_pixel(boxes, images):
    yolo_boxes = []
    for box in boxes:
        if box.size != 0:
            x_top = np.array(box[:, 0]) - np.floor(np.array(box[:, 2]) / 2)
            y_top = np.array(box[:, 1]) - np.floor(np.array(box[:, 3]) / 2)
            w = box[:, 2]
            h = box[:, 3]
            bbox = np.stack([x_top, y_top, w, h], axis=1)
        yolo_boxes.append(bbox)
    return yolo_boxes


def yolo_frac_2_coco_pixel(boxes, images):
    yolo_boxes = []
    for i, box in enumerate(boxes):
        shape = images[i].shape
        x_center = box[:, 0] * shape[1]
        y_center = box[:, 1] * shape[0]
        width = box[:, 2] * shape[1]
        height = box[:, 3] * shape[0]
        bbox = np.stack((x_center, y_center, width, height), axis=1)
        yolo_boxes.append(bbox)
    return yolo_pixel_2_coco_pixel(yolo_boxes, images)


def coco_frac_2_coco_pixel(boxes, images):
    coco_pixel_boxes = []
    for i, box in enumerate(boxes):
        shape = images[i].shape
        x = box[:, 0] * shape[1]
        y = box[:, 1] * shape[0]
        w = box[:, 2] * shape[1]
        h = box[:, 3] * shape[0]
        bbox = np.stack((x, y, w, h), axis=1)
        coco_pixel_boxes.append(bbox)
    return np.array(coco_pixel_boxes)


BBOX_FORMAT_TO_COCO_CONVERTER = {
    ("LTWH", "pixel"): lambda x, y: x,
    ("LTWH", "fractional"): coco_frac_2_coco_pixel,
    ("LTRB", "pixel"): pascal_pixel_2_coco_pixel,
    ("LTRB", "fractional"): pascal_frac_2_coco_pixel,
    ("CCWH", "pixel"): yolo_pixel_2_coco_pixel,
    ("CCWH", "fractional"): yolo_frac_2_coco_pixel,
}


def convert_to_coco_format(bbox, bbox_format, images):
    converter = BBOX_FORMAT_TO_COCO_CONVERTER[bbox_format]
    return converter(bbox, images)


def first_non_empty(bboxes):
    for box in bboxes:
        if len(box):
            return box
    raise ValueError("Empty bboxes")


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    boxes_tensor: str,
    labels_tensor: str,
    pipeline: Callable,
    bbox_info: str,
    poly2mask: bool,
):
    img = upcast_array(sample_in[images_tensor])
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    bboxes = upcast_array(sample_in[boxes_tensor])
    # TODO bbox format should be recognized outside the transform, not per sample basis.
    bboxes = convert_to_pascal_format(bboxes, bbox_info, img.shape)
    if bboxes.shape == (0, 0):  # TO DO: remove after bug will be fixed
        bboxes = np.empty((0, 4), dtype=sample_in[boxes_tensor].dtype)

    labels = upcast_array(sample_in[labels_tensor])

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    shape = img.shape

    pipeline_dict = {
        "img": np.ascontiguousarray(img, dtype=np.float32),
        "img_fields": ["img"],
        "filename": None,
        "ori_filename": None,
        "img_shape": shape,
        "ori_shape": shape,
        "gt_bboxes": bboxes,
        "gt_labels": labels,
        "bbox_fields": ["gt_bboxes"],
    }

    if masks_tensor:
        masks = upcast_array(sample_in[masks_tensor])
        if poly2mask:
            masks = mmdet_utils_.convert_poly_to_coco_format(masks)
            masks = PolygonMasks(
                [process_polygons(polygons) for polygons in masks], shape[0], shape[1]
            )
        else:
            masks = BitmapMasks(masks.astype(np.uint8).transpose(2, 0, 1), *shape[:2])

        pipeline_dict["gt_masks"] = masks
        pipeline_dict["mask_fields"] = ["gt_masks"]
    return pipeline(pipeline_dict)


def process_polygons(polygons):
    """Convert polygons to list of ndarray and filter invalid polygons.

    Args:
        polygons (list[list]): Polygons of one instance.

    Returns:
        list[numpy.ndarray]: Processed polygons.
    """

    polygons = [np.array(p) for p in polygons]
    valid_polygons = []
    for polygon in polygons:
        if len(polygon) % 2 == 0 and len(polygon) >= 6:
            valid_polygons.append(polygon)
    return valid_polygons


class MMDetTorchDataset(Dataset):
    def __init__(
        self,
        dataset,
        tensors: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.column_names = [col.name for col in self.dataset.schema.columns]
        self.last_successful_index = -1

    def __getstate__(self):
        return {
            "dataset": self.dataset,
            "transform": self.transform,
            "column_names": self.column_names,
            "last_successful_index": self.last_successful_index,
        }

    def __setstate__(self, state):
        """Restore state from pickled state."""
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)

        self.dataset = state["dataset"]
        self.transform = state["transform"]
        self.column_names = state["column_names"]
        self.last_successful_index = state["last_successful_index"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.dataset[idx]
                if self.transform:
                    return self.transform(sample)
                else:
                    out = {}
                    for col in self.column_names:
                        out[col] = sample[col]
                    return out
            except InvalidImageError as e:
                print(f"Error processing data at index {idx}: {e}")
                if self.last_successful_index == -1:
                    self.last_successful_index = idx + 1
                idx = self.last_successful_index
                continue


class MMDetDataset(MMDetTorchDataset):
    def __init__(
        self,
        *args,
        tensors_dict=None,
        mode="train",
        metrics_format="COCO",
        bbox_info=None,
        pipeline=None,
        num_gpus=1,
        batch_size=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.pipeline = pipeline
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.tensors_dict = tensors_dict
        self.bbox_info = bbox_info
        if self.mode in ("val", "test"):
            self.images = self._get_images(self.tensors_dict["images_tensor"])
            masks = self._get_masks(self.tensors_dict.get("masks_tensor", None))
            masks_type_kind = (
                self.dataset.schema[masks.name].dtype.kind
                if masks is not None and masks != []
                else None
            )
            self.masks_type_kind = masks_type_kind
            self.masks = masks[:]
            self.bboxes = self._get_bboxes(self.tensors_dict["boxes_tensor"])
            bbox_format = get_bbox_format(first_non_empty(self.bboxes), bbox_info)
            self.labels = self._get_labels(self.tensors_dict["labels_tensor"])
            self.iscrowds = self._get_iscrowds(self.tensors_dict.get("iscrowds"))
            self.CLASSES = self.get_classes(self.tensors_dict["labels_tensor"])
            self.metrics_format = metrics_format
            coco_style_bbox = convert_to_coco_format(
                self.bboxes, bbox_format, self.images
            )

            if self.metrics_format == "COCO":
                self.evaluator = mmdet_utils_.COCODatasetEvaluater(
                    pipeline,
                    classes=self.CLASSES,
                    deeplake_dataset=self.dataset,
                    imgs=self.images,
                    masks=self.masks,
                    masks_type_kind=self.masks_type_kind,
                    bboxes=coco_style_bbox,
                    labels=self.labels,
                    iscrowds=self.iscrowds,
                    bbox_format=bbox_format,
                    num_gpus=num_gpus,
                )
            else:
                self.evaluator = None

    def __getstate__(self):
        """Prepare state for pickling."""
        state = super().__getstate__() if hasattr(super(), "__getstate__") else {}

        state.update(
            {
                "mode": self.mode,
                "pipeline": self.pipeline,
                "num_gpus": self.num_gpus,
                "batch_size": self.batch_size,
                "tensors_dict": self.tensors_dict,
                "bbox_info": self.bbox_info,
            }
        )
        return state

    def __setstate__(self, state):
        """Restore state from pickled state."""
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)

        self.mode = state["mode"]
        self.pipeline = state["pipeline"]
        self.num_gpus = state["num_gpus"]
        self.batch_size = state["batch_size"]
        self.tensors_dict = state["tensors_dict"]
        self.bbox_info = state["bbox_info"]

        if self.mode in ("val", "test"):
            self.images = self._get_images(self.tensors_dict["images_tensor"])
            masks = self._get_masks(self.tensors_dict.get("masks_tensor", None))
            masks_type_kind = (
                self.dataset.schema[masks.name].dtype.kind
                if masks is not None and masks != []
                else None
            )
            self.masks_type_kind = masks_type_kind
            self.masks = masks[:]
            self.bboxes = self._get_bboxes(self.tensors_dict["boxes_tensor"])
            bbox_format = get_bbox_format(first_non_empty(self.bboxes), bbox_info)
            self.labels = self._get_labels(self.tensors_dict["labels_tensor"])
            self.iscrowds = self._get_iscrowds(self.tensors_dict.get("iscrowds"))
            self.CLASSES = self.get_classes(self.tensors_dict["labels_tensor"])
            self.metrics_format = metrics_format
            coco_style_bbox = convert_to_coco_format(
                self.bboxes, bbox_format, self.images
            )

            if self.metrics_format == "COCO":
                self.evaluator = mmdet_utils_.COCODatasetEvaluater(
                    pipeline,
                    classes=self.CLASSES,
                    deeplake_dataset=self.dataset,
                    imgs=self.images,
                    masks=self.masks,
                    masks_type_kind=self.masks_type_kind,
                    bboxes=coco_style_bbox,
                    labels=self.labels,
                    iscrowds=self.iscrowds,
                    bbox_format=bbox_format,
                    num_gpus=num_gpus,
                )
            else:
                self.evaluator = None

    def __len__(self):
        if self.mode == "val":
            per_gpu_length = math.floor(
                len(self.dataset) / (self.batch_size * self.num_gpus)
            )
            total_length = per_gpu_length * self.num_gpus
            return total_length
        return super().__len__()

    def _get_images(self, images_tensor):
        image_tensor = self.dataset[images_tensor]
        return image_tensor

    def _get_masks(self, masks_tensor):
        if masks_tensor is None:
            return []
        return self.dataset[masks_tensor]

    def _get_iscrowds(self, iscrowds_tensor):
        if iscrowds_tensor is not None:
            return iscrowds_tensor

        if "iscrowds" in [col.name for col in self.dataset.schema.columns]:
            always_warn(
                "Iscrowds was not specified, searching for iscrowds tensor in the dataset."
            )
            return self.dataset["iscrowds"][:]
        always_warn("iscrowds tensor was not found, setting its value to 0.")
        return iscrowds_tensor

    def _get_bboxes(self, boxes_tensor):
        return self.dataset[boxes_tensor][:]

    def _get_labels(self, labels_tensor):
        return self.dataset[labels_tensor][:]

    def _get_class_names(self, labels_tensor):
        return self.dataset[labels_tensor].metadata["class_names"]

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Raises:
            ValueError: when ``self.metrics`` is not valid.

        Returns:
            dict: Annotation info of specified index.
        """
        bboxes = convert_to_pascal_format(
            self.bboxes[idx], self.bbox_info, self.images[idx].shape
        )
        return {"bboxes": bboxes, "labels": self.labels[idx]}

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = self.labels[idx].astype(np.int).tolist()

        return cat_ids

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn("CustomDataset does not support filtering empty gt images.")
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_classes(self, classes):
        """Get class names of current dataset.

        Args:
            classes (str): Reresents the name of the classes tensor. Overrides the CLASSES defined by the dataset.

        Returns:
            list[str]: Names of categories of the dataset.
        """
        return self.dataset[classes].metadata["class_names"]

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,  #
        scale_ranges=None,
        **kwargs,
    ):
        """Evaluate the dataset.

        Args:
            **kwargs (dict): Keyword arguments to pass to self.evaluate object
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.

        Raises:
            KeyError: if a specified metric format is not supported

        Returns:
            OrderedDict: Evaluation metrics dictionary
        """
        if self.num_gpus > 1:
            results_ordered = []
            for i in range(self.num_gpus):
                results_ordered += results[i :: self.num_gpus]
            results = results_ordered

        if self.evaluator is None:
            if not isinstance(metric, str):
                assert len(metric) == 1
                metric = metric[0]
            allowed_metrics = ["mAP", "recall"]
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")
            annotations = [
                self.get_ann_info(i) for i in range(len(self))
            ]  # directly evaluate from hub
            eval_results = OrderedDict()
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            if metric == "mAP":
                assert isinstance(iou_thrs, list)
                mean_aps = []
                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, _ = eval_map(
                        results,
                        annotations,
                        scale_ranges=scale_ranges,
                        iou_thr=iou_thr,
                        dataset=self.CLASSES,
                        logger=logger,
                    )
                    mean_aps.append(mean_ap)
                    eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
                eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
            elif metric == "recall":
                gt_bboxes = [ann["bboxes"] for ann in annotations]  # evaluate from hub
                recalls = eval_recalls(
                    gt_bboxes, results, proposal_nums, iou_thr, logger=logger
                )
                for i, num in enumerate(proposal_nums):
                    for j, iou in enumerate(iou_thrs):
                        eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
                if recalls.shape[1] > 1:
                    ar = recalls.mean(axis=1)
                    for i, num in enumerate(proposal_nums):
                        eval_results[f"AR@{num}"] = ar[i]
            return eval_results

        return self.evaluator.evaluate(
            results,
            metric=metric,
            logger=logger,
            proposal_nums=proposal_nums,
            **kwargs,
        )

    @staticmethod
    def _coco_2_pascal(boxes):
        # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height
        return np.stack(
            (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 0] + boxes[:, 2],
                boxes[:, 1] + boxes[:, 3],
            ),
            axis=1,
        )

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = "Test"
        #  if self.test_mode else "Train"
        result = (
            f"\n{self.__class__.__name__} {dataset_type} dataset "
            f"with number of images {len(self)}, "
            f"and instance counts: \n"
        )
        if self.CLASSES is None:
            result += "Category names are not provided. \n"
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)["labels"]  # change this
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [["category", "count"] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f"{cls} [{self.CLASSES[cls]}]", f"{count}"]
            else:
                # add the background number
                row_data += ["-1 background", f"{count}"]
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == "0":
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            kwargs (dict): Additional keyword arguments to be passed.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir
