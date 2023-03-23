from typing import Callable

from mmdet.core import BitmapMasks  # type: ignore
import numpy as np
from mmdet.utils.util_distribution import *  # type: ignore
from deeplake.integrations.mmdet import mmdet_utils

from ..converters import pascal_format, polygons


class Transform:
    def __init__(
        self,
        sample_in,
        images_tensor,
        masks_tensor,
        boxes_tensor,
        labels_tensor,
        pipeline,
        bbox_info,
        poly2mask,
    ):
        self.sample_in = sample_in
        self.images_tensor = images_tensor
        self.masks_tensor = masks_tensor
        self.boxes_tensor = boxes_tensor
        self.labels_tensor = labels_tensor
        self.pipeline = pipeline
        self.bbox_info = bbox_info
        self.poly2mask = poly2mask

    @property
    def orig_img_shape(self):
        _img = self.sample_in[self.images_tensor]
        if not isinstance(_img, np.ndarray):
            _img = np.array(_img)
        return _img.shape

    @property
    def img(self):
        _img = self.sample_in[self.images_tensor]
        if not isinstance(_img, np.ndarray):
            _img = np.array(_img)

        if _img.ndim == 2:
            _img = np.expand_dims(_img, -1)

        _img = _img[..., ::-1]  # rgb_to_bgr should be optional
        if _img.shape[2] == 1:
            _img = np.repeat(_img, 3, axis=2)
        return _img

    @property
    def bboxes(self):
        _bboxes = self.sample_in[self.boxes_tensor]
        # TODO bbox format should be recognized outside the transform, not per sample basis.
        _bboxes = pascal_format.convert(_bboxes, self.bbox_info, self.orig_img_shape)
        if _bboxes.shape == (0, 0):  # TO DO: remove after bug will be fixed
            _bboxes = np.empty((0, 4), dtype=self.sample_in[self.boxes_tensor].dtype)
        return _bboxes

    @property
    def labels(self):
        return self.sample_in[self.labels_tensor]

    @property
    def shape(self):
        return self.img.shape

    @property
    def pipeline_dict(self):
        _pipeline_dict = {
            "img": np.ascontiguousarray(self.img, dtype=np.float32),
            "img_fields": ["img"],
            "filename": None,
            "ori_filename": None,
            "img_shape": self.shape,
            "ori_shape": self.shape,
            "gt_bboxes": self.bboxes,
            "gt_labels": self.labels,
            "bbox_fields": ["gt_bboxes"],
        }
        if not self.masks_tensor:
            return _pipeline_dict

        masks_tensor = self.get_masks_tensor()
        _pipeline_dict["gt_masks"] = masks_tensor
        _pipeline_dict["mask_fields"] = ["gt_masks"]

        return _pipeline_dict

    def get_masks_tensor(self):
        masks = self.sample_in[self.masks_tensor]
        return polygons.convert_polygons_to_mask(masks, self.poly2mask, self.shape)
        
    def process(self):
        return pipeline(self.pipeline_dict)
