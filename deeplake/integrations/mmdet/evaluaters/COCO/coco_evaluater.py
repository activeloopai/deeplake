import math
from typing import Any, Dict, List, Optional

import numpy as np

import deeplake
from .deeplake_coco import DeeplakeCOCO

import mmcv
from mmdet.datasets import coco as mmdet_coco


class Evaluater(mmdet_coco.CocoDataset):
    """Dataset class for COCO evaluation.

    Args:
        deeplake_dataset: A DeepLake dataset object.
        classes: A list of class names.
        img_prefix: Prefix of image file paths.
        seg_prefix: Prefix of segmentation map file paths.
        seg_suffix: Suffix of segmentation map file paths.
        proposal_file: File path of bounding box proposal file.
        test_mode: A flag to indicate if the dataset is used for testing.
        filter_empty_gt: A flag to indicate if empty ground truth annotations should be filtered out.
        file_client_args: Keyword arguments to be passed to `mmcv.FileClient`.
        imgs: images tensor.
        masks: masks deeplake tensor.
        bboxes: bboxes deeplake tensor.
        labels: labels deeplake tensor.
        iscrowds: iscrowds deeplake tensor.
        bbox_format: A string that represents the format of bounding boxes.
        batch_size: Number of samples in a batch.
        num_gpus: Number of GPUs.
    """

    def __init__(
        self,
        deeplake_dataset: Any = None,
        classes: Optional[List[str]] = None,
        img_prefix: str = "",
        seg_prefix: Optional[str] = None,
        seg_suffix: str = ".png",
        proposal_file: Optional[str] = None,
        test_mode: bool = True,
        filter_empty_gt: bool = True,
        file_client_args: Dict[str, Any] = dict(backend="disk"),
        imgs: Optional[List[Any]] = None,
        masks: Optional[deeplake.core.Tensor] = None,
        bboxes: Optional[List[deeplake.core.Tensor]] = None,
        labels: Optional[List[deeplake.core.Tensor]] = None,
        iscrowds: Optional[List[int]] = None,
        bbox_format: Optional[str] = None,
        batch_size: int = 1,
        num_gpus: int = 1,
    ):
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.seg_suffix = seg_suffix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = classes
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.data_infos = self.load_annotations(
            deeplake_dataset,
            imgs=imgs,
            labels=labels,
            masks=masks,
            bboxes=bboxes,
            iscrowds=iscrowds,
            class_names=self.CLASSES,
            bbox_format=bbox_format,
        )
        self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline

    def pipeline(self, x):
        return x

    def __len__(self):
        "Method for finding length of the evaluater"
        length = super().__len__()
        per_gpu_length = math.floor(
            length / (self.batch_size * self.num_gpus)
        )  # not sure whther batch size is needed
        total_length = per_gpu_length * self.num_gpus
        return total_length

    def load_annotations(
        self,
        deeplake_dataset: deeplake.Dataset,
        imgs: deeplake.Tensor = None,
        labels: List[np.ndarray] = None,
        masks: List[np.ndarray] = None,
        bboxes: List[np.ndarray] = None,
        iscrowds: List[np.ndarray] = None,
        class_names: List[str] = None,
        bbox_format: Dict[Dict[str, str]] = None,
    ):
        """Load annotation from COCO style annotation file.

        Args:
            deeplake_dataset (dp.Dataset): Deeplake dataset object.
            imgs (dp.Tensor): image deeplake tensor.
            labels (List[numpy]): List of labels for every every detection for each image in numpy format.
            masks (List[numpy]): List of masks for every every detection for each image in numpy format.
            bboxes (List[numpy]): List of bboxes for every every detection for each image in numpy.
            iscrowds (List[numpy]): List of iscrowds for every every detection for each image in numpy format.
            class_names (List[str]): List of class names for every every detection for each image.
            bbox_format (Dict[Dict[str, str]]): Dictionary contatining bbox format information.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = DeeplakeCOCO(
            deeplake_dataset,
            imgs=imgs,
            labels=labels,
            bboxes=bboxes,
            masks=masks,
            iscrowds=iscrowds,
            class_names=class_names,
            bbox_format=bbox_format,
        )
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids)
        return data_infos
