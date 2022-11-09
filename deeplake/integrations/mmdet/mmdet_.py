from collections import OrderedDict, defaultdict
import mmcv
import tqdm
from cProfile import label
from dataclasses import make_dataclass
from typing import Callable, Optional
from mmdet.apis.train import *
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmcv.utils import build_from_cfg, Registry
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate
from functools import partial
from typing import Optional, Sequence, Union
from deeplake.integrations.pytorch.common import PytorchTransformFunction
from deeplake.integrations.pytorch.dataset import TorchDataset
from deeplake.integrations.mmdet.mmdet_utils import HubCOCO

from mmdet.core import BitmapMasks
import albumentations as A
import deeplake as dp
from deeplake.util.warnings import always_warn
from click.testing import CliRunner
from deeplake.cli.auth import login, logout
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import tempfile
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from deeplake.integrations.mmdet import mmdet_utils


class MMDetDataset(TorchDataset):
    def __init__(
        self,
        *args,
        tensors_dict=None,
        bbox_format="PascalVOC",
        pipeline=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.CLASSES = self.get_classes()
        self.images = self._get_images(tensors_dict["images_tensor"])
        self.masks = self._get_masks(tensors_dict.get("masks_tensor", None))
        self.bbox_format = bbox_format
        self.bboxes = self._get_bboxes(tensors_dict["boxes_tensor"])
        self.labels = self._get_labels(tensors_dict["labels_tensor"])
        self.evaluator = (
            mmdet_utils.COCODatasetEvaluater(
                pipeline, classes=self.CLASSES, hub_dataset=self.dataset
            )
            if bbox_format == "COCO"
            else None
        )  # TO DO: read from htype info

    def _get_images(self, images_tensor):
        images_tensor = images_tensor or _find_tensor_with_htype(self.dataset, "image")
        return self.dataset[images_tensor].numpy(aslist=True)

    def _get_masks(self, masks_tensor):
        if masks_tensor is None:
            return None

        masks_tensor = masks_tensor or _find_tensor_with_htype(
            self.dataset, "binary_mask"
        )
        return self.dataset[masks_tensor].numpy(aslist=True)

    def _get_bboxes(self, boxes_tensor):
        boxes_tensor = boxes_tensor or _find_tensor_with_htype(self.dataset, "bbox")
        return self.dataset[boxes_tensor].numpy(aslist=True)

    def _get_labels(self, labels_tensor):
        labels_tensor = labels_tensor or _find_tensor_with_htype(
            self.dataset, "class_label"
        )
        return self.dataset[labels_tensor].numpy(aslist=True)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Raises:
            ValueError: when ``self.bbox_format`` is not valid.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.bbox_format == "PascalVOC":
            bboxes = self.bboxes[idx]
        elif self.bbox_format == "COCO":
            bboxes = self._coco_2_pascal(self.bboxes[idx])
        else:
            raise ValueError(f"Bounding boxes in {self.bbox_format} are not supported")
        return {
            "bboxes": bboxes,
            "labels": self.labels[idx],
        }

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = self.labels[idx].astype(np.int).tolist()

        return cat_ids

    # def pre_pipeline(self, results):
    #     """Prepare results dict for pipeline."""
    #     results["img_prefix"] = self.img_prefix
    #     results["seg_prefix"] = self.seg_prefix
    #     results["proposal_file"] = self.proposal_file
    #     results["bbox_fields"] = []
    #     results["mask_fields"] = []
    #     results["seg_fields"] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn("CustomDataset does not support filtering empty gt images.")
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    # def _set_group_flag(self):
    #     """Set flag according to image aspect ratio.

    #     Images with aspect ratio greater than 1 will be set as group 1,
    #     otherwise group 0.
    #     """
    #     self.flag = np.zeros(len(self), dtype=np.uint8)
    #     for i in range(len(self)):
    #         img_info = self.data_infos[i]
    #         if img_info["width"] / img_info["height"] > 1:
    #             self.flag[i] = 1

    # def _rand_another(self, idx):
    #     """Get another random index from the same group as the given index."""
    #     pool = np.where(self.flag == self.flag[idx])[0]
    #     return np.random.choice(pool)

    # def prepare_train_img(self, idx):
    #     """Get training data and annotations after pipeline.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Training data and annotation after pipeline with new keys \
    #             introduced by pipeline.
    #     """

    #     img_info = self.data_infos[idx]
    #     ann_info = self.get_ann_info(idx)
    #     results = dict(img_info=img_info, ann_info=ann_info)
    #     if self.proposals is not None:
    #         results["proposals"] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)

    # def prepare_test_img(self, idx):
    #     """Get testing data after pipeline.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Testing data after pipeline with new keys introduced by \
    #             pipeline.
    #     """

    #     img_info = self.data_infos[idx]
    #     results = dict(img_info=img_info)
    #     if self.proposals is not None:
    #         results["proposals"] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)

    def get_classes(self, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        labels_tensor = _find_tensor_with_htype(self.dataset, "class_label")
        return self.dataset[labels_tensor].info.class_names

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
        """
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

        self.evaluator.createHubIndex()
        return self.evaluator.evaluate(
            results,
            metric=metric,
            logger=logger,
            proposal_nums=proposal_nums,
            **kwargs,
        )

    # def evaluate_coco(
    #     self,
    #     results,
    #     metric="bbox",
    #     logger=None,
    #     jsonfile_prefix=None,
    #     classwise=False,
    #     proposal_nums=(100, 300, 1000),
    #     iou_thrs=None,
    #     metric_items=None,
    # ): # TO DO: Optimize this
    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ["bbox", "segm", "proposal", "proposal_fast"]
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f"metric {metric} is not supported")

    #     results_dict = self.convert_result_to_dict(results)
    #     targets_dict = self.convert_targets_to_dict()
    #     mAP_fn = MeanAveragePrecision(max_detection_thresholds=proposal_nums)
    #     mAP_fn.update(results_dict, targets_dict)
    #     s = mAP_fn.compute()
    #     eval_result = [(s_i, s[s_i].item()) for s_i in s]
    #     self._log_results(eval_result, proposal_nums)
    #     # for item in eval_result:
    #     #     print("")
    #     return OrderedDict(eval_result)

    # def _log_results(self, results, proposal_nums):
    #     out_str = "\n"
    #     for i, result in enumerate(results):
    #         if "map" in result[0]:
    #             if i == 0:
    #                 dets=proposal_nums[0]
    #             else:
    #                 dets=proposal_nums[2]

    #             if result[0] == "map":
    #                 out_str += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={dets}  ] = {result[1]:0.3f}\n"
    #             elif "50" in result[0] or "75" in result[0]:
    #                 result_int = result[0].split("_")[1]
    #                 out_str += f"Average Precision (AP) @[ IoU=0.{result_int}      | area=   all | maxDets={dets} ] = {result[1]:0.3f}\n"
    #             elif "small" in result[0]:
    #                 out_str += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets={dets} ] = {result[1]:0.3f}\n"
    #             elif "medium" in result[0]:
    #                 out_str += f"Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={dets} ] = {result[1]:0.3f}\n"
    #             elif "large" in result[0]:
    #                 out_str += f"Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets={dets} ] = {result[1]:0.3f}\n"
    #         else:
    #             if "per_class" not in result[0]:
    #                 out_str += "Average Recall    (AP) @[ IoU=0.50:0.95 | area="
    #                 if str(proposal_nums[0]) in result[0] \
    #                     or str(proposal_nums[0]) in result[0] \
    #                     or str(proposal_nums[1]) in result[0]:

    #                     result_int = result[0].split("_")[1]
    #                     int_len = len(result_int)
    #                     out_str += f'''   all | maxDets={result[0].split("_")[1]}{(5-int_len)*" "}] = {result[1]:0.3f}\n'''

    #                 elif "small" in result[0]:
    #                     out_str += f''' small | maxDets={dets} ] = {result[1]:0.3f}\n'''
    #                 elif "medium" in result[0]:
    #                     out_str += f'''medium | maxDets={dets} ] = {result[1]:0.3f}\n'''
    #                 elif "large" in result[0]:
    #                     out_str += f''' large | maxDets={dets} ] = {result[1]:0.3f}\n'''
    #     print_log(out_str)

    # def convert_result_to_dict(self, results): # TO DO: optimize this
    #     result_list = []
    #     for image_ann in tqdm.tqdm(results):
    #         result_dict = defaultdict(list)
    #         for class_id, (bboxes, masks) in enumerate(zip(image_ann[0], image_ann[1])):
    #             if bboxes.shape[0] > 1:
    #                 for i in range(bboxes.shape[0]):
    #                     result_dict["boxes"].append(bboxes[i][:4])
    #                     result_dict["scores"].append(bboxes[i][4])
    #                     result_dict["labels"].append(class_id)
    #                     result_dict["masks"].append(masks[i])
    #             elif bboxes.shape[0] == 1:
    #                 result_dict["boxes"].append(bboxes[0][:4])
    #                 result_dict["scores"].append(bboxes[0][4])
    #                 result_dict["labels"].append(class_id)
    #                 result_dict["masks"].append(masks)
    #         result_dict["boxes"] = torch.FloatTensor(np.array(result_dict["boxes"]))
    #         result_dict["scores"] = torch.FloatTensor(np.array(result_dict["scores"]))
    #         result_dict["labels"] = torch.LongTensor(
    #             np.array(result_dict["labels"], dtype=np.uint8)
    #         )
    #         result_list.append(result_dict)
    #     return result_list

    # def convert_targets_to_dict(self): # TO DO: Optimize this
    #     targets_list = []
    #     for image_labels, image_boxes in tqdm.tqdm(zip(self.labels, self.bboxes)):
    #         targets_dict = defaultdict(list)
    #         for label, box in zip(image_labels, image_boxes):
    #             targets_dict["labels"].append(label)
    #             targets_dict["boxes"].append(box)
    #         targets_dict["boxes"] = torch.FloatTensor(np.array(targets_dict["boxes"]))
    #         targets_dict["labels"] = torch.LongTensor(
    #             np.array(targets_dict["labels"], dtype=np.uint8)
    #         )
    #         targets_list.append(targets_dict)
    #     return targets_list

    @staticmethod
    def _coco_2_pascal(boxes):
        # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height
        return np.stack(
            (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 0] + np.clip(boxes[:, 2], 1, None),
                boxes[:, 1] + np.clip(boxes[:, 3], 1, None),
            ),
            axis=1,
        )

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = "Test" if self.test_mode else "Train"
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


class HubDatasetCLass:
    def __init__(self, cfg, ds_path=None):
        username = cfg.deeplake_credentials.username
        password = cfg.deeplake_credentials.password
        if username is not None:
            runner = CliRunner()
            runner.invoke(login, f"-u {username} -p {password}")
        ds_path = ds_path or cfg.deeplake_path
        self.ds = dp.load(ds_path, token=cfg.deeplake_credentials.token)
        labels_tensor = _find_tensor_with_htype(self.ds, "class_label")
        self.CLASSES = self.ds[labels_tensor].info.class_names
        self.pipeline = cfg.pipeline


rand_crop = A.Compose(
    [
        A.RandomSizedBBoxSafeCrop(width=128, height=128, erosion_rate=0.2),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels", "bbox_ids"],
        min_area=25,
        min_visibility=0.6,
    ),
)


def _find_tensor_with_htype(ds: dp.Dataset, htype: str):
    tensors = [k for k, v in ds.tensors.items() if v.meta.htype == htype]
    if not tensors:
        always_warn(f"No tensor found with htype='{htype}'")
        return None
    t = tensors[0]
    if len(tensors) > 1:
        always_warn(f"Multiple tensors with htype='{htype}' found. choosing '{t}'.")
    return t


def transform(
    sample_in,
    images_tensor: str,
    masks_tensor: str,
    boxes_tensor: str,
    labels_tensor: str,
    pipeline: Callable,
):
    img = sample_in[images_tensor]
    if masks_tensor:
        masks = sample_in[masks_tensor]
    else:
        masks = None
    bboxes = sample_in[boxes_tensor]
    labels = sample_in[labels_tensor]

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if masks is not None:
        masks = masks.transpose((2, 0, 1)).astype(np.uint8)

    shape = img.shape

    if isinstance(pipeline, list):
        pipeline = pipeline[0]

    if masks is not None:
        gt_masks = BitmapMasks(masks, *shape[:2])
    else:
        gt_masks = None

    return pipeline(
        {
            "img": img,
            "img_fields": ["img"],
            "filename": None,
            "ori_filename": None,
            "img_shape": shape,
            "ori_shape": shape,
            "gt_masks": gt_masks,
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "bbox_fields": ["gt_bboxes"],
        }
    )


def build_dataset(cfg, *args, ds_path=None, **kwargs):
    if "deeplake_path" in cfg:
        # TO DO: add preprocessing functions related to mmdet dataset classes like RepeatDataset etc...
        return HubDatasetCLass(cfg, ds_path=ds_path)
    return mmdet_build_dataset(cfg, *args, **kwargs)


def build_dataloader(
    dataset,
    images_tensor,
    masks_tensor,
    boxes_tensor,
    labels_tensor,
    **train_loader_config,
):
    if isinstance(dataset, HubDatasetCLass):
        images_tensor = images_tensor or _find_tensor_with_htype(dataset.ds, "image")
        masks_tensor = masks_tensor or _find_tensor_with_htype(
            dataset.ds, "binary_mask"
        )
        boxes_tensor = boxes_tensor or _find_tensor_with_htype(dataset.ds, "bbox")
        labels_tensor = labels_tensor or _find_tensor_with_htype(
            dataset.ds, "class_label"
        )
        pipeline = build_pipeline(dataset.pipeline)

        transform_fn = partial(
            transform,
            images_tensor=images_tensor,
            masks_tensor=masks_tensor,
            boxes_tensor=boxes_tensor,
            labels_tensor=labels_tensor,
            pipeline=pipeline,
        )
        num_workers = train_loader_config["workers_per_gpu"]
        shuffle = train_loader_config.get("shuffle", True)
        # shuffle = False
        tensors_dict = {
            "images_tensor": images_tensor,
            "boxes_tensor": boxes_tensor,
            "labels_tensor": labels_tensor,
        }
        tensors = [images_tensor, labels_tensor, boxes_tensor]
        if masks_tensor is not None:
            tensors.append(masks_tensor)
            tensors_dict["masks_tensor"] = masks_tensor

        loader = dataset.ds.pytorch(
            tensors_dict=tensors_dict,
            num_workers=num_workers,
            shuffle=shuffle,
            transform=transform_fn,
            tensors=tensors,
            collate_fn=partial(
                collate, samples_per_gpu=train_loader_config["samples_per_gpu"]
            ),
            torch_dataset=MMDetDataset,
            bbox_format=train_loader_config["bbox_format"],
            pipeline=dataset.pipeline,
            # torch_dataset=TorchDataset,
        )
        # loader.dataset.CLASSES = [
        #     c["name"] for c in dataset.ds.categories.info["category_info"]
        # ]

        labels_tensor = _find_tensor_with_htype(dataset.ds, "class_label")
        loader.dataset.CLASSES = dataset.ds[labels_tensor].info.class_names
        return loader

    return mmdet_build_dataloader(dataset, **train_loader_config)


def build_pipeline(steps):
    return Compose(
        [
            build_from_cfg(step, PIPELINES, None)
            for step in steps
            if step["type"] not in {"LoadImageFromFile", "LoadAnnotations"}
        ]
    )


def train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
    images_tensor: Optional[str] = None,  # from config file
    masks_tensor: Optional[str] = None,
    boxes_tensor: Optional[str] = None,
    labels_tensor: Optional[str] = None,
    bbox_format="PascalVOC",
):

    cfg = compat_cfg(cfg)

    tensors = cfg.get("tensors", {})
    images_tensor = images_tensor or tensors.get("img")
    masks_tensor = masks_tensor or tensors.get("gt_masks")
    boxes_tensor = boxes_tensor or tensors.get("gt_bboxes")
    labels_tensor = labels_tensor or tensors.get("gt_labels")

    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=256,
        workers_per_gpu=8,
        # `num_gpus` will be ignored if distributed
        num_gpuddes=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
        bbox_format=bbox_format,
    )

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get("train_dataloader", {}),
    }

    data_loaders = [
        build_dataloader(
            ds,
            images_tensor,
            masks_tensor,
            boxes_tensor,
            labels_tensor,
            **train_loader_cfg,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get("custom_hooks", None),
    )

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
            bbox_format=bbox_format,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get("val_dataloader", {}),
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            images_tensor,
            masks_tensor,
            boxes_tensor,
            labels_tensor,
            **val_dataloader_args,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
