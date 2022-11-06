from collections import OrderedDict
import mmcv
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
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose


class MMDetDataset(TorchDataset):
    def __init__(
        self,
        *args,
        images_tensor=None,
        masks_tensor=None,
        boxes_tensor=None,
        labels_tensor=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.CLASSES = self.get_classes()
        self.images = self._get_images(images_tensor)
        self.masks = self._get_masks(masks_tensor)
        self.bboxes = self._get_bboxes(boxes_tensor)
        self.labels = self._get_labels(labels_tensor)

    def _get_images(self, images_tensor):
        images_tensor = images_tensor or _find_tensor_with_htype(self.dataset, "image")
        return self.dataset[images_tensor].numpy(aslist=True)

    def _get_masks(self, masks_tensor):
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

        Returns:
            dict: Annotation info of specified index.
        """
        return {
            "bboxes": self.bboxes[idx],
            "labels": self.labels[idx],
        }

    # def get_cat_ids(self, idx):
    #     """Get category ids by index.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         list[int]: All categories in the image of specified index.
    #     """

    #     return self.data_infos[idx]["ann"]["labels"].astype(np.int).tolist()

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
        iou_thr=0.5,
        scale_ranges=None,
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


class HubDatasetCLass:
    def __init__(self, cfg):
        self.ds = dp.load(cfg.deeplake_path)
        labels_tensor = _find_tensor_with_htype(self.ds, "class_label")
        self.CLASSES = self.ds[labels_tensor].info.class_names
        self.pipeline = cfg.pipeline


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
    )  # we shouldn't be always doing this because some datasets can be in a different format


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
        raise ValueError("No tensor found with htype='{htype}'")
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
    masks = sample_in[masks_tensor]
    bboxes = sample_in[boxes_tensor]
    labels = sample_in[labels_tensor]

    img = img[..., ::-1]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    masks = masks.transpose((2, 0, 1)).astype(np.uint8)
    bboxes = _coco_2_pascal(bboxes)

    # transformed = rand_crop(
    #     image=img,
    #     masks=list(masks),
    #     bboxes=bboxes,
    #     bbox_ids=np.arange(len(bboxes)),
    #     labels=labels,
    # )  # why are we doing random crop?

    # img = transformed["image"].astype(np.uint8)
    shape = img.shape
    # labels = np.array(transformed["labels"], dtype=np.int64)
    # bboxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)

    # bbox_ids = transformed["bbox_ids"]
    # masks = [transformed["masks"][i] for i in transformed["bbox_ids"]]
    # if masks:
    #     masks = np.stack(masks, axis=0).astype(bool)
    # else:
    #     masks = np.empty((0, shape[0], shape[1]), dtype=bool)
    if isinstance(pipeline, list):
        pipeline = pipeline[0]

    return pipeline(
        {
            "img": img,
            "img_fields": ["img"],
            "filename": None,
            "ori_filename": None,
            "img_shape": shape,
            "ori_shape": shape,
            "gt_masks": BitmapMasks(masks, *shape[:2]),
            "gt_bboxes": bboxes,
            "gt_labels": labels,
            "bbox_fields": ["gt_bboxes"],
        }
    )


def build_dataset(cfg, *args, **kwargs):
    if "deeplake_path" in cfg:
        runner = CliRunner()
        username = "adilkhan"
        password = "142508Adoha"
        runner.invoke(login, f"-u {username} -p {password}")
        # TO DO: add preprocessing functions related to mmdet dataset classes like RepeatDataset etc...
        return HubDatasetCLass(cfg)
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
        loader = dataset.ds.pytorch(
            num_workers=num_workers,
            shuffle=shuffle,
            transform=transform_fn,
            tensors=[images_tensor, labels_tensor, boxes_tensor, masks_tensor],
            collate_fn=partial(
                collate, samples_per_gpu=train_loader_config["samples_per_gpu"]
            ),
            torch_dataset=MMDetDataset,
            # torch_dataset=TorchDataset,
        )
        loader.dataset.CLASSES = [
            c["name"] for c in dataset.ds.categories.info["category_info"]
        ]
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
    images_tensor: Optional[str] = None,
    masks_tensor: Optional[str] = None,
    boxes_tensor: Optional[str] = None,
    labels_tensor: Optional[str] = None,
):

    cfg = compat_cfg(cfg)
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
