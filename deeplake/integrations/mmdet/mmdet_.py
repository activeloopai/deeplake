from collections import OrderedDict

from typing import Callable, Optional
from mmdet.apis.train import *
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate
from functools import partial
from typing import Optional
from deeplake.integrations.pytorch.dataset import TorchDataset
from deeplake.client.client import DeepLakeBackendClient
from mmdet.core import BitmapMasks
import deeplake as dp
from deeplake.util.warnings import always_warn
import os.path as osp
import warnings
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.pipelines import Compose
import tempfile
from deeplake.integrations.mmdet import mmdet_utils
from deeplake.experimental.dataloader import indra_available, dataloader
from PIL import Image, ImageDraw
import os


def coco_2_pascal(boxes, shape):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height

    return np.stack(
        (
            np.clip(boxes[:, 0], 0, None),
            np.clip(boxes[:, 1], 0, None),
            np.clip(boxes[:, 0] + np.clip(boxes[:, 2], 1, None), 0, shape[1]),
            np.clip(boxes[:, 1] + np.clip(boxes[:, 3], 1, None), 0, shape[0]),
        ),
        axis=1,
    )


def poly_2_mask(polygons, shape):
    # TODO This doesnt fill the array inplace.
    out = np.zeros((len(polygons),) + shape, dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        im = Image.fromarray(out[i])
        d = ImageDraw.Draw(im)
        d.polygon(polygon, fill=1)
        out[i] = np.asarray(im)


class MMDetDataset(TorchDataset):
    def __init__(
        self,
        *args,
        tensors_dict=None,
        mode="train",
        metrics_format="PascalVOC",
        pipeline=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.images = self._get_images(tensors_dict["images_tensor"])
        self.masks = self._get_masks(
            tensors_dict.get("masks_tensor", None), shape=self.images[0].shape
        )
        self.bboxes = self._get_bboxes(tensors_dict["boxes_tensor"])
        self.labels = self._get_labels(tensors_dict["labels_tensor"])
        self._classes = self.dataset[tensors_dict["labels_tensor"]].info.class_names
        self.CLASSES = self.get_classes(tensors_dict["labels_tensor"])
        self.mode = mode
        self.metrics_format = metrics_format

        if self.metrics_format == "COCO" and self.mode == "val":
            self.evaluator = mmdet_utils.COCODatasetEvaluater(
                pipeline,
                classes=self.CLASSES,
                hub_dataset=self.dataset,
                imgs=self.images,
                masks=self.masks,
                bboxes=self.bboxes,
                labels=self.labels,
            )
        else:
            self.evaluator = None

    def _get_images(self, images_tensor):
        image_tensor = self.dataset[images_tensor]
        image_list = [image.shape for image in image_tensor]
        return image_list

    def _get_masks(self, masks_tensor, shape):
        if masks_tensor is None:
            return []
        ret = self.dataset[masks_tensor].numpy(aslist=True)
        if self.dataset[masks_tensor].htype == "polygon":
            ret = map(partial(poly_2_mask, shape=shape), ret)
        return ret

    def _get_bboxes(self, boxes_tensor):
        return self.dataset[boxes_tensor].numpy(aslist=True)

    def _get_labels(self, labels_tensor):
        return self.dataset[labels_tensor].numpy(aslist=True)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Raises:
            ValueError: when ``self.metrics`` is not valid.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.metrics == "PascalVOC":
            bboxes = self.bboxes[idx]
        elif self.metrics == "COCO":
            bboxes = coco_2_pascal(self.bboxes[idx], self.images[idx].shape)
        else:
            raise ValueError(f"Bounding boxes in {self.metrics} are not supported")
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
        return self._classes

    def evaluate(
        self,
        results,
        metric="mAP",
        metrics_format="PascalVOC",
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
    def __init__(self, cfg=None, ds=None, tensors=None):
        if ds:
            self.ds = ds
        else:
            creds = cfg.get("deeplake_credentials", {})
            token = creds.get("token", None)
            if token is None:
                uname = creds.get("username")
                if uname is not None:
                    pword = creds["password"]
                    client = DeepLakeBackendClient()
                    token = client.request_auth_token(username=uname, password=pword)
            ds_path = cfg.deeplake_path
            self.ds = dp.load(ds_path, token=token)
        tensors = tensors or {}
        labels_tensor = tensors.get("gt_labels") or _find_tensor_with_htype(
            self.ds, "class_label"
        )
        self.CLASSES = self.ds[labels_tensor].info.class_names
        # self.pipeline = cfg.pipeline


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
    metrics_format: str,
    poly2mask: bool,
):
    img = sample_in[images_tensor]
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    bboxes = sample_in[boxes_tensor]
    if metrics_format == "COCO":
        bboxes = coco_2_pascal(bboxes, img.shape)
    labels = sample_in[labels_tensor]

    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img = img[..., ::-1]  # rgb_to_bgr should be optional
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    shape = img.shape

    if masks_tensor:
        masks = sample_in[masks_tensor]
        if poly2mask:
            masks = poly_2_mask(masks, shape)
        else:
            masks = masks.transpose((2, 0, 1)).astype(np.uint8)
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


def build_dataset(cfg, tensors=None, *args, **kwargs):
    if isinstance(cfg, dp.Dataset):
        return HubDatasetCLass(ds=cfg, tensors=tensors)
    if "deeplake_path" in cfg:
        # TO DO: add preprocessing functions related to mmdet dataset classes like RepeatDataset etc...
        return HubDatasetCLass(cfg=cfg)
    return mmdet_build_dataset(cfg, *args, **kwargs)


def build_dataloader(
    dataset,
    images_tensor,
    masks_tensor,
    boxes_tensor,
    labels_tensor,
    implementation,
    pipeline,
    mode="train",
    shuffle=None,
    **train_loader_config,
):
    if isinstance(dataset, dp.Dataset):
        dataset = HubDatasetCLass(ds=dataset, tensors={"gt_labels": labels_tensor})
    if isinstance(dataset, HubDatasetCLass):
        classes = dataset.CLASSES
        images_tensor = images_tensor or _find_tensor_with_htype(dataset.ds, "image")
        poly2mask = False
        masks_tensor = masks_tensor or _find_tensor_with_htype(
            dataset.ds, "binary_mask"
        )
        if masks_tensor is None:
            masks_tensor = _find_tensor_with_htype(dataset.ds, "polygon")
            if masks_tensor is not None:
                poly2mask = True
        elif dataset.ds[masks_tensor].htype == "polygon":
            poly2mask = True
        boxes_tensor = boxes_tensor or _find_tensor_with_htype(dataset.ds, "bbox")
        labels_tensor = labels_tensor or _find_tensor_with_htype(
            dataset.ds, "class_label"
        )
        pipeline = build_pipeline(pipeline)
        metrics_format = train_loader_config.get("metrics_format")
        transform_fn = partial(
            transform,
            images_tensor=images_tensor,
            masks_tensor=masks_tensor,
            boxes_tensor=boxes_tensor,
            labels_tensor=labels_tensor,
            pipeline=pipeline,
            metrics_format=metrics_format,
            poly2mask=poly2mask,
        )
        num_workers = train_loader_config["workers_per_gpu"]
        if shuffle is None:
            shuffle = train_loader_config.get("shuffle", True)
        tensors_dict = {
            "images_tensor": images_tensor,
            "boxes_tensor": boxes_tensor,
            "labels_tensor": labels_tensor,
        }
        tensors = [images_tensor, labels_tensor, boxes_tensor]
        if masks_tensor is not None:
            tensors.append(masks_tensor)
            tensors_dict["masks_tensor"] = masks_tensor

        batch_size = train_loader_config.get("samples_per_gpu", 1)

        collate_fn = partial(collate, samples_per_gpu=batch_size)

        if implementation == "python":
            loader = dataset.ds.pytorch(
                tensors_dict=tensors_dict,
                num_workers=num_workers,
                shuffle=shuffle,
                transform=transform_fn,
                tensors=tensors,
                collate_fn=collate_fn,
                torch_dataset=MMDetDataset,
                metrics_format=metrics_format,
                pipeline=pipeline,
                batch_size=batch_size,
                mode=mode,
                # torch_dataset=TorchDataset,
            )
            # loader.dataset.CLASSES = [
            #     c["name"] for c in dataset.ds.categories.info["category_info"]
            # ]
        else:
            loader = (
                dataloader(dataset.ds)
                .transform(transform_fn)
                .shuffle(shuffle)
                .batch(batch_size)
                .pytorch(
                    num_workers=num_workers, collate_fn=collate_fn, tensors=tensors
                )
            )
            mmdet_ds = MMDetDataset(
                dataset=dataset.ds,
                metrics_format=metrics_format,
                pipeline=pipeline,
                tensors_dict=tensors_dict,
                tensors=tensors,
                mode=mode,
            )
            loader.dataset = mmdet_ds
        loader.dataset.CLASSES = classes
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
    dataloader: str = None,
    metrics_format=None,
    shuffle=None,
):

    cfg = compat_cfg(cfg)
    eval_cfg = cfg.get("evaluation", {})
    dl_impl = dataloader or cfg.get("deeplake_dataloader", "auto").lower()

    if dl_impl == "auto":
        dl_impl = "c++" if indra_available() else "python"
    elif dl_impl == "cpp":
        dl_impl = "c++"

    if dl_impl not in {"c++", "python"}:
        raise ValueError(
            "`deeplake_dataloader` should be one of ['auto', 'c++', 'python']."
        )

    tensors = cfg.get("deeplake_tensors", {})
    images_tensor = images_tensor or tensors.get("img")
    masks_tensor = masks_tensor or tensors.get("gt_masks")
    boxes_tensor = boxes_tensor or tensors.get("gt_bboxes")
    labels_tensor = labels_tensor or tensors.get("gt_labels")

    metrics_format = eval_cfg.get("metrics_format", "PascalVOC")

    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=cfg.data.get("samples_per_gpu", 256),
        workers_per_gpu=cfg.data.get("workers_per_gpu", 8),
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
        metrics_format=metrics_format,
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
            pipeline=cfg.get("train_pipeline", []),
            implementation=dl_impl,
            shuffle=shuffle,
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
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
            mode="val",
            metrics_format=metrics_format,
            shuffle=False,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get("val_dataloader", {}),
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, tensors=tensors)

        val_dataloader = build_dataloader(
            val_dataset,
            images_tensor,
            masks_tensor,
            boxes_tensor,
            labels_tensor,
            pipeline=cfg.get("test_pipeline", []),
            implementation=dl_impl,
            **val_dataloader_args,
        )
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
