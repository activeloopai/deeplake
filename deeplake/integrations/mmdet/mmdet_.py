from collections import OrderedDict

from typing import Callable, Optional, List, Dict
from mmdet.apis.train import *
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate
from functools import partial
from deeplake.integrations.pytorch.dataset import TorchDataset
from deeplake.client.client import DeepLakeBackendClient
from mmdet.core import BitmapMasks
import deeplake as dp
from deeplake.util.warnings import always_warn
import os.path as osp
import warnings
from collections import OrderedDict
import mmcv
from mmcv.runner import get_dist_info, init_dist
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



def coco_pixel_2_pascal_pixel(boxes, shape):
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


def poly_2_mask(polygons, shape):
    # TODO This doesnt fill the array inplace.
    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        im = Image.fromarray(out[..., i])
        d = ImageDraw.Draw(im)
        d.polygon(polygon, fill=1)
        out[..., i] = np.asarray(im)
    return out


def coco_frac_2_pascal_pixel(boxes, shape):
    x = boxes[:, 0] * shape[1]
    y = boxes[:, 1] * shape[0]
    w = boxes[:, 2] * shape[1]
    h = boxes[:, 3] * shape[0]
    bbox = np.stack((x, y, w, h), axis=1)
    return coco_pixel_2_pascal_pixel(bbox, shape)


def pascal_frac_2_pascal_pixel(boxes, shape):
    x_top = boxes[:, 0] * shape[1]
    y_top = boxes[:, 1] * shape[0]
    x_bottom = boxes[:, 2] * shape[1]
    y_bottom = boxes[:, 3] * shape[0]
    return np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)


def yolo_pixel_2_pascal_pixel(boxes, shape):
    x_top = np.array(boxes[:, 0]) - np.floor(np.array(boxes[:, 2]) / 2)
    y_top = np.array(boxes[:, 1]) - np.floor(np.array(boxes[:, 3]) / 2)
    x_bottom = np.array(boxes[:, 0]) + np.floor(np.array(boxes[:, 2]) / 2)

    y_bottom = np.array(boxes[:, 1]) + np.floor(np.array(boxes[:, 3]) / 2)

    return np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)


def yolo_frac_2_pascal_pixel(boxes, shape):
    x_center = boxes[:, 0] * shape[1]
    y_center = boxes[:, 1] * shape[0]
    width = boxes[:, 2] * shape[1]
    height = boxes[:, 3] * shape[0]
    bbox = np.stack((x_center, y_center, width, height), axis=1)
    return yolo_pixel_2_pascal_pixel(bbox, shape)


def get_bbox_format(bbox, bbox_info):
    bbox_info = bbox_info.get("coords", {})
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)


BBOX_FORMAT_TO_CONVERTER = {
    ("LTWH", "pixel"): coco_pixel_2_pascal_pixel,
    ("LTWH", "fractional"): coco_frac_2_pascal_pixel,
    ("LTRB", "pixel"): lambda x: x,
    ("LTRB", "fractional"): pascal_frac_2_pascal_pixel,
    ("CCWH", "pixel"): yolo_pixel_2_pascal_pixel,
    ("CCWH", "fractional"): yolo_frac_2_pascal_pixel,
}


def convert_to_pascal_format(bbox, bbox_info, shape):
    bbox_format = get_bbox_format(bbox, bbox_info)
    converter = BBOX_FORMAT_TO_CONVERTER[bbox_format]
    return converter(bbox, shape)


def coco_frac_2_pascal_pixel(boxes, shape):
    x = boxes[:, 0] * shape[1]
    y = boxes[:, 1] * shape[0]
    w = boxes[:, 2] * shape[1]
    h = boxes[:, 3] * shape[0]
    bbox = np.stack((x, y, w, h), axis=1)
    return coco_pixel_2_pascal_pixel(bbox, shape)


def pascal_frac_2_pascal_pixel(boxes, shape):
    x_top = boxes[:, 0] * shape[1]
    y_top = boxes[:, 1] * shape[0]
    x_bottom = boxes[:, 2] * shape[1]
    y_bottom = boxes[:, 3] * shape[0]
    return np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)


def yolo_pixel_2_pascal_pixel(boxes, shape):
    x_top = np.array(boxes[:, 0]) - np.floor(np.array(boxes[:, 2]) / 2)
    y_top = np.array(boxes[:, 1]) - np.floor(np.array(boxes[:, 3]) / 2)
    x_bottom = np.array(boxes[:, 0]) + np.floor(np.array(boxes[:, 2]) / 2)
    y_bottom = np.array(boxes[:, 1]) + np.floor(np.array(boxes[:, 3]) / 2)
    return np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)


def yolo_frac_2_pascal_pixel(boxes, shape):
    x_center = boxes[:, 0] * shape[1]
    y_center = boxes[:, 1] * shape[0]
    width = boxes[:, 2] * shape[1]
    height = boxes[:, 3] * shape[0]
    bbox = np.stack((x_center, y_center, width, height), axis=1)
    return yolo_pixel_2_pascal_pixel(bbox, shape)


def convert_to_pascal_format(bbox, bbox_info, shape):
    bbox_format = get_bbox_format(bbox, bbox_info)
    converter = BBOX_FORMAT_TO_CONVERTER[bbox_format]
    return converter(bbox, shape)


class MMDetDataset(TorchDataset):
    def __init__(
        self,
        *args,
        tensors_dict=None,
        mode="train",
        metrics_format="PascalVOC",
        bbox_info=None,
        pipeline=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bbox_info = bbox_info
        self.images = self._get_images(tensors_dict["images_tensor"])
        self.masks = self._get_masks(
            tensors_dict.get("masks_tensor", None), shape=self.images[0].shape
        )
        self.bboxes = self._get_bboxes(tensors_dict["boxes_tensor"])
        self.labels = self._get_labels(tensors_dict["labels_tensor"])
        self.iscrowds = self._get_iscrowds(tensors_dict.get("iscrowds"))
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
                iscrowds=self.iscrowds,
            )
        else:
            self.evaluator = None

    def _get_images(self, images_tensor):
        image_tensor = self.dataset[images_tensor]
        # image_list = [image.shape for image in image_tensor]
        return image_tensor

    def _get_masks(self, masks_tensor, shape):
        if masks_tensor is None:
            return []
        ret = self.dataset[masks_tensor].numpy(aslist=True)
        if self.dataset[masks_tensor].htype == "polygon":
            ret = map(partial(poly_2_mask, shape=shape), ret)
        return ret

    def _get_iscrowds(self, iscrowds_tensor):
        if iscrowds_tensor is not None:
            return iscrowds_tensor

        if "iscrowds" in self.dataset:
            always_warn(
                "Iscrowds was not specified, searching for iscrowds tensor in the dataset."
            )
            return self.dataset["iscrowds"].numpy(aslist=True)
        always_warn("iscrowds tensor was not found, setting its value to 0.")
        return iscrowds_tensor

    def _get_bboxes(self, boxes_tensor):
        return self.dataset[boxes_tensor].numpy(aslist=True)

    def _get_labels(self, labels_tensor):
        return self.dataset[labels_tensor].numpy(aslist=True)

    def _get_class_names(self, labels_tensor):
        return self.dataset[labels_tensor].info.class_names

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

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn("CustomDataset does not support filtering empty gt images.")
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

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
        return self.dataset[classes].info.class_names

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


def load_ds_from_cfg(cfg: Dict):
    creds = cfg.get("deeplake_credentials", {})
    token = creds.get("token", None)
    if token is None:
        uname = creds.get("username")
        if uname is not None:
            pword = creds["password"]
            client = DeepLakeBackendClient()
            token = client.request_auth_token(username=uname, password=pword)
    ds_path = cfg.deeplake_path
    ds = dp.load(ds_path, token=token)
    return ds


def _find_tensor_with_htype(ds: dp.Dataset, htype: str, mmdet_class=None):
    if mmdet_class is not None:
        always_warn(
            f"No deeplake tensor name specified for '{mmdet_class} in config. Fetching it using htype '{htype}'."
        )
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
    bbox_info: str,
    poly2mask: bool,
):
    img = sample_in[images_tensor]
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    bboxes = sample_in[boxes_tensor]
    bboxes = convert_to_pascal_format(bboxes, bbox_info, img.shape)
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
        elif masks.dtype != np.uint8:
            masks = masks.astype(np.uint8)
        masks = masks.transpose((2, 0, 1))
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


def build_dataset(cfg):
    if not isinstance(cfg, dp.Dataset):
        deeplake_tensors = cfg.get("deeplake_tensors", {})
        classes = deeplake_tensors.get("gt_labels")
        if classes is None:
            classes = _find_tensor_with_htype(cfg, "class_label", "gt_labels")

        if classes is None:
            raise KeyError("No tensor with htype class_label exists in the dataset")

        cfg = load_ds_from_cfg(cfg)
        cfg.CLASSES = cfg[classes].info.class_names
        return cfg
    return cfg


def _get_collate_keys(pipeline):
    if type(pipeline) == list:
        for transform in pipeline:
            if type(transform) == mmcv.utils.config.ConfigDict:
                keys = transform.get("keys") or _get_collate_keys(transform)
                if keys is not None:
                    return keys

    if type(pipeline) == mmcv.utils.config.ConfigDict:
        keys = pipeline.get("keys")
        if keys is not None:
            return keys

        for transform in pipeline:
            if type(pipeline[transform]) == list:
                keys = _get_collate_keys(pipeline[transform])
                if keys is not None:
                    return keys


def build_dataloader(
    dataset: List[dp.Dataset],
    images_tensor: str,
    masks_tensor: Optional[str],
    boxes_tensor: str,
    labels_tensor: str,
    implementation: str,
    pipeline: List,
    mode: str = "train",
    shuffle: Optional[bool] = None,
    **train_loader_config,
):
    if masks_tensor and "gt_masks" not in _get_collate_keys(pipeline):
        # TODO (adilkhan) check logic in _get_collate_keys
        # TODO (adilkhan) what if masks is not collected in the final step but required in intermediate steps?
        masks_tensor = None
    poly2mask = False
    if masks_tensor is not None:
        if dataset[masks_tensor].htype == "polygon":
            poly2mask = True
    bbox_info = dataset[boxes_tensor].info
    classes = dataset[labels_tensor].info.class_names
    dataset.CLASSES = classes
    pipeline = build_pipeline(pipeline)
    metrics_format = train_loader_config.get("metrics_format")
    dist = train_loader_config["dist"]
    if dist and implementation == "python":
        raise NotImplementedError(
            "Distributed training is not supported by the python data loader. Set deeplake_dataloader='c++' to use the C++ dtaloader instead."
        )
    transform_fn = partial(
        transform,
        images_tensor=images_tensor,
        masks_tensor=masks_tensor,
        boxes_tensor=boxes_tensor,
        labels_tensor=labels_tensor,
        pipeline=pipeline,
        bbox_info=bbox_info,
        poly2mask=poly2mask,
    )
    num_workers = train_loader_config["workers_per_gpu"] * train_loader_config.get(
        "num_gpus", 1
    )
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

    batch_size = train_loader_config["samples_per_gpu"] * train_loader_config.get(
        "num_gpus", 1
    )

    collate_fn = partial(collate, samples_per_gpu=batch_size)

    if implementation == "python":
        loader = dataset.pytorch(
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
            bbox_info=bbox_info,
        )

    else:
        loader = (
            dataloader(dataset)
            .transform(transform_fn)
            .shuffle(shuffle)
            .batch(batch_size)
            .pytorch(
                num_workers=num_workers,
                collate_fn=collate_fn,
                tensors=tensors,
                distributed=dist,
            )
        )

        # For DDP
        loader.sampler = None
        loader.batch_sampler = Dummy()
        loader.batch_sampler.sampler = None

        mmdet_ds = MMDetDataset(
            dataset=dataset,
            metrics_format=metrics_format,
            pipeline=pipeline,
            tensors_dict=tensors_dict,
            tensors=tensors,
            mode=mode,
            bbox_info=bbox_info,
        )
        loader.dataset = mmdet_ds
    loader.dataset.CLASSES = classes
    return loader


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
    train_dataset: dp.Dataset,
    cfg: mmcv.ConfigDict,
    validation_dataset: Optional[dp.Dataset] = None,
    distributed: bool = False,
    timestamp=None,
    meta=None,
    validate: bool = True,
):
    """
    Creates runner and trains the model:
    Args:
        model: model to train, should be built before passing
        train_dataset: dataset to train of type dp.Dataset
        cfg: mmcv.ConfigDict object containing all necessary configuration
        validation_dataset: validation dataset of type dp.Dataset
        distributed: bool, whether ddp training should be started, by default `False`
        timestamp: variable used in runner to make .log and .log.json filenames the same
        meta: meta data used to build runner
        validate: bool, whether validation should be conducted, by default `True`
    """
    cfg = compat_cfg(cfg)

    if isinstance(train_dataset, (list, tuple)):
        if len(train_dataset) > 1:
            raise NotImplementedError(
                "Training on multiple Deep Lake datasets is not supported."
            )
        train_dataset = train_dataset[0]

    if not hasattr(cfg, "gpu_ids"):
        if distributed:
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        else:
            cfg.gpu_ids = range(1)

    eval_cfg = cfg.get("evaluation", {})
    dl_impl = cfg.get("deeplake_dataloader", "auto").lower()

    if dl_impl == "auto":
        dl_impl = "c++" if indra_available() else "python"
    elif dl_impl == "cpp":
        dl_impl = "c++"

    if dl_impl not in {"c++", "python"}:
        raise ValueError(
            "`deeplake_dataloader` should be one of ['auto', 'c++', 'python']."
        )

    train_tensors = cfg.data.train.get("deeplake_tensors", {})
    if train_tensors:
        train_images_tensor = train_tensors["img"]
        train_boxes_tensor = train_tensors["gt_bboxes"]
        train_labels_tensor = train_tensors["gt_labels"]
        train_masks_tensor = train_tensors.get("gt_masks")
    else:
        train_images_tensor = _find_tensor_with_htype(train_dataset, "image", "img")
        train_boxes_tensor = _find_tensor_with_htype(train_dataset, "bbox", "gt_bboxes")
        train_labels_tensor = _find_tensor_with_htype(
            train_dataset, "class_label", "gt_labels"
        )
        train_masks_tensor = _find_tensor_with_htype(
            train_dataset, "binary_mask", "gt_masks"
        ) or _find_tensor_with_htype(train_dataset, "polygon", "gt_masks")

    metrics_format = eval_cfg.get("metrics_format", "PascalVOC")

    logger = get_root_logger(log_level=cfg.log_level)

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

    data_loader = build_dataloader(
        train_dataset,
        train_images_tensor,
        train_masks_tensor,
        train_boxes_tensor,
        train_labels_tensor,
        pipeline=cfg.get("train_pipeline", []),
        implementation=dl_impl,
        **train_loader_cfg,
    )

    # put model on gpus
    if distributed:
        init_dist("pytorch", **cfg.dist_params)
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
    if validation_dataset or validate:

        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
            mode="val",
            metrics_format=metrics_format,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get("val_dataloader", {}),
        }

        val_tensors = cfg.data.val.get("deeplake_tensors", {})

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = validation_dataset or build_dataset(cfg.data.val)

        if val_tensors:
            val_images_tensor = val_tensors["img"]
            val_boxes_tensor = val_tensors["gt_bboxes"]
            val_labels_tensor = val_tensors["gt_labels"]
            val_masks_tensor = val_tensors.get("gt_masks")
        else:
            val_images_tensor = _find_tensor_with_htype(val_dataset, "image", "img")
            val_boxes_tensor = _find_tensor_with_htype(val_dataset, "bbox", "gt_bboxes")
            val_labels_tensor = _find_tensor_with_htype(
                val_dataset, "class_label", "gt_labels"
            )
            val_masks_tensor = _find_tensor_with_htype(
                val_dataset, "binary_mask", "gt_masks"
            ) or _find_tensor_with_htype(val_dataset, "polygon", "gt_masks")

        val_dataloader = build_dataloader(
            val_dataset,
            val_images_tensor,
            val_masks_tensor,
            val_boxes_tensor,
            val_labels_tensor,
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
    runner.run([data_loader], cfg.workflow)
