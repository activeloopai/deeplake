import math
import os.path as osp
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from deeplake.integrations.mmdet.evaluaters import mmdet_evaluaters
from deeplake.integrations.mmdet.utils import mmdataset_info
from deeplake.integrations.pytorch.dataset import TorchDataset
from deeplake.util.warnings import always_warn
from deeplake.integrations.mmdet.converters import (
    coco_format,
    pascal_format,
    bbox_format,
)

from mmdet.utils.util_distribution import *  # type: ignore


class MMDetDataset(TorchDataset):
    """
    A PyTorch dataset class for handling data in MMDetection format.

    Args:
        *args: Variable length argument list.
        tensors_dict (dict): A dictionary containing the image, label, and bounding box tensors.
        mode (str): The mode of the dataset. Can be "train", "val", or "test".
        metrics_format (str): The format of metrics used to evaluate the dataset. Can be "COCO" or "PASCAL_VOC".
        bbox_info (dict): A dictionary containing information about the bounding boxes.
        pipeline (list): A list of preprocessing transformations.
        num_gpus (int): The number of GPUs to use for training/validation.
        batch_size (int): The batch size to use for training/validation.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dataset_info (mmdataset_info.DatasetInfo): The dataset information.
        mode (str): The mode of the dataset. Can be "train", "val", or "test".
        pipeline (list): A list of preprocessing transformations.
        num_gpus (int): The number of GPUs to use for training/validation.
        batch_size (int): The batch size to use for training/validation.
        bbox_info (dict): A dictionary containing information about the bounding boxes.
        images (list): A list of image tensors.
        masks (list): A list of mask tensors.
        bboxes (list): A list of bounding box tensors.
        bbox_format (str): The format of the bounding boxes.
        labels (list): A list of label tensors.
        iscrowds (list): A list of iscrowd values.
        CLASSES (list): A list of class names.
        metrics_format (str): The format of metrics used to evaluate the dataset. Can be "COCO" or "PASCAL_VOC".
        coco_style_bbox (list): A list of bounding boxes in COCO format.
        evaluator (mmdet_evaluaters.Evaluater): The evaluator used to evaluate the dataset.
    """

    def __init__(
        self,
        *args: Any,
        tensors_dict: Optional[Dict[str, Any]] = None,
        mode: str = "train",
        metrics_format: str = "COCO",
        bbox_info: Optional[Dict[str, Any]] = None,
        pipeline: Optional[List[Any]] = None,
        num_gpus: int = 1,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MMDetDataset.

        Args:
            *args: Variable length argument list.
            tensors_dict (dict): A dictionary containing the image, label, and bounding box tensors.
            mode (str): The mode of the dataset. Can be "train", "val", or "test".
            metrics_format (str): The format of metrics used to evaluate the dataset. Can be "COCO" or "PASCAL_VOC".
            bbox_info (dict): A dictionary containing information about the bounding boxes.
            pipeline (list): A list of preprocessing transformations.
            num_gpus (int): The number of GPUs to use for training/validation.
            batch_size (int): The batch size to use for training/validation.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if mode in ("val", "test"):
            self._initialize_mmdet_related_attributes(
                tensors_dict,
                bbox_info,
                metrics_format,
                pipeline,
                mode,
                num_gpus,
                batch_size,
            )
            self._initialize_metrics()
            self._initialize_dataset_info()

    def _initialize_dataset_info(self) -> None:
        """
        Initialize the dataset infor class that's used when print() is called on this class.
        """
        self.dataset_info = mmdataset_info.DatasetInfo(
            mode=self.mode,
            classes=self.CLASSES,
        )

    def _initialize_mmdet_related_attributes(
        self,
        tensors_dict: Dict[str, Any],
        bbox_info: Dict[str, Any],
        metrics_format: str,
        pipeline: List[Any],
        mode: str,
        num_gpus: int,
        batch_size: int,
    ) -> None:
        """
        Initialize the MMDetection related attributes.

        Args:
            tensors_dict (dict): A dictionary containing the image, label, masks and bounding box tensors.
            bbox_info (dict): A dictionary containing information about the bounding boxes.
            metrics_format (str): The format of metrics used to evaluate the dataset. Can be "COCO" or "PASCAL_VOC".
            pipeline (list): A list of preprocessing transformations.
            mode (str): The mode of the dataset. Can be "train", "val", or "test".
            num_gpus (int): The number of GPUs to use for training/validation.
            batch_size (int): The batch size to use for training/validation.
        """
        self.mode = mode
        self.pipeline = pipeline
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.bbox_info = bbox_info
        self.images = self._get_images(tensors_dict["images_tensor"])
        self.masks = self._get_masks(tensors_dict.get("masks_tensor", None))
        self.bboxes = self._get_bboxes(tensors_dict["boxes_tensor"])
        self.bbox_format = bbox_format.get_bbox_format(
            bbox_format.first_non_empty(self.bboxes), bbox_info
        )
        self.labels = self._get_labels(tensors_dict["labels_tensor"])
        self.iscrowds = self._get_iscrowds(tensors_dict.get("iscrowds"))
        self.CLASSES = self.get_classes(tensors_dict["labels_tensor"])
        self.metrics_format = metrics_format
        self.coco_style_bbox = coco_format.convert(
            self.bboxes, self.bbox_format, self.images
        )

    def _initialize_metrics(self) -> None:
        """
        Initialize the evaluator used during evaluation.
        """
        Evaluater = mmdet_evaluaters.create_metric_class(self.metrics_format)
        self.evaluator = Evaluater(
            pipeline=self.pipeline,
            classes=self.CLASSES,
            deeplake_dataset=self.dataset,
            imgs=self.images,
            masks=self.masks,
            bboxes=self.coco_style_bbox,
            labels=self.labels,
            iscrowds=self.iscrowds,
            bbox_format=self.bbox_format,
            num_gpus=self.num_gpus,
        )

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self.mode == "val":
            per_gpu_length = math.floor(
                len(self.dataset) / (self.batch_size * self.num_gpus)
            )
            total_length = per_gpu_length * self.num_gpus
            return total_length
        return super().__len__()

    def _get_images(self, images_tensor: str) -> Any:
        """
        Return the images from the dataset.

        Args:
            images_tensor (str): The name of the tensor containing the images.

        Returns:
            deeplae.core.Tensor: The image tensor from the dataset.
        """
        image_tensor = self.dataset[images_tensor]
        return image_tensor

    def _get_masks(self, masks_tensor: Optional[str]) -> List[Any]:
        """
        Return the masks from the dataset.

        Args:
            masks_tensor (str): The name of the tensor containing the masks.

        Returns:
            List[ deeplae.core.Tensor]: The mask tensor from the dataset.
        """
        if masks_tensor is None:
            return []
        return self.dataset[masks_tensor]

    def _get_iscrowds(self, iscrowds_tensor: Optional[str]) -> Union[List[int], None]:
        """
        Return the iscrowd values from the dataset.

        Args:
            iscrowds_tensor (str): The name of the tensor containing the iscrowd values.

        Returns:
            Union[List[int], None]: The iscrowd values from the dataset.
        """
        if iscrowds_tensor is not None:
            return iscrowds_tensor

        if "iscrowds" in self.dataset:
            always_warn(
                "Iscrowds was not specified, searching for iscrowds tensor in the dataset."
            )
            return self.dataset["iscrowds"].numpy(aslist=True)
        always_warn("iscrowds tensor was not found, setting its value to 0.")
        return iscrowds_tensor

    def _get_bboxes(self, boxes_tensor: str) -> List[np.ndarray]:
        """
        Return the bounding boxes from the dataset.

        Args:
            boxes_tensor (str): The name of the tensor containing the bounding boxes.

        Returns:
            List[np.ndarray]: The bounding boxes from the dataset.
        """
        return self.dataset[boxes_tensor].numpy(aslist=True)

    def _get_labels(self, labels_tensor: str) -> List[np.ndarray]:
        """
        Return the labels from the dataset.

        Args:
            labels_tensor (str): The name of the tensor containing the labels.

        Returns:
            List[np.ndarray]: The labels from the dataset.
        """
        return self.dataset[labels_tensor].numpy(aslist=True)

    def _get_class_names(self, labels_tensor: str) -> List[str]:
        """
        Return the class names from the dataset.

        Args:
            labels_tensor (str): The name of the tensor containing the labels.

        Returns:
            List[str]: The class names from the dataset.
        """
        return self.dataset[labels_tensor].info.class_names

    def get_ann_info(self, idx: int) -> Dict[str, Any]:
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Raises:
            ValueError: when ``self.metrics`` is not valid.

        Returns:
            dict: Annotation info of specified index.
        """
        bboxes = pascal_format.convert(
            self.bboxes[idx], self.bbox_info, self.images[idx].shape
        )
        return {"bboxes": bboxes, "labels": self.labels[idx]}

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = self.labels[idx].astype(np.int).tolist()

        return cat_ids

    def _filter_imgs(self, min_size: int = 32) -> List[int]:
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn("CustomDataset does not support filtering empty gt images.")
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_classes(self, classes: str) -> List[str]:
        """Get class names of current dataset.

        Args:
            classes (str): Reresents the name of the classes tensor. Overrides the CLASSES defined by the dataset.

        Returns:
            list[str]: Names of categories of the dataset.
        """
        return self.dataset[classes].info.class_names

    def _reorder_results(
        self, results: List[Union[Tuple, np.ndarray]]
    ) -> List[Union[Tuple, np.ndarray]]:
        """Reordering results after ddp so that eval occurs correctly

        Args:
            results (list[tuple | numpy.ndarray]): results produced after evaluating model
        """
        if self.num_gpus > 1:
            results_ordered = []
            for i in range(self.num_gpus):
                results_ordered += results[i :: self.num_gpus]
            results = results_ordered
        return results

    def evaluate(
        self,
        results: List[Union[Tuple, np.ndarray]],
        metric: Union[str, List[str]] = "mAP",
        logger: Optional[Union[logging.Logger, str]] = None,
        proposal_nums: Sequence[int] = (100, 300, 1000),
        iou_thr: Union[float, List[float]] = 0.5,
        scale_ranges: Optional[List[Tuple]] = None,
        **kwargs: Any,
    ) -> OrderedDict:
        """Evaluate the dataset.

        Args:
            **kwargs (dict): Keyword arguments to pass to self.evaluate object
            results (list[tuple | numpy.ndarray]): Testing results of the dataset.
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
        results = self._reorder_results(results)
        return self.evaluator.evaluate(
            results,
            metric=metric,
            logger=logger,
            proposal_nums=proposal_nums,
            iou_thr=iou_thr,
            scale_ranges=scale_ranges,
            **kwargs,
        )

    def __repr__(self) -> self:
        """Print the number of instance number."""
        return self.dataset_info

    def format_results(
        self,
        results: List[Union[Tuple, np.ndarray]],
        jsonfile_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[tempfile.TemporaryDirectory]]:
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
