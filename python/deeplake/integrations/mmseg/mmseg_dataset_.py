from collections import OrderedDict
import math
import numpy as np

from typing import Optional, Callable, Sequence
from torch.utils.data import Dataset
from prettytable import PrettyTable  # type: ignore

import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

from deeplake.integrations.mm.exceptions import InvalidImageError, InvalidSegmentError
from deeplake.integrations.mm.upcast_array import upcast_array
import time


class MMSegTorchDataset(Dataset):
    def __init__(
        self,
        dataset,
        tensors=None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.column_names = [col.name for col in self.dataset.schema.columns]
        self.last_successful_index = -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.dataset[idx]
                result = None
                if self.transform:
                    result = self.transform(sample)
                else:
                    out = {}
                    for col in self.column_names:
                        out[col] = sample[col]
                    result = out
                self.last_successful_index = idx
                return result
            except (InvalidImageError, InvalidSegmentError) as e:
                print(f"Error processing data at index {idx}: {e}")
                if self.last_successful_index == -1:
                    self.last_successful_index = idx + 1
                idx = self.last_successful_index
                continue


class MMSegDataset(MMSegTorchDataset):
    def __init__(
        self,
        *args,
        tensors_dict,
        mode="train",
        num_gpus=1,
        batch_size=1,
        ignore_index=255,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.masks_tensor_name = tensors_dict["masks_tensor"]
        if self.mode in ("val", "test"):
            self.CLASSES = self.get_classes(tensors_dict["masks_tensor"])[:]

    def __len__(self):
        if self.mode == "val":
            per_gpu_length = math.floor(
                len(self.dataset) / (self.batch_size * self.num_gpus)
            )
            total_length = per_gpu_length * self.num_gpus
            return total_length
        return super().__len__()

    def _get_masks(self, masks_tensor):
        if masks_tensor is None:
            return []
        return self.dataset[masks_tensor]

    def get_classes(self, classes):
        """Get class names of current dataset.

        Args:
            classes (str): Reresents the name of the classes tensor. Overrides the CLASSES defined by the dataset.

        Returns:
            list[str]: Names of categories of the dataset.
        """
        return self.dataset[classes].metadata["class_names"]

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                "DeprecationWarning: ``efficient_test`` has been deprecated "
                "since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory "
                "friendly by default. "
            )

        mask_col = self._get_masks(self.masks_tensor_name)
        last_successful_index = -1
        for idx in range(len(self)):
            try:
                result = upcast_array(mask_col[idx])
                last_successful_index = idx
                yield result
            except Exception as e:
                print(f"Error processing mask at index {idx}: {e}")
                if last_successful_index == -1:
                    continue
                else:
                    yield upcast_array(mask_col[last_successful_index])

    def evaluate(self, results, metric="mIoU", logger=None, gt_seg_maps=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        ..
            # noqa: DAR101

        Raises:
            KeyError: if a specified metric format is not supported

        Returns:
            dict[str, float]: Default metrics.
        """

        if self.num_gpus > 1:
            results_ordered = []
            for i in range(self.num_gpus):
                results_ordered += results[i :: self.num_gpus]
            results = results_ordered

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label,
            )
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {
                    key + "." + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                }
            )

        return eval_results
