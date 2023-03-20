from collections import OrderedDict
from typing import List, Dict, Any

from deeplake.integrations.mmdet.evaluaters import base_evaluater

from mmdet.core import eval_map
from mmdet.utils import print_log


class MAPEvaluator(base_evaluater.BaseEvaluator):
    """
    Mean Average Precision Evaluator class.
    """
    def evaluate(self) -> Dict[str, float]:
        if self.num_gpus > 1:
            results_ordered = []
            for i in range(self.num_gpus):
                results_ordered += self.results[i :: self.num_gpus]
            self.results = results_ordered

        annotations = [self.dataset.get_ann_info(i) for i in range(len(self.dataset))]
        eval_results = OrderedDict()
        iou_thrs = [self.iou_thr] if isinstance(self.iou_thr, float) else self.iou_thr
        eval_results = self.calculate_mAP(self.results, annotations, iou_thrs)

        return eval_results

    def calculate_mAP(self, results: List[Any], annotations: List[Any], iou_thrs: List[float]) -> Dict[str, float]:
        """
        Calculates mAP for given results, annotations, and IoU thresholds.

        Args:
            results: The results of the evaluation.
            annotations: The annotations for the dataset.
            iou_thrs: List of IoU thresholds.

        Returns:
            Dict[str, float]: A dictionary containing the mAP results.
        """
        eval_results = OrderedDict()
        mean_aps = []
        for iou_thr in iou_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                dataset=self.dataset.CLASSES,
                logger=self.logger,
            )
            mean_aps.append(mean_ap)
            eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
        eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
        return eval_results
