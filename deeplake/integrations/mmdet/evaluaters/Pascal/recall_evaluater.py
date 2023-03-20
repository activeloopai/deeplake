from collections import OrderedDict
from typing import List, Dict, Any

from matterport.deeplake.deeplake.integrations.mmdet.evaluaters.Pascal import base_evaluater

from mmdet.core import eval_recalls


class RecallEvaluator(base_evaluater.BaseEvaluator):
    """
    Recall Evaluator class.
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
        eval_results = self.calculate_recall(self.results, annotations, iou_thrs)

        return eval_results

    def calculate_recall(
        self, results: List[Any], annotations: List[Any], iou_thrs: List[float]
    ) -> Dict[str, float]:
        """
        Calculates recall for given results, annotations, and IoU thresholds.

        Args:
            results: The results of the evaluation.
            annotations: The annotations for the dataset.
            iou_thrs: List of IoU thresholds.

        Returns:
            Dict[str, float]: A dictionary containing the recall results.
        """
        eval_results = OrderedDict()
        gt_bboxes = [ann["bboxes"] for ann in annotations]
        recalls = eval_recalls(
            gt_bboxes, results, self.proposal_nums, iou_thrs, logger=self.logger
        )
        for i, num in enumerate(self.proposal_nums):
            for j, iou in enumerate(iou_thrs):
                eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
        if recalls.shape[1] > 1:
            ar = recalls.mean(axis=1)
            for i, num in enumerate(self.proposal_nums):
                eval_results[f"AR@{num}"] = ar[i]
        return eval_results
