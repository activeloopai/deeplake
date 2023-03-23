from typing import List, Tuple, Dict, Any

import deeplake
import recall_evaluater
import map_evaluater


class Evaluator:
    """
    Evaluator class that wraps MAPEvaluator and RecallEvaluator.
    """

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(
        self,
        dataset: deeplake.Dataset,
        results: List[Dict[str, Any]],
        num_gpus: int = 1,
        metric: str = "mAP",
        logger=None,
        proposal_nums: Tuple[int, int, int] = (100, 300, 1000),
        iou_thr: float = 0.5,
        scale_ranges=None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate the dataset.

        Args:
            dataset: The dataset to be evaluated.
            results: The testing results of the dataset.
            num_gpus: The number of GPUs used in evaluation. Default is 1.
            metric: The metric to be evaluated, either "mAP" or "recall". Default is "mAP".
            logger: The logger used for printing related information during evaluation. Default is None.
            proposal_nums: A tuple of proposal numbers used for evaluating recalls, such as recall@100, recall@1000.
                           Default is (100, 300, 1000).
            iou_thr: The IoU threshold. Default is 0.5.
            scale_ranges: The scale ranges for evaluating mAP. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            OrderedDict: The evaluation metrics dictionary.
        """

        if metric == "mAP":
            evaluator = map_evaluater.MAPEvaluator(
                dataset, results, num_gpus, logger, proposal_nums, iou_thr, scale_ranges
            )
        elif metric == "recall":
            evaluator = recall_evaluater.RecallEvaluator(
                dataset, results, num_gpus, logger, proposal_nums, iou_thr, scale_ranges
            )
        else:
            raise KeyError(f"metric {metric} is not supported")

        return evaluator.evaluate(**kwargs)
