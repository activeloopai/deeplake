from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional, Dict, Any

import deeplake


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation classes.

    Args:
        dataset: The dataset to be evaluated.
        results: The results of the evaluation.
        num_gpus (int, optional): The number of GPUs to use.
        logger (optional): Logger to log the evaluation process.
        proposal_nums (tuple[int, int, int], optional): Tuple containing the proposal numbers.
        iou_thr (float, optional): The Intersection over Union (IoU) threshold.
        scale_ranges (optional): Scale ranges for the evaluation.
    """

    def __init__(
        self,
        dataset: deeplake.Dataset,
        results: List[Any],
        num_gpus: int = 1,
        logger=None,
        proposal_nums: Tuple[int, int, int] = (100, 300, 1000),
        iou_thr: float = 0.5,
        scale_ranges=None,
    ):
        self.dataset = dataset
        self.results = results
        self.num_gpus = num_gpus
        self.logger = logger
        self.proposal_nums = proposal_nums
        self.iou_thr = iou_thr
        self.scale_ranges = scale_ranges

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, float]:
        """
        Abstract method to be implemented by the child classes for evaluation.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation results.
        """
        pass
