from typing import Union
from .COCO import coco_evaluater
from .Pascal import pascal_evaluater
from deeplake.util.exceptions import UnsupportedMMDetMetric


METRIC_FORMAT_TO_EVALUATER_TYPE = {
    "COCO": coco_evaluater.Evaluater,
    "PascalVOC": pascal_evaluater.Evaluator,
}


def create_metric_class(
    metric: str,
) -> Union[coco_evaluater.Evaluater, pascal_evaluater.Evaluator]:
    """
    Creates a metric evaluator for a specified metric.

    Args:
        metric (str): The name of the metric to evaluate.

    Returns:
        Union[coco_evaluater.Evaluater, pascal_evaluater.Evaluator]: An evaluator for the specified metric.

    Raises:
        UnsupportedMMDetMetric: If the specified metric is not supported.
    """
    if metric in METRIC_FORMAT_TO_EVALUATER_TYPE:
        return METRIC_FORMAT_TO_EVALUATER_TYPE[metric]
    raise UnsupportedMMDetMetric(metric)
