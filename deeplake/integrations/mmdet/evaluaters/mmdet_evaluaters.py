from typing import Union
from deeplake.integrations.mmdet import mmdet_utils
from matterport.deeplake.deeplake.integrations.mmdet.evaluaters.Pascal import (
    pascal_evaluater,
)
from deeplake.util.exceptions import UnsupportedMMDetMetric


METRIC_FORMAT_TO_EVALUATER_TYPE = {
    "COCO": mmdet_utils.COCODatasetEvaluater,
    "PascalVOC": pascal_evaluater.Evaluator,
}


def create_metric_class(
    metric: str,
) -> Union[mmdet_utils.COCODatasetEvaluater, pascal_evaluater.Evaluator]:
    """
    Creates a metric evaluator for a specified metric.

    Args:
        metric (str): The name of the metric to evaluate.

    Returns:
        Union[mmdet_utils.COCODatasetEvaluater, pascal_evaluater.Evaluator]: An evaluator for the specified metric.

    Raises:
        UnsupportedMMDetMetric: If the specified metric is not supported.
    """
    if metric in METRIC_FORMAT_TO_EVALUATER_TYPE:
        return METRIC_FORMAT_TO_EVALUATER_TYPE[metric]
    raise UnsupportedMMDetMetric(metric)
