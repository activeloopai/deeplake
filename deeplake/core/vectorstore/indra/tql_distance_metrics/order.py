METRIC_TO_ORDER_TYPE = {
    "l1": "DESC",
    "l2": "DESC",
    "cos": "ASC",
    "max": "DESC",
}


def get_order_type(distance_metric):
    return METRIC_TO_ORDER_TYPE[distance_metric]
