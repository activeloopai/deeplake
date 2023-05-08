METRIC_TO_ORDER_TYPE = {
    "l1": "ASC",
    "l2": "ASC",
    "cos": "DESC",
    "max": "ASC",
}


def get_order_type(distance_metric):
    return METRIC_TO_ORDER_TYPE[distance_metric]
