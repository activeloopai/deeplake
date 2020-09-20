from typing import Tuple

import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector


class Tensor(FeatureConnector):
    def __init__(self, shape: Tuple[int, ...] = None, dtype: str = None):
        pass
