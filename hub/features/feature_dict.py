from typing import Dict

import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector


class FeatureDict(FeatureConnector):
    def __init__(self, dict_: Dict[str, any]):
        pass
