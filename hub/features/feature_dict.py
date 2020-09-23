from typing import Dict, List, Union

import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector
RawFeatureConnector = Union[FeatureConnector, str, dict]


class FeatureDict(FeatureConnector):
    def __init__(self, dict_: Dict[str, RawFeatureConnector]):
        dict_ = {
            key: FeatureDict(value) if isinstance(value, dict) else value
            for key, value in dict_.items()
        }
        self.dict_ = dict_

    def _flatten(self):
        return [
            (f"/{key}{item[0]}", item[1], item[2])
            for item in value._flatten()
            for key, value in self.dict_.items()
        ]
