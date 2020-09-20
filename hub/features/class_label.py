from typing import List

import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector


class ClassLabel(FeatureConnector):
    def __init__(
        self, num_classes: int = None, names: List[str] = None, names_file: str = None
    ):
        pass

    @property
    def names(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def int2str(self, int_: int) -> str:
        raise NotImplementedError()

    def str2int(self, name: str) -> int:
        raise NotImplementedError()
