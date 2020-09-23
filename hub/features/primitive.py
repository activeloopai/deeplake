import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector


class Primitive(FeatureConnector):
    def __init__(self, dtype: str):
        self.dtype = dtype

    def _flatten(self):
        return "", (), self.dtype
