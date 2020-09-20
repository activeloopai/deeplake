import hub.features.feature_connector as feature_connector

FeatureConnector = feature_connector.FeatureConnector


class Sequence(FeatureConnector):
    def __init__(feature: FeatureConnector, length: int = None):
        pass

    @property
    def feature(self):
        raise NotImplementedError()
