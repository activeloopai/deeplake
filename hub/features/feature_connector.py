class FeatureConnector:
    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()
