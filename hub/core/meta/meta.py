import hub
from hub.core.storage.cachable import Cachable


class Meta(Cachable):
    def __init__(self):
        self.version = hub.__version__

    def as_dict(self) -> dict:
        return {"version": self.version}
