from .api.hub_backend import HubBackend
from .storage import Storage

class HubBackendImpl(HubBackend):
    _storage: Storage = None

    def __init__(self, storage: Storage):
        self._storage = storage

    @property
    def storage(self):
        return self._storage