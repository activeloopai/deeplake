from .storage import Storage

class RecursiveStorage(Storage):
    _base: Storage = None
    _curr: Storage = None

    def __init__(self, current_storage: Storage, base_storage: Storage):
        self._curr = current_storage
        self._base = base_storage

    def get(self, path: str) -> bytes:
        ans = self._curr.get_or_none(path) 
        if ans is None:
            ans = self._base.get(path)
            self._curr.put(path, ans)
            return ans
        else:
            return ans

    def put(self, path: str, content: bytes):
        self._base.put(path, content)
        self._curr.put(path, content)

    def exists(self, path: str) -> bool:
        return self._curr.exists(path) or self._base.exists(path)
    
    def delete(self, path: str):
        self._curr.delete(path)
        self._base.delete(path)