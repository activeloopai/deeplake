from .base import Base

class Recursive(Base):
    _core: Base = None
    _wrapper: Base = None

    def __init__(self, core: Base, wrapper: Base):
        super().__init__()
        self._core = core
        self._wrapper = wrapper

    def get(self, path: str) -> bytes:
        ans = self._wrapper.get_or_none(path) 
        if ans is None:
            ans = self._core.get(path)
            self._wrapper.put(path, ans)
            return ans
        else:
            return ans

    def put(self, path: str, content: bytes):
        self._core.put(path, content)
        self._wrapper.put(path, content)

    def exists(self, path: str) -> bool:
        return self._wrapper.exists(path) or self._core.exists(path)
    
    def delete(self, path: str):
        self._wrapper.delete(path)
        self._core.delete(path)