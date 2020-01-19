from typing import Optional

class Base():
    def get(self, path: str) -> bytes:
        raise NotImplementedError

    def put(self, path: str, content: bytes):
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError
    
    def delete(self, path: str):
        raise NotImplementedError()

    def get_or_none(self, path: str) -> Optional[bytes]:
        if self.exists(path):
           return self.get(path)
        else:
            return None