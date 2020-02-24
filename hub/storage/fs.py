import os
import shutil

from hub.config import CACHE_FILE_PATH
from hub.exceptions import FileSystemException

from .base import Base

class FS(Base):
    def __init__(self, dir: str):
        super().__init__()
        self._dir = dir

    def get(self, path: str) -> bytes:
        path = os.path.join(self._dir, path)        
        with open(path, 'rb') as f:
            return f.read()

    def put(self, path: str, content: bytes):
        path = os.path.join(self._dir, path)
        folder = '/'.join(path.split('/')[:-1])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        with open(path, 'wb') as f:
            f.write(content)

    def exists(self, path: str) -> bool:
        path = os.path.join(self._dir, path)
        return os.path.exists(path)

    def delete(self, path: str):
        path = os.path.join(self._dir, path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)