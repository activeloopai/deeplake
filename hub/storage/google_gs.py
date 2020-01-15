from google.cloud import storage as gcs
from google.auth.credentials import Creds 
from typing import Optional

from .storage import Storage

class GoogleGS(Storage):
    def __init__(self, bucket: str):
        raise NotImplementedError()
    
    def get(self, path: str) -> bytes:
        raise NotImplementedError()

    def put(self, path: str, content: bytes):
        raise NotImplementedError()

    def exists(self, path: str) -> bool:
        raise NotImplementedError()

    def get_or_none(self, path: str) -> Optional[bytes]:
        if self.exists(path):
           return self.get(path)
        else:
            return None 
    
    def delete(self, path: str):
        raise NotImplementedError()