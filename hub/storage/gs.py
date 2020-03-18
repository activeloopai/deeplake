from typing import *
from google.cloud.storage import Client, Bucket
from google.oauth2 import service_account

import json

from typing import Optional

from .base import Base

class GS(Base):
    _creds: service_account.Credentials = None
    _project: str = None
    _bucket: Bucket = None

    def __init__(self, bucket: str, creds_path: Optional[str] = None):
        super().__init__()
        if creds_path is not None:
            self._creds = service_account.Credentials.from_service_account_file(creds_path)
            with open(creds_path, 'rt') as f:
                self._project = json.loads(f.read())['project_id']
        
            self._bucket = Client(self._project, self._creds).bucket(bucket)
        else:
            self._bucket = Client().bucket(bucket)
        
    
    def get(self, path: str) -> bytes:
        return self._bucket.get_blob(path).download_as_string()

    def put(self, path: str, content: bytes):
        self._bucket.blob(path).upload_from_string(content)

    def exists(self, path: str) -> bool:
        return self._bucket.get_blob(path) is not None
    
    def delete(self, path: str):
        blobs = self._bucket.list_blobs(prefix=path)
        for blob in blobs:
            blob.delete()