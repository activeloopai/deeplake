from typing import Optional

from .hub_bucket import HubBucket

class HubBackend():
    def s3(self, bucket: Optional[str] = None, aws_creds_filepath: Optional[str] = None, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None) -> 'HubBackend':
        raise NotImplementedError()

    def fs(self, dir: str) -> 'HubBackend':
        raise NotImplementedError()

    def client(self) -> 'HubBucket':
        raise NotImplementedError()