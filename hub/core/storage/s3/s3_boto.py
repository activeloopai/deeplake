from hub.core.storage.provider import Provider
from hub.core.storage.s3.s3_boto_storage import S3BotoStorage
from typing import Optional


class S3BotoProvider(Provider):
    def __init__(
        self,
        root: str,
        aws_access_key_id: Optional[int] = None,
        aws_secret_access_key: Optional[int] = None,
        aws_session_token: Optional[int] = None,
        endpoint_url: Optional[int] = None,
        aws_region: Optional[int] = None,
    ):
        # passing no creds, would cause boto to read credentials
        self.mapper = S3BotoStorage(
            root,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            endpoint_url=endpoint_url,
        )

    def __iter__(self):
        yield from self.mapper.__iter__()

    def __len__(self):
        return len(self.mapper)
