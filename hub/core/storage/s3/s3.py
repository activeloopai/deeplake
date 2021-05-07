from hub.core.storage.provider import Provider
from hub.core.storage.s3.s3_mapper import S3Mapper
from typing import Optional


class S3Provider(Provider):
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
        self.mapper = S3Mapper(
            root,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            endpoint_url=endpoint_url,
        )
