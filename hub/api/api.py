from typing import Optional

from .hub_bucket import HubBucket

def AmazonS3(bucket: str, aws_creds_filepath: Optional[str], aws_key_id: Optional[str], aws_secret_key: Optional[str]) -> HubBucket:
    raise NotImplementedError()

# def GoogleGS(bucket: str, ...)

def Filesystem(dir: str) -> HubBucket:
    raise NotImplementedError()

# bucket = hub.AmazonS3() 
# bucket.array_create()