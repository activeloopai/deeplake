import boto3
import botocore

from .storage import Storage

class AmazonS3(Storage):
    def __init__(self, bucket: str, aws_access_key_id: str, aws_secret_access_key: str):
        super().__init__()
        self._bucket = boto3.resource('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key).Bucket(bucket)
    
    def get(self, path: str) -> bytes:
        return self._bucket.Object(path).get()['Body'].read()

    def put(self, path: str, content: bytes) -> None:
            self._bucket.Object(path).put({'Body': content, 'ContentType': ('application/octet-stream')})
    
    def exists(self, path: str) -> bool:
        try:
            self._bucket.Object(path).load()
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            else:
                raise ex
        return True